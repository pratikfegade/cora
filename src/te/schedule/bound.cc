/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file bound.cc
 * \brief The bound inference logic.
 */
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/uf_equality.h>

#include <unordered_map>
#include <unordered_set>

#include "../../arith/interval_set.h"
#include "../../runtime/thread_storage_scope.h"
#include "../../tir/ir/var_replacer.h"
#include "../operation/op_util.h"
#include "graph.h"
#include "message_passing.h"
#include "schedule_utils.h"

namespace tvm {
namespace te {

using runtime::StorageRank;
using runtime::StorageScope;
using runtime::ThreadScope;

IntSet inlineUFunCalls(IntSet s) {
  if (s.as<tvm::arith::IntervalSetNode>()) {
    return tvm::arith::IntSet::interval(Simplify(UninterpFun::InlineUninterpFunCalls(s.min())),
                                        Simplify(UninterpFun::InlineUninterpFunCalls(s.max())));
  }
  return s;
}

/*! \brief The graph context used during bound inference. */
struct GraphContext {
  /*! \brief The feed graph */
  FeedGraph feed_graph;
  /*! \brief Attachment path */
  Map<Operation, Array<IterVar>> attach_path;
  /*! \brief Attachment path oeprations */
  Map<Operation, Array<Operation>> attach_path_ops;
  /*! \brief The bind map */
  std::unordered_map<IterVar, IterVar> bind_map;
  /*! \brief map from op to stage */
  std::unordered_map<const Object*, Stage> op2stage_;
  // /*! \brief map storing mapping from cached to original ops for
  //     equality purposes. */
  // Map<FunctionRef, CacheInfo> cacheTensorInfos;
};

InferBoundsResult InferBoundsResultNode::make(Map<IterVar, Range> bounds,
                                              Map<Stage, Map<std::string, Range>> env_bounds,
                                              Map<Stage, Map<std::string, IterVar>> env_vars) {
  ObjectPtr<InferBoundsResultNode> n = make_object<InferBoundsResultNode>();
  n->bounds = bounds;
  n->env_bounds = env_bounds;
  n->env_vars = env_vars;
  return InferBoundsResult(n);
}

class Simplifier : public ExprMutator {
 public:
  Simplifier(const Schedule& sch) {
    struct FuncTriple {
      UninterpFun fused_to_outer_uf;
      UninterpFun fused_to_inner_uf;
      UninterpFun outer_inner_to_fused_uf;
    };

    std::unordered_map<const Object*, int> fused_fun_map;
    int count = 0;
    auto handle_rel = [&](Dimension fused_dim, UninterpFun fused_to_outer_uf,
                          UninterpFun fused_to_inner_uf, UninterpFun outer_inner_to_fused_uf) {
      auto it = fused_fun_map.find(fused_dim.get());
      if (it != fused_fun_map.end()) {
        auto& id = it->second;
        equivalences[fused_to_outer_uf.get()] = id;
        equivalences[fused_to_inner_uf.get()] = id;
        equivalences[outer_inner_to_fused_uf.get()] = id;
      } else {
        fused_fun_map[fused_dim.get()] = count++;
      }
    };

    // Process stages in sorted order so that global stages are
    // processed before shared stages which in turn are processed before
    // local stages. This ensures that the most exhaustive fusion
    // function is selected for generation in case stages can share
    // fusion functions

    std::vector<Stage> stages;
    for (auto s : sch->stages) {
      stages.push_back(s);
    }
    struct less_than_stage {
      inline bool operator()(const Stage& stage1, const Stage& stage2) {
        auto r1 = (stage1->scope.length() == 0) ? runtime::StorageRank::kGlobal
                                                : runtime::StorageScope::make(stage1->scope).rank;
        auto r2 = (stage2->scope.length() == 0) ? runtime::StorageRank::kGlobal
                                                : runtime::StorageScope::make(stage2->scope).rank;
        return (r1 < r2);
      }
    };
    std::sort(stages.begin(), stages.end(), less_than_stage());

    for (auto s : stages) {
      // std::cout << "[Stage] Stage " << s << std::endl;
      for (auto rel : s->relations) {
        if (auto frel = rel.as<RaggedFuseNode>()) {
          handle_rel(frel->fused_to_outer_uf->dimensions[0], frel->fused_to_outer_uf,
                     frel->fused_to_inner_uf, frel->outer_inner_to_fused_uf);
        }
      }
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) override {
    if (auto func_node = op->func.as<UninterpFunNode>()) {
      if (func_node->type == UninterpFunNode::kFOFun ||
          func_node->type == UninterpFunNode::kFIFun) {
        CHECK_EQ(op->args.size(), 1);
        auto fused_val = op->args[0];
        auto fused_val_call_node = fused_val.as<CallNode>();
        if (fused_val_call_node) {
          auto fused_val_callee_node = fused_val_call_node->func.as<UninterpFunNode>();
          // std::cout << "[RS] Simplifying " << GetRef<PrimExpr>(op) << std::endl;
          // std::cout << "[RS]   " << equivalences[fused_val_callee_node] << " "
          //           << equivalences[func_node] << std::endl;
          if (fused_val_callee_node && fused_val_callee_node->type == UninterpFunNode::kOIFFun &&
              equivalences[fused_val_callee_node] == equivalences[func_node]) {
            auto ret = func_node->type == UninterpFunNode::kFOFun ? fused_val_call_node->args[0]
                                                                  : fused_val_call_node->args[1];
            // std::cout << "[RS]   To " << ret << std::endl;
            return this->VisitExpr(ret);
          }
        }
      } else if (func_node->type == UninterpFunNode::kOIFFun) {
        CHECK_EQ(op->args.size(), 2);
        auto outer_val = op->args[0];
        auto inner_val = op->args[1];
        auto outer_val_call_node = outer_val.as<CallNode>();
        auto inner_val_call_node = inner_val.as<CallNode>();
        if (outer_val_call_node && inner_val_call_node) {
          auto outer_val_callee_node = outer_val_call_node->func.as<UninterpFunNode>();
          auto inner_val_callee_node = inner_val_call_node->func.as<UninterpFunNode>();
          // std::cout << "[RS] Simplifying " << GetRef<PrimExpr>(op) << std::endl;
          // std::cout << "[RS]   " << equivalences[outer_val_callee_node] << " "
          //           << equivalences[func_node] << std::endl;
          // std::cout << "[RS]   " << equivalences[inner_val_callee_node] << " "
          //           << equivalences[func_node] << std::endl;
          if (outer_val_callee_node && inner_val_callee_node &&
              outer_val_callee_node->type == UninterpFunNode::kFOFun &&
              inner_val_callee_node->type == UninterpFunNode::kFIFun &&
              equivalences[inner_val_callee_node] == equivalences[func_node] &&
              equivalences[outer_val_callee_node] == equivalences[func_node]) {
            CHECK_EQ(outer_val_call_node->args.size(), 1);
            CHECK_EQ(inner_val_call_node->args.size(), 1);
            if (ExprEquality()(outer_val_call_node->args[0], inner_val_call_node->args[0])) {
              auto ret = outer_val_call_node->args[0];
              // std::cout << "[RS]   To " << ret << std::endl;
              return this->VisitExpr(ret);
            }
          }
        }
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  IntSet Simplify(IntSet& set) {
    if (set.as<arith::IntervalSetNode>()) {
      PrimExpr min = this->VisitExpr(set.min());
      // std::cout << " " << set.max() << std::endl;
      PrimExpr max = this->VisitExpr(set.max());
      IntSet ret = IntSet::interval(min, max);
      const_cast<arith::IntSetNode*>(ret.operator->())
          ->set_fusion_fields(set->f_fun, set->fused_range);
      return ret;
    } else {
      return set;
    }
  }

  Range Simplify(Range& r) {
    PrimExpr min = this->VisitExpr(r->min);
    PrimExpr extent = this->VisitExpr(r->extent);
    Range ret = Range::make_by_min_extent(min, extent);
    const_cast<RangeNode*>(ret.operator->())->set_fusion_fields(r->f_fun, r->fused_range);
    return ret;
  }

 private:
  std::unordered_map<const Object*, int> equivalences;
};

bool NeedRelax(const IterVar& iv, bool found_attach,
               const std::unordered_map<IterVar, IterVar>& bind_map,
               const runtime::StorageScope& scope, bool print) {
  auto it = bind_map.find(iv);
  const std::string& tag = (it != bind_map.end() ? it->second->thread_tag : iv->thread_tag);
  if (tag.length() == 0 || tag == "pipeline") {
    if (print) std::cout << "[NRLX]      1" << std::endl;
    return !found_attach;
  }
  ThreadScope ts = ThreadScope::make(tag);

  // When there is warp memory
  // threadIdx.x must be set to be warp index.
  if (scope.rank == StorageRank::kWarp && ts.rank == 1 && ts.dim_index == 0) {
    if (print) std::cout << "[NRLX]      2" << std::endl;
    return true;
  }
  if (print)
    std::cout << "[NRLX]      3 " << scope.to_string() << " " << tag << " "
              << (static_cast<int>(scope.rank) <= ts.rank) << std::endl;
  return static_cast<int>(scope.rank) <= ts.rank;
}

// infer storage scope, if not given
StorageScope InferStorageScope(const Stage& stage, const GraphContext& ctx) {
  if (stage->scope.length() != 0) {
    StorageScope s = StorageScope::make(stage->scope);
    const_cast<Stage&>(stage)->storage_scope_rank = static_cast<int>(s.rank);
    // std::cout << "[Scope] 1 " << stage << " " << s.to_string() << std::endl;
    return s;
  }
  int max_rank = -1;
  for (IterVar iv : ctx.attach_path.at(stage->op)) {
    auto it = ctx.bind_map.find(iv);
    const std::string& tag = (it != ctx.bind_map.end() ? it->second->thread_tag : iv->thread_tag);
    if (tag != "pipeline" && tag.length() != 0) {
      max_rank = std::max(max_rank, ThreadScope::make(tag).rank);
    }
  }
  StorageScope s;
  s.rank = runtime::DefaultStorageRank(max_rank);
  const_cast<Stage&>(stage)->storage_scope_rank = static_cast<int>(s.rank);
  // std::cout << "[Scope] 2 " << stage << " " << s.to_string() << std::endl;
  return s;
}

Range TranslateIterVarsFromConsumerToProducer(Range range, Operation consumer, Operation producer) {
  const BaseVarDimOpNode* c = GetBaseVarDimOp(consumer);
  const BaseVarDimOpNode* p = GetBaseVarDimOp(producer);

  if (c == nullptr || p == nullptr) return range;

  std::unordered_map<const VarNode*, PrimExpr> vsub;
  for (const auto& dim2var_map : c->dim2var_maps) {
    for (const auto& it : dim2var_map) {
      auto dim = it.first;
      auto var_node = it.second.iv->var.as<VarNode>();

      // if (p->dim2var_maps[tensor->value_index].count(dim)) {
      // vsub[var_node] = p->dim2var_maps[tensor->value_index].at(dim).iv->var;
      // }
      if (p->dim2var_maps[0].count(dim)) {
        vsub[var_node] = p->dim2var_maps[0].at(dim).iv->var;
      }
    }
  }

  VarReplacer replacer(vsub);
  return Range::make_by_min_extent(replacer(range->min), replacer(range->extent));
  // return range;
}

void CollectIterVarMappingFromConsumerToProducer(
    Operation consumer, Operation producer, std::unordered_map<const VarNode*, PrimExpr>* p_map) {
  const BaseVarDimOpNode* c = GetBaseVarDimOp(consumer);
  const BaseVarDimOpNode* p = GetBaseVarDimOp(producer);

  if (c == nullptr || p == nullptr) return;

  std::unordered_map<const VarNode*, PrimExpr> vsub;
  for (const auto& dim2var_map : c->dim2var_maps) {
    for (const auto& it : dim2var_map) {
      auto dim = it.first;
      auto var_node = it.second.iv->var.as<VarNode>();

      if (p->dim2var_maps[0].count(dim)) {
        (*p_map)[var_node] = p->dim2var_maps[0].at(dim).iv->var;
      }
    }
  }
}

bool MarkedNoRelax(const Stage& stage, const GraphContext& ctx, IterVar iv) {
  bool print = false;  //(stage->op->name == "r_gate.ila");
  if (print) std::cout << "[NORELAX]     " << iv << std::endl;
  for (auto miv : stage->no_relax_ivs) {
    if (iv == miv) {
      if (print) std::cout << "[NORELAX2]     " << iv << std::endl;
      return true;
    }
    if (ctx.bind_map.count(iv)) {
      if (equalCudaThreads(miv, ctx.bind_map.at(iv))) {
        if (print) std::cout << "[NORELAX4]     " << iv << std::endl;
        return true;
      }
    }
  }
  return false;
}

bool MarkedNoRelax(const Stage& stage, std::string name) {
  bool print = false;  //(stage->op->name == "r_gate.ila");
  if (print) std::cout << "[NORELAX]     " << name << std::endl;
  for (auto miv : stage->no_relax_ivs) {
    if (isCudaThread(miv) && miv->var->name_hint == name) {
      if (print) std::cout << "[NORELAX4]     " << name << std::endl;
      return true;
    }
  }
  return false;
}

void InferRootBound(const Stage& stage, const GraphContext& ctx, Simplifier& simplifier,
                    std::unordered_map<IterVar, Range>* rmap) {
  CHECK_NE(stage->attach_type, kInline) << "call schedule.normalize before scheduleops";
  if (stage->attach_type == kInlinedAlready) return;
  if (stage->is_output) {
    // verify correctness.
    CHECK_EQ(stage.GetAttachSpec()->attach_type, kGroupRoot) << "Output must be attached at root";
  }
  if (stage->is_output || stage->op.as<PlaceholderOpNode>()) {
    for (auto iv : stage->op->root_iter_vars()) {
      CHECK(iv->dom.defined());
      CHECK(!rmap->count(iv)) << iv << " " << stage;
      (*rmap)[iv] = iv->dom;
    }
    return;
  }
  // The tensor domain.
  std::unordered_map<Tensor, TensorDom> tmap;
  // The consumers of the op.
  std::unordered_set<Operation> consumers;
  for (int i = 0; i < stage->op->num_outputs(); ++i) {
    Tensor t = stage->op.output(i);
    tmap.emplace(t, TensorDom(static_cast<int>(t.ndim())));
    auto it = ctx.feed_graph.find(t);
    if (it != ctx.feed_graph.end()) {
      for (const Operation& op : it->second) {
        consumers.insert(op);
      }
    } else {
      LOG(INFO) << t << " " << i << " not found in the feed graph = " << stage->op;
    }
  }
  // storage scope.
  runtime::StorageScope scope = InferStorageScope(stage, ctx);
  // Bound prop by other consumers.
  // - Compute bound by relaxation rules: NeedRelax
  //   - For normal index, use relative location of loop nest./
  //   - For thread index, use the thread scope.
  //
  Array<IterVar> stage_attach = ctx.attach_path.at(stage->op);

  // bool print = false;
  bool print = (stage->op->name == "A.shared");
  // The parent set.
  for (const Operation& op : consumers) {
    if (print) std::cout << "[IRB] " << stage->op->name << std::endl;
    std::unordered_map<const VarNode*, IntSet> relax_set;
    std::unordered_map<IterVar, IntSet> up_state;
    bool found_attach = false;
    CHECK(ctx.op2stage_.count(op.get())) << op << " " << stage->op;
    const Stage& op_stage = ctx.op2stage_.at(op.get());
    if (print) std::cout << "[IRB] Consumer " << op << " " << op_stage << std::endl;
    std::unordered_map<const VarNode*, PrimExpr> consumer_to_producer_vsub;
    /************************* Phase 1 *************************/
    // Consumer nest
    CollectIterVarMappingFromConsumerToProducer(op, stage->op, &consumer_to_producer_vsub);

    for (size_t i = op_stage->leaf_iter_vars.size(); i != 0; --i) {
      IterVar iv = op_stage->leaf_iter_vars[i - 1];
      if (stage_attach.size() != 0 && iv == stage_attach[0]) {
        found_attach = true;
      }
      auto it = rmap->find(iv);
      CHECK(it != rmap->end()) << iv->var << " " << iv.get();
      const Range& vrange = it->second;

      IterVarAttr it_attr;
      if (op_stage->iter_var_attrs.count(iv)) {
        it_attr = op_stage->iter_var_attrs[iv];
      }

      if (print)
        std::cout << "[IRB]  LV " << iv << " " << iv->iter_type << " " << it_attr << std::endl;
      if (is_one(vrange->extent)) {
        up_state[iv] = IntSet::single_point(vrange->min);
        if (print) std::cout << "[IRB]    upb1 " << iv << " " << up_state[iv] << std::endl;
      } else if (!NeedRelax(iv, found_attach, ctx.bind_map, scope, false) &&
                 /* If an IV is opaque to loop nest creation, it means
                    we would not have a loop corresponding to such an
                    IV and so it doesn't make sense to not relax */
                 iv->iter_type != kLoopNestOpaque &&
                 (iv->iter_type != kSplit &&
                  (!it_attr.defined() || it_attr->iter_type != kSplit))) {
        CHECK(is_zero(vrange->min)) << "InferBound requires every leaf iter var's min equals 0, "
                                    << " call schedule.normalize to achieve this. " << vrange << " "
                                    << iv << " " << op_stage->op;

        up_state[iv] = IntSet::single_point(iv->var);
        if (print) std::cout << "[IRB]    upb2 " << iv << " " << up_state[iv] << std::endl;
      } else if (MarkedNoRelax(stage, ctx, iv)) {
        up_state[iv] = IntSet::single_point(iv->var);
        if (print) std::cout << "[IRB]    upb3 " << iv << " " << up_state[iv] << std::endl;
      } else {
        up_state[iv] = IntSet::range(vrange);
        if (print) std::cout << "[IRB]    upb4 " << iv << " " << up_state[iv] << std::endl;
      }
    }
    // Consumer's attach nest
    for (size_t i = 0; i < ctx.attach_path.at(op).size(); ++i) {
      IterVar iv = ctx.attach_path.at(op)[i];
      Operation iv_op = ctx.attach_path_ops.at(op)[i];
      if (stage_attach.size() != 0 && iv == stage_attach[0]) {
        found_attach = true;
      }
      Range vrange = rmap->at(iv);

      CollectIterVarMappingFromConsumerToProducer(iv_op, stage->op, &consumer_to_producer_vsub);
      // vrange = TranslateIterVarsFromConsumerToProducer(vrange, iv_op, stage->op);
      CHECK(is_zero(vrange->min)) << "InferBound requires every leaf iter var's min equals 0, "
                                  << "call schedule.normalize to achieve this. " << vrange << " "
                                  << iv;
      if (print)
        std::cout << "[RLX]    Try relax " << iv << " " << iv_op << " " << found_attach << " "
                  << scope.to_string() << std::endl;
      if (NeedRelax(iv, found_attach, ctx.bind_map, scope, false) &&
          !MarkedNoRelax(stage, ctx, iv)) {
        if (print) std::cout << "[RLX]      Relaxed " << vrange << std::endl;
        relax_set[iv->var.get()] = IntSet::range(vrange);
        if (ctx.bind_map.count(iv)) {
          relax_set[ctx.bind_map.at(iv)->var.get()] = IntSet::range(vrange);
          if (print) std::cout << "[RLX]       BindRelaxed " << ctx.bind_map.at(iv) << std::endl;
        }
      }
    }
    CHECK(found_attach || stage_attach.size() == 0)
        << "Invalid Schedule, cannot find the producer " << stage->op
        << " along the loop nest specified by compute_at of consumer " << op;

    /************************* Phase 2 *************************/
    // Get the domain of the consumer
    PassUpDomain(op_stage, *rmap, &up_state);
    // Relax if needed.
    std::unordered_map<const VarNode*, IntSet> dom_map;
    arith::Analyzer analyzer;

    for (auto iv : op->root_iter_vars()) {
      Range orig_range;
      if (up_state.count(iv)) {
        orig_range = up_state.at(iv).cover_range(iv->dom);
        if (print) std::cout << "[IRB]    upa1 " << iv->var << " " << orig_range << std::endl;
      } else {
        orig_range = iv->dom;
        if (print) std::cout << "[IRB]    upa2 " << iv->var << " " << orig_range << std::endl;
      }

      if (print) {
        if (orig_range->f_fun.defined()) {
          std::cout << "[IRB] Found " << orig_range->f_fun << " " << orig_range->fused_range
                    << std::endl;
        }
      }

      if (relax_set.size() != 0) {
        auto vars = VarCollector().collect(orig_range);
        std::unordered_map<std::string, const VarNode*> name_var_map;
        for (auto var : vars) {
          name_var_map[var->name_hint] = var;
        }

        std::unordered_map<const VarNode*, IntSet> relax_set_updated;
        std::unordered_set<std::string> to_relax_env_threads;
        for (auto it : relax_set) {
          relax_set_updated[it.first] = it.second;
          if (isCudaThread(it.first->name_hint) || isCPUEnvThread(it.first->name_hint))
            to_relax_env_threads.insert(it.first->name_hint);
        }

        std::unordered_map<const VarNode*, std::string> bind_rmap;
        for (auto it : ctx.bind_map) {
          bind_rmap[it.first->var.as<VarNode>()] = it.second->var->name_hint;
        }

        for (auto var : vars) {
          if (bind_rmap.count(var)) {
            auto name = bind_rmap.at(var);
            if (to_relax_env_threads.count(name) && !MarkedNoRelax(stage, name)) {
              Range r = NullValue<Range>();
              for (auto it : *rmap)
                if (it.first->var.get() == var) r = it.second;
              if (r.defined()) {
                if (print)
                  std::cout << "[IRB]       RSU " << var->name_hint << " " << r << std::endl;
                relax_set_updated[var] = IntSet::range(r);
              }
            }
          }
        }

        Range r = Range::make_by_min_extent(arith::Simplify(orig_range->min),
                                            arith::Simplify(orig_range->extent));
        if (print) {
          std::cout << "[IRB] WEFVAEWGVWERSGWSGVW#E$RTGVDFVSDBFRGBNRSTN0  START " << iv->var
                    << std::endl;
        }
        const_cast<RangeNode*>(r.as<RangeNode>())
            ->set_fusion_fields(orig_range->f_fun, orig_range->fused_range);
        dom_map[iv->var.get()] = EvalSet(r, relax_set_updated, rmap);
        if (print) {
          std::cout << "[IRB] WEFVAEWGVWERSGWSGVW#E$RTGVDFVSDBFRGBNRSTN0  END " << iv->var
                    << std::endl;
          std::cout << "[IRB]     dom1 " << dom_map[iv->var.get()] << std::endl;
        }
      } else {
        dom_map[iv->var.get()] = IntSet::range(orig_range);
        if (print) std::cout << "[IRB]     dom2 " << dom_map[iv->var.get()] << std::endl;
      }
      dom_map[iv->var.get()] =
          arith::ReplaceIntSet(dom_map[iv->var.get()], consumer_to_producer_vsub);
      dom_map[iv->var.get()] = simplifier.Simplify(dom_map[iv->var.get()]);
      analyzer.Bind(iv->var, VarReplacer(consumer_to_producer_vsub).replace(orig_range));
    }
    /************************* Phase 3 *************************/
    op->PropBoundToInputs(op, &analyzer, dom_map, &tmap);
  }
  // if (stage->op->name == "A.shared") {
  // exit(0);
  // }
  /************************* Phase 4 *************************/
  stage->op->GatherBound(stage->op, tmap, rmap, {});
}

InferBoundsResult InferBound(const Schedule& sch) {
  CheckSchedule(const_cast<Schedule&>(sch), "bound.cc:238");

  // Prepare context
  GraphContext ctx;
  Array<Operation> roots;
  arith::Analyzer analyzer;

  for (Operation op : sch->outputs) {
    roots.push_back(sch->stage_map[op]->op);
  }
  ctx.feed_graph = CreateFeedGraph(CreateReadGraph(roots, false));
  // TODO: Mighty mighty global variable hack
  tir::UfBodyEquality::cacheTensorInfos = sch->cacheTensorInfos;

  for (Stage stage : sch->stages) {
    for (auto kv : stage->iter_var_attrs) {
      if (kv.second->bind_thread.defined()) {
        CHECK(!ctx.bind_map.count(kv.first));
        ctx.bind_map[kv.first] = kv.second->bind_thread;
      }
    }
    ctx.op2stage_[stage->op.get()] = stage;
  }
  auto path_ops = CreateAttachPath(sch);
  ctx.attach_path = path_ops.first;
  ctx.attach_path_ops = path_ops.second;

  Simplifier simplifier(sch);

  // Run inference.
  std::unordered_map<IterVar, Range> ret;
  Map<Stage, Map<std::string, Range>> env_bounds;
  Map<Stage, Map<std::string, IterVar>> env_vars;
  for (size_t i = sch->stages.size(); i != 0; --i) {
    const Stage& stage = sch->stages[i - 1];

    std::unordered_map<std::string, Range> op_env_bounds;
    std::unordered_map<std::string, IterVar> op_env_vars;
    Operation op = stage->op;
    for (auto iv : ctx.attach_path.at(op)) {
      IterVar enviv = NullValue<IterVar>();
      if (isCudaThread(iv) || isCPUEnvThread(iv)) {
        enviv = iv;
      } else if (ctx.bind_map.count(iv)) {
        enviv = ctx.bind_map.at(iv);
      }

      if (enviv.defined()) {
        Range r = NullValue<Range>();

        if (ret.count(enviv)) {
          r = ret.at(enviv);
        } else if (enviv->dom.defined()) {
          r = enviv->dom;
        }
        op_env_bounds[enviv->var->name_hint] = r;
        op_env_vars[enviv->var->name_hint] = enviv;
      }
    }

    for (auto iv : stage->env_threads) {
      IterVar enviv = NullValue<IterVar>();
      if (isCudaThread(iv) || isCPUEnvThread(iv)) {
        enviv = iv;
      } else if (ctx.bind_map.count(iv)) {
        enviv = ctx.bind_map.at(iv);
      }

      if (enviv.defined()) {
        Range r = NullValue<Range>();

        if (ret.count(enviv)) {
          r = ret.at(enviv);
        } else if (enviv->dom.defined()) {
          r = enviv->dom;
        }
        op_env_bounds[enviv->var->name_hint] = r;
        op_env_vars[enviv->var->name_hint] = enviv;
      }
    }

    env_bounds.Set(stage, op_env_bounds);
    env_vars.Set(stage, op_env_vars);

    for (IterVar iv : stage->env_threads) {
      CHECK(iv->dom.defined());
      ret[iv] = iv->dom;
      analyzer.Bind(iv->var, iv->dom);
    }

    InferRootBound(stage, ctx, simplifier, &ret);

    // bind bound of root iter vars.
    for (auto iv : stage->op->root_iter_vars()) {
      auto it = ret.find(iv);
      if (it != ret.end()) {
        analyzer.Bind(iv->var, it->second);
      }
    }

    // pass down to get bound of all iter vars.
    PassDownDomain(stage, &ret, &analyzer);

    for (IterVar iv : stage->env_threads) {
      CHECK(iv->dom.defined()) << iv;
      ret[iv] = iv->dom;
    }

    for (auto iv : stage->all_iter_vars) {
      ret[iv] = simplifier.Simplify(ret[iv]);
    }
  }

  // bind bound of thread bound vars as PassDownDomain skips them.
  std::unordered_set<IterVar> updated;
  for (Stage stage : sch->stages) {
    for (auto kv : stage->iter_var_attrs) {
      if (kv.second->bind_thread.defined() && !updated.count(kv.second->bind_thread)) {
        CHECK(ret.count(kv.second->bind_thread));
        analyzer.Bind(kv.second->bind_thread->var, ret[kv.second->bind_thread]);
        updated.insert(kv.second->bind_thread);
      }
    }
  }

  auto mutable_sch = const_cast<Schedule&>(sch);
  mutable_sch->InvalidateCache();
  mutable_sch->InitCache();

  auto set_storage_rank = [&](const Stage& stage, const Array<Tensor>& inputs) {
    for (auto input : inputs) {
      auto input_stage = mutable_sch->op2stage_cache_.at(input->op.get());
      input_stage->storage_scope_rank = stage->storage_scope_rank;
    }
  };

  for (Stage stage : sch->stages) {
    Operation op = stage->op;
    if (auto scan_op = op.as<ScanOpNode>()) {
      set_storage_rank(stage, scan_op->update);
    } else if (auto sk_op = op.as<SingleKernelEnvelopeOpNode>()) {
      set_storage_rank(stage, sk_op->inputs);
    } else if (auto c_op = op.as<ConditionalOpNode>()) {
      set_storage_rank(stage, c_op->then_case);
      set_storage_rank(stage, c_op->else_case);
    }
  }

  return InferBoundsResultNode::make(Map<IterVar, Range>(ret.begin(), ret.end()), env_bounds,
                                     env_vars);
}

TVM_REGISTER_GLOBAL("schedule.InferBound").set_body_typed(InferBound);

}  // namespace te
}  // namespace tvm
