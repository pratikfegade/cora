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

bool MarkedNoRelax(const Stage& stage, std::string thread_tag) {
  bool print = false;  //(stage->op->name == "r_gate.ila");
  if (print) std::cout << "[NORELAX]     " << thread_tag << std::endl;
  for (auto miv : stage->no_relax_ivs) {
    if (isCudaThread(miv) && miv->thread_tag == thread_tag) {
      if (print) std::cout << "[NORELAX4]     " << thread_tag << std::endl;
      return true;
    }
  }
  return false;
}

void InferRootBound(const Stage& stage, const GraphContext& ctx,
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
      // std::cout << "[IRB] Stage " << stage << " " << iv << std::endl;
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

  bool print = false;
  // bool print = (stage->op->name == "S");
  // The parent set.
  for (const Operation& op : consumers) {
    if (print) std::cout << "[IRB] " << stage->op->name << std::endl;
    std::unordered_map<const IterVarNode*, IntSet> relax_set;
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
      } else if (!NeedRelax(iv, found_attach, ctx.bind_map, scope, print) &&
                 /* If an IV is opaque to loop nest creation, it means
                    we would not have a loop corresponding to such an
                    IV and so it doesn't make sense to not relax */
                 iv->iter_type != kLoopNestOpaque &&
                 (iv->iter_type != kSplit &&
                  (!it_attr.defined() || it_attr->iter_type != kSplit))) {
        // std::cout << "[IRB] Checking " << op_stage->op << " " << iv << " " << vrange->min
        // << std::endl;
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
      // if (print)
      //   std::cout << "[RLX]    Try relax " << iv << " " << iv_op << " " << found_attach << " "
      //             << scope.to_string() << std::endl;
      if (NeedRelax(iv, found_attach, ctx.bind_map, scope, false) &&
          !MarkedNoRelax(stage, ctx, iv)) {
        // if (print) std::cout << "[RLX]      Relaxed " << vrange << std::endl;
        relax_set[iv.operator->()] = IntSet::range(vrange);
        if (ctx.bind_map.count(iv)) {
          relax_set[ctx.bind_map.at(iv).operator->()] = IntSet::range(vrange);
          // if (print) std::cout << "[RLX]       BindRelaxed " << ctx.bind_map.at(iv) << std::endl;
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
      Range r;
      if (up_state.count(iv)) {
        r = up_state.at(iv).cover_range(iv->dom);
        if (print) std::cout << "[IRB]    upa1 " << iv->var << " " << up_state.at(iv) << std::endl;
      } else {
        r = iv->dom;
        if (print) std::cout << "[IRB]    upa2 " << iv->var << " " << r << std::endl;
      }
      if (relax_set.size() != 0) {
        auto vars = VarCollector().collect(r);
        std::unordered_map<const VarNode*, IntSet> relax_set_updated;
        std::unordered_set<std::string> to_relax_env_threads;
        for (auto it : relax_set) {
          relax_set_updated[it.first->var.get()] = it.second;
          if (isCudaThread(it.first->thread_tag) || isCPUEnvThread(it.first->thread_tag))
            to_relax_env_threads.insert(it.first->thread_tag);
        }

        std::unordered_map<const VarNode*, std::string> bind_rmap;
        for (auto it : ctx.bind_map) {
          bind_rmap[it.first->var.as<VarNode>()] = it.second->thread_tag;
        }

        for (auto var : vars) {
          if (bind_rmap.count(var)) {
            auto thread_tag = bind_rmap.at(var);
            if (to_relax_env_threads.count(thread_tag) && !MarkedNoRelax(stage, thread_tag)) {
              Range r1 = NullValue<Range>();
              for (auto it : *rmap)
                if (it.first->var.get() == var) r1 = it.second;
              if (r1.defined()) {
                // if (print)
                // std::cout << "[IRB]       RSU " << var->name_hint << " " << r1 << std::endl;
                relax_set_updated[var] = IntSet::range(r1);
              }
            }
          }
        }

        r = Range::make_by_min_extent(arith::Simplify(r->min), arith::Simplify(r->extent));
        if (print) {
          std::cout << "[IRB] WEFVAEWGVWERSGWSGVW#E$RTGVDFVSDBFRGBNRSTN0  START " << iv
                    << std::endl;
          std::cout << "[IRB]  ToRelax " << r << std::endl;
          for (auto it : relax_set_updated) {
            std::cout << "[IRB]   ToRelax " << it.first->name_hint << " " << it.second << std::endl;
          }
        }
        dom_map[iv->var.get()] = EvalSet(r, relax_set_updated, rmap);
        if (print) {
          std::cout << "[IRB]  Relaxed " << dom_map[iv->var.get()] << std::endl;
          std::cout << "[IRB] WEFVAEWGVWERSGWSGVW#E$RTGVDFVSDBFRGBNRSTN0  END " << iv << std::endl;
        }
      } else {
        dom_map[iv->var.get()] = IntSet::range(r);
        if (print) std::cout << "[IRB]     dom2 " << dom_map[iv->var.get()] << std::endl;
      }

      //////////////////////////////////////////////////////////////////
      dom_map[iv->var.get()] =
          arith::ReplaceIntSet(dom_map[iv->var.get()], consumer_to_producer_vsub);
      analyzer.Bind(iv->var, VarReplacer(consumer_to_producer_vsub).replace(r));
      // analyzer.Bind(iv->var, r);
      //////////////////////////////////////////////////////////////////
    }
    /************************* Phase 3 *************************/
    op->PropBoundToInputs(op, &analyzer, dom_map, &tmap);
  }
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
        // std::cout << "[IRB] BindIV " << enviv->var << " " << enviv->thread_tag << std::endl;
        op_env_bounds[enviv->thread_tag] = r;
        op_env_vars[enviv->thread_tag] = enviv;
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
        op_env_bounds[enviv->thread_tag] = r;
        op_env_vars[enviv->thread_tag] = enviv;
      }
    }

    env_bounds.Set(stage, op_env_bounds);
    env_vars.Set(stage, op_env_vars);

    for (IterVar iv : stage->env_threads) {
      CHECK(iv->dom.defined());
      ret[iv] = iv->dom;
      analyzer.Bind(iv->var, iv->dom);
    }

    InferRootBound(stage, ctx, &ret);

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
