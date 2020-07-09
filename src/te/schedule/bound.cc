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
               const runtime::StorageScope& scope) {
  auto it = bind_map.find(iv);
  const std::string& tag = (it != bind_map.end() ? it->second->thread_tag : iv->thread_tag);
  if (tag.length() == 0 || tag == "pipeline") {
    // std::cout << "[NRLX]      1" << std::endl;
    return !found_attach;
  }
  ThreadScope ts = ThreadScope::make(tag);

  // When there is warp memory
  // threadIdx.x must be set to be warp index.
  if (scope.rank == StorageRank::kWarp && ts.rank == 1 && ts.dim_index == 0) {
    // std::cout << "[NRLX]      2" << std::endl;
    return true;
  }
  // std::cout << "[NRLX]      3" << std::endl;
  return static_cast<int>(scope.rank) <= ts.rank;
}

// infer storage scope, if not given
StorageScope InferStorageScope(const Stage& stage, const GraphContext& ctx) {
  if (stage->scope.length() != 0) {
    StorageScope s = StorageScope::make(stage->scope);
    const_cast<Stage&>(stage)->storage_scope_rank = static_cast<int>(s.rank);
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
        // if (producer->name == "cl_hz_gate" && consumer->name == "c_next_h")
        //   std::cout << "[TRANS] " << var_node->name_hint << " " << var_node << " "
        //             << p->dim2var_maps[0].at(dim).iv->var->name_hint << " "
        //             << p->dim2var_maps[0].at(dim).iv->var.as<VarNode>() << std::endl;
      }
    }
  }

  VarReplacer replacer(vsub);
  return Range::make_by_min_extent(replacer(range->min), replacer(range->extent));
  // return range;
}

void InferRootBound(const Stage& stage, const GraphContext& ctx,
                    std::unordered_map<IterVar, Range>* rmap) {
  // std::cout << "[TVM] Inferring bounds for " << stage << std::endl;
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
      // (*rmap)[iv] = UninterpFun::InlineUninterpFunCalls(iv->dom);
      (*rmap)[iv] = iv->dom;
      // if (stage->is_output)
      // std::cout << "[OUT] " << stage->op << " " << iv->var << " "
      //           << UninterpFun::InlineUninterpFunCalls(iv->dom) << std::endl;
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
  // if (stage->op->name == "css_init") {
  //   std::cout << "[IRB] Attach for " << stage->op << stage->attach_stage << std::endl;
  //   for (size_t i = 0; i < stage_attach.size(); ++i) {
  //     std::cout << "[IRB]   Attach " << stage_attach[i] << " "
  //               << ctx.attach_path_ops.at(stage->op)[i] << std::endl;
  //   }
  // }

  // The parent set.
  for (const Operation& op : consumers) {
    bool print = false;  //(stage->op->name == "next_v");
    if (print) std::cout << stage->op->name << std::endl;
    std::unordered_map<const VarNode*, IntSet> relax_set;
    std::unordered_map<IterVar, IntSet> up_state;
    bool found_attach = false;
    CHECK(ctx.op2stage_.count(op.get())) << op << " " << stage->op;
    const Stage& op_stage = ctx.op2stage_.at(op.get());
    /************************* Phase 1 *************************/
    // Consumer nest
    for (size_t i = op_stage->leaf_iter_vars.size(); i != 0; --i) {
      IterVar iv = op_stage->leaf_iter_vars[i - 1];
      if (stage_attach.size() != 0 && iv == stage_attach[0]) {
        found_attach = true;
      }
      auto it = rmap->find(iv);
      CHECK(it != rmap->end()) << iv->var << " " << op_stage->op << " " << stage;
      const Range& vrange = it->second;

      IterVarAttr it_attr;
      if (op_stage->iter_var_attrs.count(iv)) {
        it_attr = op_stage->iter_var_attrs[iv];
      }

      if (print)
        std::cout << "[IRB]  LV " << iv << " " << iv->iter_type << " " << it_attr << " " << op_stage
                  << std::endl;
      if (is_one(vrange->extent)) {
        up_state[iv] = IntSet::single_point(vrange->min);
        if (print) std::cout << "[IRB]    upb1 " << iv << " " << up_state[iv] << std::endl;
      } else if (!NeedRelax(iv, found_attach, ctx.bind_map, scope) &&
                 /* If an IV is opaque to loop nest creation, it means
                    we would not have a loop corresponding to such an
                    IV and so it doesn't make sense to not relax */
                 iv->iter_type != kLoopNestOpaque &&
                 (iv->iter_type != kSplit &&
                  (!it_attr.defined() || it_attr->iter_type != kSplit))) {
        CHECK(is_zero(vrange->min)) << "InferBound requires every leaf iter var's min equals 0, "
                                    << " call schedule.normalize to achieve this. " << vrange << " "
                                    << iv << " " << op_stage->op;
        // if (ctx.bind_map.count(iv)) {
        //   up_state[iv] = IntSet::single_point(ctx.bind_map.at(iv)->var);
        //   std::endl;
        // } else {
        //   up_state[iv] = IntSet::single_point(iv->var);
        //   // if (print) std::cout << "[IRB]    upb3 " << iv << " " << up_state[iv] <<
        //   std::endl;
        // }

        up_state[iv] = IntSet::single_point(iv->var);
        if (print) std::cout << "[IRB]    upb2 " << iv << " " << up_state[iv] << std::endl;
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

      vrange = TranslateIterVarsFromConsumerToProducer(vrange, iv_op, stage->op);
      CHECK(is_zero(vrange->min)) << "InferBound requires every leaf iter var's min equals 0, "
                                  << "call schedule.normalize to achieve this. " << vrange << " "
                                  << iv;
      if (print)
        std::cout << "[RLX]    Try relax " << iv << "" << iv_op << " " << found_attach << " "
                  << scope.to_string() << std::endl;
      if (NeedRelax(iv, found_attach, ctx.bind_map, scope)) {
        if (print)
          std::cout << "[RLX]      Relaxed "
                    << " " << IntSet::range(vrange) << std::endl;
        relax_set[iv->var.get()] = IntSet::range(vrange);
        if (ctx.bind_map.count(iv)) {
          relax_set[ctx.bind_map.at(iv)->var.get()] = IntSet::range(vrange);
          if (print)
            std::cout << "[RLX]      BindRelaxed "
                      << " " << ctx.bind_map.at(iv) << std::endl;
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
        // if (print) std::cout << "[IRB]    upa1 " << iv << " " << r << std::endl;
      } else {
        r = iv->dom;
        // if (print) std::cout << "[IRB]    upa2 " << iv << " " << r << std::endl;
      }
      if (relax_set.size() != 0) {
        auto vars = VarCollector().collect(r);
        std::unordered_map<std::string, const VarNode*> name_var_map;
        for (auto var : vars) {
          name_var_map[var->name_hint] = var;
        }

        std::unordered_map<const VarNode*, IntSet> relax_set_updated;
        std::unordered_set<std::string> to_relax_env_threads;
        for (auto it : relax_set) {
          relax_set_updated[it.first] = it.second;
          if (isCudaThread(it.first->name_hint)) to_relax_env_threads.insert(it.first->name_hint);
          //   if (isCudaThread(it.first->name_hint) && name_var_map.count(it.first->name_hint)) {
          //     relax_set_updated[name_var_map.at(it.first->name_hint)] = it.second;
          //     std::cout << "[RLXUPD] " << it.first->name_hint << " " << it.first << " "
          //               << name_var_map.at(it.first->name_hint) << " " << it.second << std::endl;
          //   }
        }

        std::unordered_map<const VarNode*, std::string> bind_rmap;
        for (auto it : ctx.bind_map) {
          bind_rmap[it.first->var.as<VarNode>()] = it.second->var->name_hint;
        }

        for (auto var : vars) {
          if (bind_rmap.count(var)) {
            auto name = bind_rmap.at(var);
            if (to_relax_env_threads.count(name)) {
              Range r = NullValue<Range>();
              for (auto it : *rmap)
                if (it.first->var.get() == var) r = it.second;
              if (r.defined()) relax_set_updated[var] = IntSet::range(r);
            }
          }
        }

        // r = VarReplacer(bind_rmap).replace(r);

        dom_map[iv->var.get()] = EvalSet(r, relax_set_updated);
        if (print)
          std::cout << "[IRB]    iv1 " << iv << " " << r << " " << dom_map[iv->var.get()]
                    << std::endl;
      } else {
        dom_map[iv->var.get()] = IntSet::range(r);
        if (print) std::cout << "[IRB]    iv2 " << iv << " " << dom_map[iv->var.get()] << std::endl;
      }
      analyzer.Bind(iv->var, r);
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
  // std::cout << "[Bound] FeedGraph" << std::endl;
  ctx.feed_graph = CreateFeedGraph(CreateReadGraph(roots, false));
  // TODO: Mighty mighty global variable hack
  // for (auto it : sch->cacheTensorInfos) {
  // std::cout << "[Bound]   Map " << it.first << " " << it.second->orig << std::endl;
  // }
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
    // std::cout << "[ENV] Op " << stage << std::endl;
    for (auto iv : ctx.attach_path.at(op)) {
      IterVar enviv = NullValue<IterVar>();
      if (isCudaThread(iv)) {
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

    // for (auto it : op_env_bounds) {
    //   std::cout << "[ENV]   " << it.first << " " << it.second << std::endl;
    // }

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
        // std::cout << "Updating " << kv.second->bind_thread << " to " <<
        // ret[kv.second->bind_thread]
        //           << std::endl;
        analyzer.Bind(kv.second->bind_thread->var, ret[kv.second->bind_thread]);
        updated.insert(kv.second->bind_thread);
      }
    }
  }

  for (auto& p : ret) {
    ret[p.first] = Range::make_by_min_extent(analyzer.Simplify(p.second->min),
                                             analyzer.Simplify(p.second->extent));
  }

  return InferBoundsResultNode::make(Map<IterVar, Range>(ret.begin(), ret.end()), env_bounds, env_vars);
}

TVM_REGISTER_GLOBAL("schedule.InferBound").set_body_typed(InferBound);

}  // namespace te
}  // namespace tvm
