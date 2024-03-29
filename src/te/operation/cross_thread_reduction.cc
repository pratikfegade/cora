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
 * \brief Logics related to cross thread reduction, used by ComputeOpNode.
 * \file cross_thread_reduction.cc
 */
#include <tvm/tir/ir_pass.h>

#include "compute_op.h"
#include "op_util.h"

namespace tvm {
namespace te {
using namespace tir;

Stmt MakeCrossThreadReduction(const ComputeOpNode* self, const Stage& stage,
                              const std::unordered_map<IterVar, Range>& dom_map,
                              const std::unordered_map<std::string, Range>& env_dom_map,
                              const std::unordered_map<std::string, IterVar>& env_var_map,
                              const std::unordered_map<const VarNode*, std::string>& bind_map,
                              const Map<Stage, Array<Stage>>& attach_stages,
                              const Map<Stage, Array<IterVar>>& attach_vars,
                              bool debug_keep_trivial_loop) {
  std::unordered_map<const DimensionNode*, Range> dim_doms;
  for (auto dim : self->root_index_dimensions) {
    auto iv = self->GetIterVarFromDim(0, dim);
    if (dom_map.count(iv)) {
      dim_doms[dim.operator->()] = dom_map.at(self->GetIterVarFromDim(0, dim));
    } else {
      dim_doms[dim.operator->()] = iv->dom;
    }
  }

  DimensionPassDownDomain(stage, self, &dim_doms, true);

  std::unordered_map<const DimensionNode*, PrimExpr> dim_vals;
  for (auto dim : self->root_index_dimensions) {
    dim_vals[dim.operator->()] = self->GetIterVarFromDim(0, dim)->var;
  }

  DimensionPassDownValues(stage, self, dim_doms, &dim_vals, true);
  // DimensionPassDownValues(stage, self, &dim_vals, true);

  Array<PrimExpr> args;
  for (auto dim : stage->dim_relation_graph->leaf_dimensions) {
    args.push_back(dim_vals[dim.operator->()]);
  }

  // Array<PrimExpr> args;
  // for (auto dim : self->root_index_dimensions) {
  //   args.push_back(self->GetIterVarFromDim(0, dim)->var);
  // }
  std::unordered_map<IterVar, PrimExpr> value_map;
  auto nest = MakeComputeOpLoopNest(stage, dom_map, 0, false, std::unordered_set<IterVar>(),
                                    &value_map, debug_keep_trivial_loop, self->all_dimensions);

  auto conds = MakeBoundCheck(stage, dom_map, env_dom_map, env_var_map, bind_map, value_map, false,
                              std::unordered_set<IterVar>(), attach_stages, attach_vars);

  size_t size = self->body.size();
  CHECK_GT(size, 0);
  std::vector<const ReduceNode*> reduces(size);
  for (size_t i = 0; i < size; ++i) {
    const ReduceNode* reduce = self->body[i].as<ReduceNode>();
    CHECK(reduce);
    reduces[i] = reduce;
  }
  PrimExpr cond = reduces[0]->condition;
  for (PrimExpr v : conds) {
    cond = cond && v;
  }
  Array<PrimExpr> freduce_args;
  freduce_args.push_back(make_const(DataType::UInt(32), static_cast<uint32_t>(size)));
  for (size_t i = 0; i < size; ++i) {
    freduce_args.push_back(reduces[0]->source[i]);
  }
  freduce_args.push_back(cond);
  std::vector<Var> res_handles(size);
  for (size_t idx = 0; idx < size; ++idx) {
    res_handles[idx] = Var("reduce_temp" + std::to_string(idx), DataType::Handle());
    freduce_args.push_back(res_handles[idx]);
  }

  IterVar reduction_thread;
  for (IterVar iv : stage->leaf_iter_vars) {
    if (iv->iter_type == kCommReduce) {
      auto it = stage->iter_var_attrs.find(iv);
      if (it != stage->iter_var_attrs.end() && (*it).second->bind_thread.defined()) {
        IterVar tv = (*it).second->bind_thread;
        freduce_args.push_back(tv->var);
        reduction_thread = tv;
      }
    }
  }
  // Checks for the thread.
  std::vector<PrimExpr> thread_head_check;
  if (stage->store_predicate.defined()) {
    thread_head_check.emplace_back(stage->store_predicate);
  }

  if (stage->op.as<ComputeOpNode>()) {
    for (const auto& p : self->pred) {
      thread_head_check.emplace_back(p);
    }
  }

  Stmt reduce_body = EvaluateNode::make(CallNode::make(
      DataType::Handle(), tir::intrinsic::tvm_thread_allreduce, freduce_args, CallNode::Intrinsic));
  reduce_body = AttrStmtNode::make(reduces[0]->combiner, attr::reduce_scope,
                                   make_zero(DataType::Handle()), reduce_body);
  std::vector<Stmt> assigns(size);
  for (size_t idx = 0; idx < size; ++idx) {
    DataType t = reduces[idx]->dtype;
    assigns[idx] = ProvideNode::make(
        stage->op, idx,
        LoadNode::make(t, res_handles[idx], 0, const_true(t.lanes()),
                       self->attrs.count("no_sync") ? tvm::tir::kNone : tvm::tir::kAll),
        args);
  }
  Stmt assign_body = SeqStmt::Flatten(assigns);
  assign_body = MergeNest(MakeIfNest(thread_head_check), assign_body);
  assign_body = MergeNest(MakeIfNest(conds), assign_body);
  // assign_body = IfThenElseNode::make(reduction_thread->var < 1, assign_body,
  // EvaluateNode::make(0));
  Stmt body = SeqStmt::Flatten(reduce_body, assign_body);
  for (size_t idx = size; idx != 0; --idx) {
    body =
        AllocateNode::make(res_handles[idx - 1], reduces[idx - 1]->dtype, {1}, const_true(), body);
    body = AttrStmtNode::make(res_handles[idx - 1], attr::storage_scope,
                              StringImmNode::make("local"), body);
  }
  body = Substitute(body, value_map);
  Stmt body_with_loops = MergeNest(nest, body);
  return Substitute(body_with_loops, value_map);
}
}  // namespace te
}  // namespace tvm
