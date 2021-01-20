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
 * \file buffer.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/te_capsule.h>

namespace tvm {
namespace tir {
std::unordered_map<std::string, const TECapsuleNode*> TECapsule::capsules;

TECapsule TECapsuleNode::make(std::string name, Array<tir::Var> input_vars,
                              Array<te::Tensor> inputs, Array<te::Tensor> outputs) {
  CHECK(!TECapsule::capsules.count(name));
  auto n = make_object<TECapsuleNode>();
  n->name = name;
  n->input_vars = std::move(input_vars);
  n->inputs = std::move(inputs);
  n->outputs = std::move(outputs);
  auto ret = TECapsule(n);
  TECapsule::capsules[name] = ret.as<TECapsuleNode>();
  return ret;
}

TVM_REGISTER_NODE_TYPE(TECapsuleNode);

TVM_REGISTER_GLOBAL("tir.CreateTECapsule")
    .set_body_typed([](Array<tir::Var> input_vars, Array<te::Tensor> inputs,
                       Array<te::Tensor> outputs, std::string name) {
      return TECapsuleNode::make(name, input_vars, inputs, outputs);
    });

tir::Stmt TECapsuleNode::LowerToTIR(const BuildConfig& config,
                                    Map<te::Tensor, tir::Buffer> buf_bindings,
                                    Map<te::Tensor, tir::Buffer> partial_buf_bindings,
                                    Map<te::Tensor, Array<PrimExpr>> partial_index_bindings) const {
  Array<te::Operation> output_ops;
  for (auto t : outputs) {
    if (!output_ops.Contains(t->op)) {
      output_ops.push_back(t->op);
    }
  }

  te::Schedule sch = te::create_schedule(output_ops);

  // Phase 0
  auto bounds = te::InferBound(sch);
  auto stmt = te::ScheduleOps(sch, bounds, false);
  stmt = tir::InjectPrefetch(stmt);

  bool compact = tir::VerifyCompactBuffer(stmt);
  stmt = tir::StorageFlatten(stmt, buf_bindings, partial_buf_bindings, partial_index_bindings, 64,
                             config->instrument_bound_checkers);
  stmt = tir::CanonicalSimplify(stmt);

  return stmt;
}

}  // namespace tir
}  // namespace tvm
