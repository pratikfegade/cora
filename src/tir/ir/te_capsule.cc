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
                              Array<te::Tensor> inputs, Array<te::Tensor> outputs,
                              te::Schedule schedule, tir::Stmt scheduled_output) {
  // CHECK(!TECapsule::capsules.count(name));
  auto n = make_object<TECapsuleNode>();
  n->name = name;
  n->input_vars = std::move(input_vars);
  n->inputs = std::move(inputs);
  n->outputs = std::move(outputs);
  n->schedule = std::move(schedule);
  n->scheduled_output = std::move(scheduled_output);

  auto ret = TECapsule(n);
  TECapsule::capsules[name] = ret.as<TECapsuleNode>();
  std::cout << "[MK] New TECapsule " << name << " " << ret->input_vars.size() << " "
            << ret->inputs.size() << " " << ret.get() << std::endl;
  return ret;
}

TVM_REGISTER_NODE_TYPE(TECapsuleNode);

TVM_REGISTER_GLOBAL("tir.CreateTECapsule")
    .set_body_typed([](Array<tir::Var> input_vars, Array<te::Tensor> inputs,
                       Array<te::Tensor> outputs, std::string name) {
      return TECapsuleNode::make(name, input_vars, inputs, outputs);
    });

TECapsule TECapsuleNode::ScheduleToTIR(Array<tir::IterVar> env_threads) const {
  this->InitSchedule();

  auto capsule = GetRef<TECapsule>(this);

  if (!this->scheduled_output.defined()) {
    if (const auto* f = runtime::Registry::Get(this->name + "_schedule")) {
      std::cout << "[IS] Invoking schedule function for " << this->name << std::endl;
      (*f)(GetRef<TECapsule>(this));
    }

    if (env_threads.defined() && env_threads.size() > 0) {
      capsule = this->EnvThreads(env_threads);
    }

    auto bounds = te::InferBound(capsule->schedule);
    auto stmt = te::ScheduleOps(capsule->schedule, bounds, false);
    stmt = tir::InjectPrefetch(stmt);

    const_cast<TECapsuleNode*>(capsule.as<TECapsuleNode>())->scheduled_output = stmt;
  }

  return capsule;
}

tir::Stmt TECapsuleNode::LowerToTIR(const BuildConfig& config,
                                    Map<te::Tensor, tir::Buffer> buf_bindings,
                                    Map<te::Tensor, tir::Buffer> partial_buf_bindings,
                                    Map<te::Tensor, Array<PrimExpr>> partial_index_bindings) const {
  // // Phase 0
  // auto bounds = te::InferBound(this->schedule);
  // auto stmt = te::ScheduleOps(this->schedule, bounds, false);
  // stmt = tir::InjectPrefetch(stmt);

  // bool compact = tir::VerifyCompactBuffer(stmt);
  auto stmt = tir::StorageFlatten(this->scheduled_output, buf_bindings, partial_buf_bindings,
                                  partial_index_bindings, 64, config->instrument_bound_checkers);
  stmt = tir::CanonicalSimplify(stmt);

  return stmt;
}

void TECapsuleNode::InitSchedule() const {
  if (!this->schedule.defined()) {
    Array<te::Operation> output_ops;
    for (auto t : outputs) {
      if (!output_ops.Contains(t->op)) {
        output_ops.push_back(t->op);
      }
    }

    this->schedule = te::create_schedule(output_ops);
  }
}

TECapsule TECapsuleNode::EnvThreads(Array<tir::IterVar> env_threads) const {
  this->InitSchedule();

  std::string new_name = this->name + "_sk";
  te::Operation single_kernel = this->schedule.single_kernel(
      new_name, "", {}, Array<te::Tensor>(inputs), Array<te::Tensor>(outputs), false, env_threads);

  Array<te::Tensor> new_outputs;
  for (size_t i = 0; i < this->outputs.size(); ++i) {
    // new_outputs.push_back(single_kernel.output(i));
    outputs.Set(i, single_kernel.output(i));
  }
  // auto ret = TECapsuleNode::make(this->name, this->input_vars, this->inputs, new_outputs,
  // this->schedule, this->scheduled_output);
  // std::cout << "[ET] Creating new TECapsule for " << this->name << " " << ret->input_vars.size()
  // << " " << ret->inputs.size() << " " << ret.get() << " " << this << std::endl;
  return GetRef<TECapsule>(this);
}

TVM_REGISTER_GLOBAL("tir.InitSchedule").set_body_typed([](TECapsule capsule) {
  capsule->InitSchedule();
});

}  // namespace tir
}  // namespace tvm
