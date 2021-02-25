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

#include "../../te/schedule/graph.h"

namespace tvm {
namespace tir {
std::unordered_map<std::string, const TECapsuleNode*> TECapsule::capsules;

TECapsule TECapsuleNode::make(std::string name, Array<tir::Var> input_vars,
                              Array<te::Tensor> inputs, Array<te::Tensor> outputs,
                              Array<te::Tensor> non_external_inputs, te::Schedule schedule) {
  auto n = make_object<TECapsuleNode>();
  n->name = name;
  n->input_vars = std::move(input_vars);
  n->inputs = std::move(inputs);
  n->non_external_inputs = std::move(non_external_inputs);
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

TECapsule TECapsuleNode::ScheduleToTIR(te::Schedule& schedule,
                                       Array<tir::IterVar> env_threads) const {
  auto capsule = GetRef<TECapsule>(this);

  Array<te::Tensor> outputs = this->outputs;
  if (const auto* f = runtime::Registry::Get(this->name + "_schedule")) {
    // std::cout << "[IS] Invoking schedule function for " << this->name << std::endl;
    outputs = (*f)(schedule, GetRef<TECapsule>(this));
  }

  if (env_threads.defined() && env_threads.size() > 0) {
    capsule = this->EnvThreads(schedule, env_threads, outputs);
  }

  return capsule;
}

Array<te::Operation> TECapsuleNode::GetOutputOps() const {
  Array<te::Operation> output_ops;
  for (auto t : outputs) {
    if (!output_ops.Contains(t->op)) {
      output_ops.push_back(t->op);
    }
  }
  return output_ops;
}

TECapsule TECapsuleNode::EnvThreads(te::Schedule& schedule, Array<tir::IterVar> env_threads,
                                    Array<te::Tensor> updated_outputs) const {
  std::string new_name = this->name + "_sk";

  Array<te::Tensor> all_inputs;
  all_inputs.push_back_all(inputs);
  // all_inputs.push_back_all(non_external_inputs);
  te::Operation single_kernel = schedule.single_kernel(
      new_name, "", {}, all_inputs, Array<te::Tensor>(updated_outputs), false, env_threads);

  for (size_t i = 0; i < updated_outputs.size(); ++i) {
    auto old_output = updated_outputs[i];
    auto new_output = single_kernel.output(i);
    outputs.Set(i, new_output);

    std::cout << "[ENV] Single kernel " << old_output << " " << new_output << std::endl;

    if (interface_tensor_buffer_bounds.count(old_output)) {
      interface_tensor_buffer_bounds.Set(new_output, interface_tensor_buffer_bounds.at(old_output));
    }
  }

  return GetRef<TECapsule>(this);
}

Array<te::Tensor> TECapsuleNode::GetAllGlobalTensors(te::Schedule& schedule) const {
  Array<te::Tensor> ret;
  for (auto stage : schedule->stages) {
    if (stage->scope == "global") {
      for (size_t i = 0; i < stage->op->num_outputs(); ++i) {
        ret.push_back(stage->op.output(i));
      }
    }
  }
  return ret;
}

void TECapsuleNode::RefreshAllOps(bool recompute) const {
  if (recompute || this->all_ops_.size() == 0) {
    this->all_ops_ = GetSubGraphOrAllGraph(this->outputs, this->inputs, true);
  }
}

te::Tensor TECapsuleNode::GetTensor(std::string name, int idx) {
  this->RefreshAllOps(false);
  if (this->all_ops_.size() == 0) {
    this->all_ops_ = GetSubGraph(this->outputs, this->inputs, true);
  }

  for (auto op : this->all_ops_) {
    if (op->name == name) {
      return op.output(idx);
    }
  }
  for (auto op : this->all_ops_) {
    std::cout << "[DEBUG] " << op << std::endl;
  }

  CHECK(false) << "No such tensor " << name << " in capsule " << this->name;
  return {};
}

TVM_REGISTER_GLOBAL("tir.TECapsuleGetTensor")
    .set_body_typed([](TECapsule capsule, std::string name, int idx) {
      return const_cast<TECapsuleNode*>(capsule.as<TECapsuleNode>())->GetTensor(name, idx);
    });

TVM_REGISTER_GLOBAL("tir.TECapsuleSetInterfaceBounds")
    .set_body_typed([](TECapsule capsule, Map<te::Tensor, Array<Range>> inteface_bounds) {
      capsule.as<TECapsuleNode>()->interface_tensor_buffer_bounds = inteface_bounds;
    });

}  // namespace tir
}  // namespace tvm
