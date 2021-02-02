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
 * \file tvm/tir/buffer.h
 * \brief Symbolic n-dimensional array, to represent a memory buffer.
 */
#ifndef TVM_TE_CAPSULE_ARRAY_H_
#define TVM_TE_CAPSULE_ARRAY_H_

#include <tvm/node/container.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>

#include <string>
#include <unordered_map>

namespace tvm {
namespace tir {
// TECapsule
class TECapsuleNode;

/*!
 * \brief TECapsule encapsulates a TE operation graph.
 */
class TECapsule : public ObjectRef {
 public:
  static std::unordered_map<std::string, const TECapsuleNode*> capsules;

  TECapsule() {}
  explicit TECapsule(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const TECapsuleNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = TECapsuleNode;
};

/*! \brief Node to represent a TECapsule */
class TECapsuleNode : public Object {
 public:
  std::string name = "panorma";
  Array<tir::Var> input_vars;
  Array<te::Tensor> inputs;
  Map<te::Tensor, Array<Range>> interface_tensor_buffer_bounds;
  mutable Array<te::Tensor> outputs;
  mutable te::Schedule schedule;
  mutable tir::Stmt scheduled_output;
  mutable Array<te::Operation> all_ops_;

  /*! \brief constructor */
  TECapsuleNode() {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("input_vars", &input_vars);
    v->Visit("inputs", &inputs);
    v->Visit("outputs", &outputs);
    v->Visit("interface_tensor_buffer_bounds", &interface_tensor_buffer_bounds);
    v->Visit("name", &name);
    v->Visit("schedule", &schedule);
    v->Visit("scheduled_output", &scheduled_output);
  }

  TVM_DLL TECapsule ScheduleToTIR(Array<tir::IterVar> env_threads) const;

  TVM_DLL tir::Stmt LowerToTIR(const BuildConfig& config, Map<te::Tensor, tir::Buffer> buf_bindings,
                               Map<te::Tensor, tir::Buffer> partial_buf_bindings,
                               Map<te::Tensor, Array<PrimExpr>> partial_index_bindings) const;

  TVM_DLL static TECapsule make(std::string name, Array<tir::Var> input_vars,
                                Array<te::Tensor> inputs, Array<te::Tensor> outputs,
                                Map<te::Tensor, Array<Range>> interface_tensor_buffer_bounds = {},
                                te::Schedule schedule = {}, tir::Stmt scheduled_output = {});

  TVM_DLL TECapsule EnvThreads(Array<IterVar> env_threads, Array<te::Tensor> updated_outputs) const;

  TVM_DLL void InitSchedule() const;

  TVM_DLL te::Tensor GetTensor(std::string name, int idx);

  static constexpr const char* _type_key = "TECapsule";
  TVM_DECLARE_BASE_OBJECT_INFO(TECapsuleNode, Object);
};

inline const TECapsuleNode* TECapsule::operator->() const {
  return static_cast<const TECapsuleNode*>(get());
}

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TE_CAPSULE_ARRAY_H_
