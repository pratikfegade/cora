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
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/tensor_array.h>

namespace tvm {
namespace tir {
// TVM_REGISTER_NODE_TYPE(TensorArrayNode);

TensorArray decl_region_tensor_array(Array<PrimExpr> shape, Array<PrimExpr> tensor_shape,
                                     DataType dtype, std::string name) {
  return RegionTensorArrayNode::make(Var(name, DataType::Handle()), dtype, shape, tensor_shape,
                                     name);
}

TensorArray RegionTensorArrayNode::make(Var ta_var, DataType dtype, Array<PrimExpr> shape,
                                        Array<PrimExpr> tensor_shape, std::string name) {
  auto n = make_object<RegionTensorArrayNode>();
  n->ta_var = std::move(ta_var);
  n->dtype = dtype;
  n->shape = std::move(shape);
  n->tensor_shape = std::move(tensor_shape);
  n->name = std::move(name);

  auto ret = TensorArray(n);
  n->base_region_ta = ret;
  return ret;
}

TensorArray RegionTensorArrayNode::make(Var ta_var, DataType dtype, Array<PrimExpr> shape,
                                        Array<PrimExpr> tensor_shape, std::string name,
                                        TensorArray base_region_ta) {
  auto n = make_object<RegionTensorArrayNode>();
  n->ta_var = std::move(ta_var);
  n->dtype = dtype;
  n->shape = std::move(shape);
  n->tensor_shape = std::move(tensor_shape);
  n->name = std::move(name);
  n->base_region_ta = std::move(base_region_ta);
  // std::cout << "[TA] Creating BaseTA " << n->name << " " << n->base_region_ta << std::endl;

  return TensorArray(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RegionTensorArrayNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const RegionTensorArrayNode*>(node.get());
      p->stream << "region_ta(" << op->name << ", " << op << ")";
    });

TVM_REGISTER_NODE_TYPE(RegionTensorArrayNode);

TVM_REGISTER_GLOBAL("tir.RegionTensorArray")
    .set_body_typed([](Var data, DataType dtype, Array<PrimExpr> shape,
                       Array<PrimExpr> tensor_shape, std::string name) {
      return RegionTensorArrayNode::make(data, dtype, shape, tensor_shape, name);
    });

TVM_REGISTER_GLOBAL("tir.RegionTensorArrayWithBase")
    .set_body_typed([](Var data, DataType dtype, Array<PrimExpr> shape,
                       Array<PrimExpr> tensor_shape, std::string name, TensorArray base) {
      return RegionTensorArrayNode::make(data, dtype, shape, tensor_shape, name, base);
    });

TensorArray decl_pointer_tensor_array(Array<PrimExpr> shape, TensorArray region_ta,
                                      std::string name) {
  return PointerTensorArrayNode::make(Var(name, DataType::Handle()), region_ta, shape, name);
}

TensorArray PointerTensorArrayNode::make(Var ta_var, TensorArray base_region_ta,
                                         Array<PrimExpr> shape, std::string name) {
  CHECK(base_region_ta.as<RegionTensorArrayNode>());
  auto n = make_object<PointerTensorArrayNode>();
  n->ta_var = std::move(ta_var);
  n->base_region_ta = base_region_ta;
  n->shape = std::move(shape);
  n->name = std::move(name);

  return TensorArray(n);
}

TensorArray PointerTensorArrayNode::GetBaseTensorArray() const {
  return this->base_region_ta->GetBaseTensorArray();
}

TensorArray RegionTensorArrayNode::GetBaseTensorArray() const {
  TensorArray base = this->base_region_ta;
  while (base != base->base_region_ta) {
    base = base->base_region_ta;
  }
  // std::cout << "[TA] BaseTA " << this->name << " " << base->name << std::endl;
  return base;
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PointerTensorArrayNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PointerTensorArrayNode*>(node.get());
      p->stream << "pointer_ta(" << op->name << ", " << op << ")";
    });

TVM_REGISTER_NODE_TYPE(PointerTensorArrayNode);

TVM_REGISTER_GLOBAL("tir.PointerTensorArray")
    .set_body_typed([](Var data, TensorArray region_ta, Array<PrimExpr> shape, std::string name) {
      return PointerTensorArrayNode::make(data, region_ta, shape, name);
    });
}  // namespace tir
}  // namespace tvm
