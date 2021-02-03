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
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/ta_declarations.h>

namespace tvm {
namespace tir {
TALayout TALayoutNode::make(std::string storage_scope, const Array<Range> layout) {
  auto n = make_object<TALayoutNode>();
  n->storage_scope = std::move(storage_scope);
  n->layout = std::move(layout);
  return TALayout(n);
}

TVM_REGISTER_NODE_TYPE(TALayoutNode);

TVM_REGISTER_GLOBAL("tir.CreateTALayout")
    .set_body_typed([](std::string storage_scope, const Array<Range> layout) {
      return TALayoutNode::make(storage_scope, layout);
    });

TADeclarations TADeclarationsNode::make(const Array<TensorArray> tensor_arrays,
                                        const Array<Buffer> buffers) {
  auto n = make_object<TADeclarationsNode>();
  for (auto ta : tensor_arrays) {
    n->var2ta_map[ta->ta_var.get()] = ta;
  }
  for (auto buf : buffers) {
    n->var2buf_map[buf->data.get()] = buf;
  }
  return TADeclarations(n);
}

TVM_REGISTER_NODE_TYPE(TADeclarationsNode);

TVM_REGISTER_GLOBAL("tir.CreateTADeclarations")
    .set_body_typed([](const Array<TensorArray> tensor_arrays, const Array<Buffer> buffers) {
      return TADeclarationsNode::make(tensor_arrays, buffers);
    });

TVM_REGISTER_GLOBAL("tir.TADeclarationsGetTensorArray")
    .set_body_typed([](TADeclarations decls, const std::string& name) {
      TADeclarationsNode* node = decls.operator->();
      for (auto it : node->var2ta_map) {
        if (it.second->name == name) return it.second;
      }
      return NullValue<TensorArray>();
    });

TVM_REGISTER_GLOBAL("tir.TADeclarationsAddTALayouts")
    .set_body_typed([](TADeclarations decls, const Map<TensorArray, TALayout> ta_layouts) {
      TADeclarationsNode* node = decls.operator->();
      node->ta_layouts = ta_layouts;
    });

TADeclarations TADeclarations::add_tensor_array(TensorArray ta) {
  TADeclarationsNode* node = (*this).operator->();
  node->var2ta_map[ta->ta_var.get()] = ta;
  return *this;
}

TADeclarations TADeclarations::add_buffer(Buffer buf) {
  TADeclarationsNode* node = (*this).operator->();
  node->var2buf_map[buf->data.get()] = buf;
  return *this;
}

TensorArray TADeclarations::get_tensor_array(Var var) {
  TADeclarationsNode* node = (*this).operator->();
  auto it = node->var2ta_map.find(var.get());
  CHECK(it != node->var2ta_map.end());
  return it->second;
}

Buffer TADeclarations::get_buffer(Var var) {
  TADeclarationsNode* node = (*this).operator->();
  auto it = node->var2buf_map.find(var.get());
  CHECK(it != node->var2buf_map.end());
  return it->second;
}

Array<TensorArray> TADeclarations::get_base_tensor_arrays() const {
  const TADeclarationsNode* node = (*this).operator->();
  Array<TensorArray> tas;
  for (auto it : node->var2ta_map) {
    auto base_ta = it.second->GetBaseTensorArray();
    if (!tas.Contains(base_ta)) {
      tas.push_back(base_ta);
    }
  }
  return tas;
}

Array<TensorArray> TADeclarations::get_all_tensor_arrays() const {
  const TADeclarationsNode* node = (*this).operator->();
  Array<TensorArray> tas;
  for (auto it : node->var2ta_map) {
    auto ta = it.second;
    if (!tas.Contains(ta)) {
      tas.push_back(ta);
    }
  }
  return tas;
}

}  // namespace tir
}  // namespace tvm
