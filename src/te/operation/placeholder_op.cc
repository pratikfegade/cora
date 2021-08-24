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
 * \brief Placeholder op.
 * \file placeholder_op.cc
 */
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>

namespace tvm {
namespace te {

// PlaceholderOpNode
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PlaceholderOpNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PlaceholderOpNode*>(node.get());
      p->stream << "placeholder(" << op->name << ", " << op << ")";
    });

TVM_REGISTER_NODE_TYPE(PlaceholderOpNode);

Dimension PlaceholderOpNode::GetBaseIndexDimension(size_t val_idx, size_t dim_idx) const {
  return self_index_dimensions[dim_idx];
}

Array<DimInfo> PlaceholderOpNode::GetAllDimensions() const { return all_dimensions; }

Array<Dimension> PlaceholderOpNode::GetRootIndexDimensions(size_t val_idx) const {
  return self_index_dimensions;
}

int PlaceholderOpNode::num_outputs() const { return 1; }

Array<IterVar> PlaceholderOpNode::root_iter_vars() const { return {}; }

DataType PlaceholderOpNode::output_dtype(size_t i) const {
  CHECK_EQ(i, 0U);
  return dtype;
}

Array<PrimExpr> PlaceholderOpNode::output_shape(size_t i) const {
  CHECK_EQ(i, 0U);
  return shape;
}

void CreateDimVarMappings(PlaceholderOpNode* op) {
  op->dim2var_maps.clear();
  std::unordered_map<const DimensionNode*, DimVarEntry> dim2var_map;
  for (auto di : op->all_dimensions) {
    dim2var_map[di->dim.as<DimensionNode>()] = {di->dim, di->iv};
    op->var2dim_map[di->iv->var.as<VarNode>()] = di->dim.as<DimensionNode>();
  }
  for (size_t i = 0; i < static_cast<size_t>(op->num_outputs()); ++i) {
    op->dim2var_maps.push_back(std::move(dim2var_map));
  }
}

void PlaceholderOpNode::set_storage_layout(Modes layout) { this->layout = layout; }

Operation PlaceholderOpNode::make(std::string name, Array<PrimExpr> shape, DataType dtype) {
  auto n = make_object<PlaceholderOpNode>();
  n->name = name;
  n->shape = shape;
  n->dtype = dtype;
  return Operation(n);
}

Operation PlaceholderOpNode::make(std::string name, Array<PrimExpr> shape, Modes layout,
                                  DataType dtype, Array<Dimension> self_index_dimensions,
                                  Array<Dimension> dimensions, Array<IterVar> itervars,
                                  Array<UninterpFun> uninterpfuns) {
  if (!layout.defined()) {
    layout = ModesNode::make(name, shape, false);
  }
  auto n = make_object<PlaceholderOpNode>();
  n->name = name;
  n->shape = shape;
  n->layout = layout;
  n->dtype = dtype;
  n->self_index_dimensions = self_index_dimensions;

  for (size_t i = 0; i < uninterpfuns.size(); ++i) {
    CHECK(dimensions[i]->type != DimensionNode::kFunDim);
    n->all_dimensions.push_back(DimInfoNode::make(dimensions[i], itervars[i]));
  }
  CreateDimVarMappings(n.get());
  auto ret = Operation(n);
  // std::cout << "[PL] PL op with layout " << ret << " " << layout << std::endl;
  return ret;
}

Tensor placeholder(Array<PrimExpr> shape, DataType dtype, std::string name) {
  return PlaceholderOpNode::make(name, shape, dtype).output(0);
}

TVM_REGISTER_GLOBAL("te.Placeholder")
    .set_body_typed([](Array<PrimExpr> shape, DataType dtype, std::string name) {
      return placeholder(shape, dtype, name);
    });

TVM_REGISTER_GLOBAL("te.IndirectPlaceholder")
    .set_body_typed([](Array<PrimExpr> shape, Modes layout, Array<Dimension> self_index_dimensions,
                       Array<Dimension> dimensions, Array<IterVar> itervars,
                       Array<UninterpFun> uninterpfuns, DataType dtype, std::string name) {
      return PlaceholderOpNode::make(name, shape, layout, dtype, self_index_dimensions, dimensions,
                                     itervars, uninterpfuns)
          .output(0);
    });

Array<Tensor> PlaceholderOpNode::InputTensors() const { return {}; }

Operation PlaceholderOpNode::ReplaceInputs(const Operation& self,
                                           const std::unordered_map<Tensor, Tensor>& rmap) const {
  return self;
}

void PlaceholderOpNode::PropBoundToInputs(
    const Operation& self, arith::Analyzer* analyzer,
    const std::unordered_map<const VarNode*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {}

void PlaceholderOpNode::GatherBound(const Operation& self,
                                    const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                                    std::unordered_map<IterVar, Range>* out_dom_map,
                                    const Map<FunctionRef, CacheInfo> cacheTensorInfos) const {}

Stmt PlaceholderOpNode::BuildRealize(const Stage& stage,
                                     const std::unordered_map<IterVar, Range>& realize_map,
                                     const Stmt& body) const {
  return body;
}

Stmt PlaceholderOpNode::BuildProvide(
    const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
    const std::unordered_map<std::string, Range>& env_dom_map,
    const std::unordered_map<std::string, IterVar>& env_var_map,
    const std::unordered_map<const VarNode*, std::string>& bind_map,
    const Map<Stage, Array<Stage>>& attach_stages, const Map<Stage, Array<IterVar>>& attach_vars,
    bool debug_keep_trivial_loop) const {
  return Stmt();
}

TVM_REGISTER_GLOBAL("te.PlaceholderOpGetRootIndexDimensions")
    .set_body_typed([](Operation op, int value_index) {
      auto p_op = op.as<PlaceholderOpNode>();
      if (p_op->self_index_dimensions.size() == 0) {
        return NullValue<Array<Dimension>>();
      }
      return p_op->self_index_dimensions;
    });

}  // namespace te
}  // namespace tvm
