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
 * \brief Scan Operator.
 * \file scan_op.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include "op_util.h"
#include "../schedule/graph.h"
#include "../../arith/interval_set.h"
#include "../../tir/ir/var_replacer.h"

namespace tvm {
namespace te {
using namespace tir;

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<SingleKernelEnvelopeOpNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const SingleKernelEnvelopeOpNode*>(node.get());
    p->stream << "single_kernel(" << op->name << ", " << op << ")";
});
TVM_REGISTER_NODE_TYPE(SingleKernelEnvelopeOpNode);

int SingleKernelEnvelopeOpNode::num_outputs() const {
  return static_cast<int>(inputs.size());
}
inline bool prove_equal(PrimExpr lhs, PrimExpr rhs) {
  return is_zero(tir::Simplify(lhs - rhs));
}
Array<IterVar> SingleKernelEnvelopeOpNode::root_iter_vars() const {
  Array<IterVar> ret;
  for (const auto& dim2var_map: dim2var_maps) {
    for (const auto& it: dim2var_map) {
      if (it.first->type <= DimensionNode::kRangeDim) {
	ret.push_back(it.second.iv);
      }
    }
  }
  return ret;
}

DataType SingleKernelEnvelopeOpNode::output_dtype(size_t i) const {
  return input_ops[i]->output_dtype(inputs[i]->value_index);
}

Array<PrimExpr> SingleKernelEnvelopeOpNode::output_shape(size_t i) const {
  return input_ops[i]->output_shape(inputs[i]->value_index);
}

Dimension SingleKernelEnvelopeOpNode::GetBaseIndexDimension(size_t val_idx, size_t dim_idx) const {
  return input_ops[val_idx]->GetBaseIndexDimension(inputs[val_idx]->value_index, dim_idx);
}

Operation SingleKernelEnvelopeOpNode::make(std::string name,
                           std::string tag,
                           Map<std::string, ObjectRef> attrs,
                           Array<Tensor> inputs) {
  if (!attrs.defined()) {
    attrs = Map<std::string, ObjectRef>();
  }
  auto n = make_object<SingleKernelEnvelopeOpNode>();

  std::vector<const BaseVarDimOpNode*> input_ops;
  for (auto input: inputs) {
    if (input->op.as<ScanOpNode>()) input_ops.push_back(input->op.as<ScanOpNode>());
    else if (input->op.as<ComputeOpNode>()) input_ops.push_back(input->op.as<ComputeOpNode>());
    else if (input->op.as<SpecializationEnvelopeOpNode>()) input_ops.push_back(input->op.as<SpecializationEnvelopeOpNode>());
    else CHECK(false)
	   << "All participating ops should be scans or computes but instead we have a " << input->op;
  }

  auto num_outputs = inputs.size();

  std::unordered_map<const VarNode*, PrimExpr> vmap;
  std::unordered_set<const BaseVarDimOpNode*> input_ops_set(input_ops.begin(), input_ops.end());
  for (const auto& op: input_ops_set) {
    for (const auto& dim2var_map: op->dim2var_maps) {
      for (const auto& it: dim2var_map) {
	vmap[it.second.iv->var.as<VarNode>()] = it.second.iv->var.copy_with_suffix(".env");
      }
    }
  }

  n->dim2var_maps = std::vector<std::unordered_map<const DimensionNode*, DimVarEntry>>(num_outputs);

  VarReplacer var_replacer(vmap);
  for (size_t i = 0; i < num_outputs; ++i) {
    const auto& op = input_ops[i];
    for (const auto& dim2var_map: op->dim2var_maps) {
      for (const auto& it: dim2var_map) {
	Dimension dim = GetRef<Dimension>(it.first);
	auto entry = it.second;
	IterVar iv = IterVarNode::make(Range::make_by_min_extent(var_replacer(entry.iv->dom->min),
								 var_replacer(entry.iv->dom->extent)),
				       Downcast<Var>(vmap[entry.iv->var.as<VarNode>()]),
				       kOpaque);
	n->dim2var_maps[i][it.first] = { dim, iv, entry.value_expr };
      }
    }
  }

  for (size_t j = 0; j < inputs.size(); ++j) {
    auto op = input_ops[j];
    for (size_t i = 0; i < inputs[j].ndim(); ++i) {
      n->spatial_dimensions_.push_back(op->GetBaseIndexDimension(inputs[j]->value_index, i));
    }
  }

  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->inputs = std::move(inputs);
  n->input_ops = std::move(input_ops);
  return Operation(n);
}

TVM_REGISTER_GLOBAL("te.SingleKernelEnvelopeOp")
.set_body_typed([](std::string name,
		   std::string tag,
		   Map<std::string, ObjectRef> attrs,
		   Array<Tensor> input_tensors) {
  return SingleKernelEnvelopeOpNode::make(name, tag, attrs, input_tensors);
});


Array<Tensor> SingleKernelEnvelopeOpNode::InputTensors() const {
  Array<Tensor> ret;
  for (const auto& t: inputs) {
    ret.push_back(t);
  }
  return ret;
}

Operation SingleKernelEnvelopeOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  bool replaced = false;
  Array<Tensor> new_inputs;
  for (size_t i = 0; i < this->inputs.size(); ++i) {
    if (rmap.count(this->inputs[i])) {
      new_inputs.push_back(rmap.at(this->inputs[i]));
      replaced = true;
    }
    else {
      new_inputs.push_back(this->inputs[i]);
    }
  }

  if (replaced) {
    return SingleKernelEnvelopeOpNode::make(this->name, this->tag, this->attrs, new_inputs);
  }
  else {
    return self;
  }
}

void SingleKernelEnvelopeOpNode::PropBoundToInputs(
    const Operation& self,
    arith::Analyzer* analyzer,
    const std::unordered_map<const VarNode*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
  for (int i = 0, sp_idx = 0; i < this->num_outputs(); ++i) {
    Tensor t = inputs[i];
    TensorDom* tdom = nullptr;
    if (out_dom_map->count(t)) { tdom = &out_dom_map->at(t); }
    // The update dimensions
    for (size_t k = 0; k < this->inputs[i]->shape.size(); ++k, ++sp_idx) {
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      IterVar sp_ax = this->dim2var_maps[i].at(sp_dim.as<DimensionNode>()).iv;

      PrimExpr inlined_arg;
      if (sp_dim->type <= DimensionNode::kRangeDim) {
	inlined_arg = sp_ax->var;
      }
      else {
	CHECK(dim2var_maps[i].count(sp_dim.as<DimensionNode>())) << sp_dim->name;
	auto ufun = dim2var_maps[i].at(sp_dim.as<DimensionNode>()).value_expr;
	Array<Dimension> loop_dims;
	Array<PrimExpr> axis_vars;
	for (const auto& it: dim2var_maps[i]) {
	  if (it.first->type <= DimensionNode::kRangeDim) {
  	    loop_dims.push_back(GetRef<Dimension>(it.first));
	    axis_vars.push_back(it.second.iv->var);
	  }
	}
	inlined_arg = UninterpFun::MakeCallTo(ufun, axis_vars, loop_dims);
      }

      IntSet arg_intset = EvalSet(inlined_arg, dom_map);

      const arith::IntervalSetNode* arg_interval = arg_intset.as<arith::IntervalSetNode>();
      if (arg_interval) {
	PrimExpr shape_i_min_value = make_zero(t->shape[k].dtype());
	PrimExpr shape_i_max_value = t->shape[k] - 1;
	PrimExpr min_value = arg_interval->min_value;
	PrimExpr max_value = arg_interval->max_value;
	// Prefer the shape bounds only when we can prove they are tighter.
	if (arith::is_neg_inf(min_value) ||
	    analyzer->CanProve(shape_i_min_value >= min_value)) {
	  min_value = shape_i_min_value;
	}
	if (arith::is_pos_inf(max_value) ||
	    analyzer->CanProve(shape_i_max_value <= max_value)) {
	  max_value = shape_i_max_value;
	}

	if (tdom) { tdom->data[k].push_back(IntSet::interval(min_value, max_value)); }
      } else {
	if (tdom) { tdom->data[k].push_back(arg_intset); }
      }
    }
  }
}

void SingleKernelEnvelopeOpNode::GatherBound(
    const Operation& self,
    const std::unordered_map<Tensor, TensorDom>& tensor_dom,
    std::unordered_map<IterVar, Range>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
  std::vector<Tensor> output(this->num_outputs());
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = self.output(i);
  }

  // Update for spatial axis.
  size_t sp_idx = 0;
  for (size_t i = 0; i < output.size(); ++i) {
    const TensorDom& d = tensor_dom.at(output[i]);
    Map<IterVar, IntSet> lv_sets_map;
    for (size_t k = 0; k < this->inputs[i]->shape.size(); ++k, ++sp_idx) {
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      IterVar sp_ax = this->dim2var_maps[i].at(sp_dim.as<DimensionNode>()).iv;

      IntSet iv_set = arith::Union(d.data[k]);
      if (sp_dim->type <= DimensionNode::kRangeDim) {
	// CHECK(/* Check if loop dim */)
	IterVar lv = this->GetIterVarFromDim(i, sp_dim);
	if (lv_sets_map.count(lv)) {
	  lv_sets_map.Set(lv, arith::Union({ lv_sets_map.at(lv), iv_set }));
	} else {
	  lv_sets_map.Set(lv, iv_set);
	}
      } else {
	Map<Dimension, IntSet> lv_sets =
	  arith::ProjectInverse(iv_set, dim2var_maps[i].at(sp_dim.operator->()).value_expr);
	if (lv_sets.defined()) {
	  for (auto pair: lv_sets) {
	    Dimension dim = pair.first;
	    IntSet lv_set = pair.second;
	    IterVar lv = this->GetIterVarFromDim(i, dim);
	    if (lv_sets_map.count(lv)) {
	      lv_sets_map.Set(lv, arith::Union({ lv_sets_map.at(lv), lv_set }));
	    } else {
	      lv_sets_map.Set(lv, lv_set);
	    }
	  }
	}
      }
    }

    for (auto it: lv_sets_map) {
      if (out_dom_map->find(it.first) == out_dom_map->end()) {
	std::cout << "[GBSc] " << it.first->var << " " << it.second.cover_range(it.first->dom) << std::endl;
	(*out_dom_map)[it.first] = it.second.cover_range(it.first->dom);
      }
    }

    for (auto sp_dim: this->spatial_dimensions_) {
      IterVar sp_ax = this->dim2var_maps[i].at(sp_dim.as<DimensionNode>()).iv;
      if (out_dom_map->find(sp_ax) == out_dom_map->end()) {
	std::cout << "[GBSc] " << sp_ax->var << " " << sp_ax->dom << std::endl;
	(*out_dom_map)[sp_ax] = sp_ax->dom;
      }
    }
  }
}

Stmt SingleKernelEnvelopeOpNode::BuildRealize(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    const Stmt& body) const {
  CHECK_EQ(stage->op.get(), this);
  Stmt ret = body;
  size_t sp_idx = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    Tensor t = stage->op.output(i);
    CHECK_EQ(static_cast<size_t>(t->value_index), i);
    Region bounds;
    for (size_t k = 0; k < this->inputs[i]->shape.size(); ++k, ++sp_idx) {
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      IterVar sp_ax = this->GetDimVarEntry(i, sp_dim).iv;
      bounds.push_back(dom_map.count(sp_ax) ? dom_map.at(sp_ax) : sp_ax->dom);
    }
    ret = tir::RealizeNode::make(t->op, t->value_index, t->dtype,
                            bounds, const_true(), ret);
  }
  return ret;
}

Stmt SingleKernelEnvelopeOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  Stmt provide = AttrStmtNode::make(
      stage->op, attr::single_kernel_input_scope, 0,
      EvaluateNode::make(0));

  std::unordered_map<IterVar, PrimExpr> vmap;
  std::unordered_set<IterVar> empty;
  auto nest = MakeLoopNest(
      stage, dom_map, 0, false, empty, &vmap, debug_keep_trivial_loop);
  nest.push_back(
      MakeIfNest(
          MakeBoundCheck(stage, dom_map, vmap, false, empty)));
  return MergeNest(nest, provide);
  // return EvaluateNode::make(0);
}
}  // namespace te
}  // namespace tvm
