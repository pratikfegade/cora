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

namespace tvm {
namespace te {
using namespace tir;

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<ScanEnvelopeOpNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const ScanEnvelopeOpNode*>(node.get());
    p->stream << "scan_envelope(" << op->name << ", " << op << ")";
});
TVM_REGISTER_NODE_TYPE(ScanEnvelopeOpNode);

int ScanEnvelopeOpNode::num_outputs() const {
  return static_cast<int>(scans[0].size());
}
inline bool prove_equal(PrimExpr lhs, PrimExpr rhs) {
  return is_zero(tir::Simplify(lhs - rhs));
}
Array<IterVar> ScanEnvelopeOpNode::root_iter_vars() const {
  Array<IterVar> ret;
  for (auto it: dim2var_map) {
    if (it.first->type <= DimensionNode::kRangeDim) {
      ret.push_back(it.second.iv);
    }
  }
  return ret;
}

DataType ScanEnvelopeOpNode::output_dtype(size_t i) const {
  return scans[0][i]->dtype;
}

Array<PrimExpr> ScanEnvelopeOpNode::output_shape(size_t i) const {
  return scans[0][i]->shape;
}

Operation ScanEnvelopeOpNode::make(std::string name,
                           std::string tag,
                           Map<std::string, ObjectRef> attrs,
                           Array<Array<Tensor>> scans) {
  if (!attrs.defined()) {
    attrs = Map<std::string, ObjectRef>();
  }
  auto n = make_object<ScanEnvelopeOpNode>();

  for (auto scan: scans) {
    CHECK(scan[0]->op.as<ScanOpNode>()) << "All participating ops should be scans";

    for (auto t: scan) {
      CHECK_EQ(scan[0]->op, t->op) << "Tensors belong to different operations";
    }
  }

  for (auto scan: scans) {
    CHECK_EQ(scans[0].size(), scan.size()) <<
      "All participating scans should have the same number of outputs";
  }

  auto num_outputs = scans[0].size();

  for (size_t i = 0; i < num_outputs; ++i) {
    for (auto scan: scans) {
      CHECK_EQ(scan[i]->dtype, scans[0][i]->dtype);
      CHECK_EQ(scan[i].ndim(), scans[0][i].ndim());
      for (size_t k = 0; k < scan[i].ndim(); ++k) {
	CHECK(prove_equal(scan[i]->shape[k], scans[0][i]->shape[k]));
      }
    }
  }

  std::vector<const ScanOpNode*> scan_ops;
  for (auto scan: scans) {
    scan_ops.push_back(scan[0]->op.as<ScanOpNode>());
  }

  for (auto scan_op: scan_ops) {
    CHECK_EQ(scan_op->scan_dim, scan_ops[0]->scan_dim);
    for (size_t j = 0; j < scan_ops[0]->spatial_dimensions_.size(); ++j) {
      CHECK_EQ(scan_op->spatial_dimensions_[j], scan_ops[0]->spatial_dimensions_[j]);
    }
  }

  for (auto it: scan_ops[0]->dim2var_map) {
    Dimension dim = GetRef<Dimension>(it.first);
    auto entry = it.second;

    IterVar iv = IterVarNode::make(entry.iv->dom,
				   entry.iv->var.copy_with_suffix("env"),
				   kOpaque);
    n->dim2var_map[it.first] = { dim, iv, entry.value_expr };
    n->spatial_dimensions_.push_back(dim);
  }

  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->scans = std::move(scans);
  n->scan_ops = std::move(scan_ops);
  return Operation(n);
}

TVM_REGISTER_GLOBAL("te.ScanEnvelopeOp")
.set_body_typed([](std::string name,
		   std::string tag,
		   Map<std::string, ObjectRef> attrs,
		   Array<Array<Tensor>> scan_tensors) {
		  return ScanEnvelopeOpNode::make(name, tag, attrs, scan_tensors);
		});


Array<Tensor> ScanEnvelopeOpNode::InputTensors() const {
  Array<Tensor> ret;
  for (auto scan : scans) {
    for (auto t : scan) {
      ret.push_back(t);
    }
  }
  return ret;
}

Operation ScanEnvelopeOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  auto n = make_object<ScanEnvelopeOpNode>(*this);
  for (size_t i = 0; i < n->scans.size(); ++i) {
    auto scan = scans[i];
    for (size_t j = 0; j < scan.size(); ++j) {
      if (rmap.count(scan[j])) {
	scan.Set(j, rmap.at(scan[j]));
      }
    }
  }
  if (!n->scans.same_as(scans)) {
    return Operation(n);
  } else {
    return self;
  }
}

void ScanEnvelopeOpNode::PropBoundToInputs(
    const Operation& self,
    arith::Analyzer* analyzer,
    const std::unordered_map<const VarNode*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
  for (int i = 0, sp_idx = 0; i < this->num_outputs(); ++i) {
    // The update dimensions
    for (size_t k = 0; k < this->scans[0][i]->shape.size(); ++k, ++sp_idx) {
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      IterVar sp_ax = this->dim2var_map.at(sp_dim.as<DimensionNode>()).iv;

      PrimExpr inlined_arg;
      if (sp_dim->type <= DimensionNode::kRangeDim) {
	inlined_arg = sp_ax->var;
      }
      else {
	CHECK(dim2var_map.count(sp_dim.as<DimensionNode>())) << sp_dim->name;
	auto ufun = dim2var_map.at(sp_dim.as<DimensionNode>()).value_expr;
	auto dtype = DataType::Int(32);
	auto fun_name = ufun->func_name();

	Array<Dimension> loop_dims;
	Array<PrimExpr> axis_vars;
	for (auto it: dim2var_map) {
	  if (it.first->type <= DimensionNode::kRangeDim) {
  	    loop_dims.push_back(GetRef<Dimension>(it.first));
	    axis_vars.push_back(it.second.iv->var);
	  }
	}
	inlined_arg = CallNode::make(dtype, fun_name,
				     axis_vars, CallNode::CallType::PureExtern,
				     loop_dims, ufun, 0);
      }

      IntSet arg_intset = EvalSet(inlined_arg, dom_map);

      const arith::IntervalSetNode* arg_interval = arg_intset.as<arith::IntervalSetNode>();
      if (arg_interval) {
	PrimExpr shape_i_min_value = make_zero(scans[0][i]->shape[k].dtype());
	PrimExpr shape_i_max_value = scans[0][i]->shape[k] - 1;
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

	for (size_t j = 0; j < this->scans.size(); ++j) {
	  Tensor t = scans[j][i];
	  if (out_dom_map->count(t)) {
	    TensorDom* update_dom = &out_dom_map->at(t);
	    update_dom->data[k].push_back(IntSet::interval(min_value, max_value));
	  }
	}
      } else {
	for (size_t j = 0; j < this->scans.size(); ++j) {
	  Tensor t = scans[j][i];
	  if (out_dom_map->count(t)) {
	    TensorDom* update_dom = &out_dom_map->at(t);
	    update_dom->data[k].push_back(arg_intset);
	  }
	}
      }
    }
  }
}

DimVarEntry ScanEnvelopeOpNode::GetDimVarEntry(Dimension dim, bool only_loop_dims) const {
  auto it = this->dim2var_map.find(dim.as<DimensionNode>());
  CHECK(it != this->dim2var_map.end()) << "No such dimension " << dim->name;
  return it->second;
}

IterVar ScanEnvelopeOpNode::GetIterVarFromDim(Dimension dim, bool only_loop_dims) const {
  return GetDimVarEntry(dim, only_loop_dims).iv;
}

void ScanEnvelopeOpNode::GatherBound(
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
    for (size_t k = 0; k < this->scans[0][i]->shape.size(); ++k, ++sp_idx) {
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      IterVar sp_ax = this->dim2var_map.at(sp_dim.as<DimensionNode>()).iv;

      IntSet iv_set = arith::Union(d.data[k]);
      if (sp_dim->type <= DimensionNode::kRangeDim) {
	// CHECK(/* Check if loop dim */)
	IterVar lv = this->GetIterVarFromDim(sp_dim);
	if (lv_sets_map.count(lv)) {
	  lv_sets_map.Set(lv, arith::Union({ lv_sets_map.at(lv), iv_set }));
	} else {
	  lv_sets_map.Set(lv, iv_set);
	}
      } else {
	Map<Dimension, IntSet> lv_sets =
	  arith::ProjectInverse(iv_set, dim2var_map.at(sp_dim.operator->()).value_expr);
	if (lv_sets.defined()) {
	  for (auto pair: lv_sets) {
	    Dimension dim = pair.first;
	    IntSet lv_set = pair.second;
	    IterVar lv = this->GetIterVarFromDim(dim);
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
	(*out_dom_map)[it.first] = it.second.cover_range(it.first->dom);
      }
    }

    for (auto sp_dim: this->spatial_dimensions_) {
      IterVar sp_ax = this->dim2var_map.at(sp_dim.as<DimensionNode>()).iv;
      if (out_dom_map->find(sp_ax) == out_dom_map->end()) {
	(*out_dom_map)[sp_ax] = sp_ax->dom;
      }
    }
  }
}

Stmt ScanEnvelopeOpNode::BuildRealize(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    const Stmt& body) const {
  CHECK_EQ(stage->op.get(), this);
  Stmt ret = body;
  size_t sp_idx = 0;
  for (size_t i = 0; i < scans[0].size(); ++i) {
    Tensor t = stage->op.output(i);
    CHECK_EQ(static_cast<size_t>(t->value_index), i);
    Region bounds;
    for (size_t k = 0; k < this->scans[0][i]->shape.size(); ++k, ++sp_idx) {
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      IterVar sp_ax = this->GetDimVarEntry(sp_dim).iv;
      bounds.push_back(dom_map.count(sp_ax) ? dom_map.at(sp_ax) : sp_ax->dom);
    }
    ret = tir::RealizeNode::make(t->op, t->value_index, t->dtype,
                            bounds, const_true(), ret);
  }
  return ret;
}

Stmt ScanEnvelopeOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  Stmt provide = AttrStmtNode::make(
      stage->op, attr::scan_update_scope, this->scan_axis->var,
      EvaluateNode::make(0));
  Stmt init = AttrStmtNode::make(
      stage->op, attr::scan_init_scope, 0,
      EvaluateNode::make(0));
  size_t begin_scan = 0;
  for (size_t  i = 0; i < stage->leaf_iter_vars.size(); ++i) {
    if (stage->leaf_iter_vars[i]->iter_type == kThreadIndex) {
      CHECK_EQ(begin_scan, i);
      begin_scan = i + 1;
    }
  }
  std::unordered_map<IterVar, PrimExpr> vmap;
  std::unordered_set<IterVar> empty;
  auto nest = MakeLoopNest(
      stage, dom_map, 0, false, empty, &vmap, debug_keep_trivial_loop);
  nest[begin_scan].push_back(init);
  nest.push_back(
      MakeIfNest(
          MakeBoundCheck(stage, dom_map, vmap, false, empty)));
  return MergeNest(nest, provide);
}
}  // namespace te
}  // namespace tvm
