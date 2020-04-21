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
.set_dispatch<ScanOpNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const ScanOpNode*>(node.get());
    p->stream << "scan(" << op->name << ", " << op << ")";
});
TVM_REGISTER_NODE_TYPE(ScanOpNode);

inline bool prove_equal(PrimExpr lhs, PrimExpr rhs) {
  return is_zero(tir::Simplify(lhs - rhs));
}

int ScanOpNode::num_outputs() const {
  return static_cast<int>(update.size());
}
Array<IterVar> ScanOpNode::root_iter_vars() const {
  Array<IterVar> ret;
  for (auto it: dim2var_map) {
    if (it.first->type <= DimensionNode::kRangeDim) {
      ret.push_back(it.second.iv);
    }
  }
  return ret;
}

DataType ScanOpNode::output_dtype(size_t i) const {
  return update[i]->dtype;
}

Array<PrimExpr> ScanOpNode::output_shape(size_t i) const {
  CHECK_LT(i, state_placeholder.size());
  return state_placeholder[i]->shape;
}

Operation ScanOpNode::make(std::string name,
                           std::string tag,
                           Map<std::string, ObjectRef> attrs,
			   UninterpFun range_uf,
                           Dimension scan_dim,
                           Array<Tensor> init,
                           Array<Tensor> update,
                           Array<Tensor> state_placeholder,
                           Array<Tensor> inputs) {
  if (!attrs.defined()) {
    attrs = Map<std::string, ObjectRef>();
  }
  auto n = make_object<ScanOpNode>();
  CHECK_EQ(init.size(), update.size());
  CHECK_EQ(init.size(), state_placeholder.size());

  for (size_t i = 0; i < init.size(); ++i) {
    CHECK_EQ(init[i]->dtype, state_placeholder[i]->dtype);
    CHECK_EQ(init[i]->dtype, update[i]->dtype);
    CHECK_EQ(state_placeholder[i].ndim(), init[i].ndim())
        << "The dimension of init need to match state_placeholder";
    CHECK_EQ(update[i].ndim(), state_placeholder[i].ndim())
        << "The update.ndim need to be state_placeholder.ndim - 1";
    for (size_t k = 0;  k < update[i].ndim(); ++k) {
      CHECK(prove_equal(update[i]->shape[k], state_placeholder[i]->shape[k]));
    }

    for (size_t k = 1;  k < init[i].ndim(); ++k) {
      CHECK(prove_equal(
          init[i]->shape[k], state_placeholder[i]->shape[k]));
    }
  }

  IterVar axis;
  {
    Array<PrimExpr> args;
    Array<Dimension> arg_dims;
    // Create scan_axis
    for (size_t i = 0; i < update.size(); ++i) {
      Tensor t = update[i];
      auto update_op = t->op.as<ComputeOpNode>();
      CHECK(update_op) << "Only ComputeOp allowed to be the update for a scan";

      for (size_t k = 0; k < update_op->root_index_dimensions.size(); ++k) {
	auto dim = update_op->root_index_dimensions[k];
	if (range_uf->dimensions.Contains(dim)) {
	  auto entry = update_op->GetDimVarEntry(dim);
	  IterVar iv = IterVarNode::make(entry.iv->dom,
					 entry.iv->var.copy_with_suffix(".sc"),
					 dim->type == DimensionNode::kScanDim ? kOrdered : kOpaque,
					 entry.iv->thread_tag);
	  args.push_back(iv->var);
	  arg_dims.push_back(dim);
	  n->dim2var_map[dim.as<DimensionNode>()] = { entry.dim, iv, entry.value_expr };
	}
      }

      for (auto dim: update_op->loop_dimensions) {
	if (!n->dim2var_map.count(dim.as<DimensionNode>())) {
	  if (range_uf->dimensions.Contains(dim)) {
	    auto entry = update_op->GetDimVarEntry(dim);
	    IterVar iv = IterVarNode::make(entry.iv->dom,
					   entry.iv->var.copy_with_suffix(".sc"),
					   dim->type == DimensionNode::kScanDim ? kOrdered : kOpaque,
					   entry.iv->thread_tag);
	    args.push_back(iv->var);
	    arg_dims.push_back(dim);
	    n->dim2var_map[dim.as<DimensionNode>()] = { entry.dim, iv, entry.value_expr };
	  }
	}
      }

      if (args.size() == range_uf->dimensions.size()) break;
    }

    CHECK_EQ(args.size(), range_uf->parameters.size());

    PrimExpr range_max = CallNode::make(DataType::Int(32), range_uf->fname, args,
					CallNode::PureExtern, arg_dims, range_uf, 0);
    axis = IterVarNode::make(Range(0, range_max), Var(name + ".idx"), kOrdered, "");
  }

  for (size_t i = 0; i < update.size(); ++i) {
    Tensor t = update[i];
    auto update_op = t->op.as<ComputeOpNode>();
    CHECK(update_op) << "Only ComputeOp allowed to be the update for a scan";

    for (size_t k = 0; k < update_op->root_index_dimensions.size(); ++k) {
      auto dim = update_op->root_index_dimensions[k];
      if (!n->dim2var_map.count(dim.as<DimensionNode>())) {

	IterVar iv = axis;
	if (dim != scan_dim) {
	  // setup spatial axis
	  std::ostringstream spatial_name;
	  spatial_name << name << ".out" << i << ".i" << k;
	  iv = IterVarNode::make(Range::make_by_min_extent(0, update[i]->shape[k]),
				 Var(spatial_name.str()),
				 dim->type == DimensionNode::kScanDim ? kOrdered : kOpaque);
	}

	auto entry = update_op->GetDimVarEntry(dim);
	n->dim2var_map[dim.as<DimensionNode>()] = { entry.dim, iv, entry.value_expr };
      }
    }

    for (auto dim: update_op->loop_dimensions) {
      if (!n->dim2var_map.count(dim.as<DimensionNode>())) {
	IterVar iv = axis;
	auto entry = update_op->GetDimVarEntry(dim);
	if (dim != scan_dim) {
	  iv = IterVarNode::make(entry.iv->dom,
				 entry.iv->var.copy_with_suffix(".sc"),
				 dim->type == DimensionNode::kScanDim ? kOrdered : kOpaque,
				 entry.iv->thread_tag);
	}
	n->dim2var_map[dim.as<DimensionNode>()] = { entry.dim, iv, entry.value_expr };
      }
    }

    for (size_t k = 0; k < update_op->root_index_dimensions.size(); ++k) {
      auto dim = update_op->root_index_dimensions[k];
      n->spatial_dimensions_.push_back(dim);
      n->spatial_axis_.push_back(n->dim2var_map.at(dim.as<DimensionNode>()).iv);
    }
  }

  n->scan_dim = std::move(scan_dim);
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->scan_axis = std::move(axis);
  n->init = std::move(init);
  n->update = std::move(update);
  n->state_placeholder = std::move(state_placeholder);
  n->inputs = std::move(inputs);
  return Operation(n);
}

// TVM_REGISTER_GLOBAL("te.ScanOp")
// .set_body_typed(ScanOpNode::make);

TVM_REGISTER_GLOBAL("te.ScanOp")
.set_body_typed([](std::string name,
		   std::string tag,
		   Map<std::string, ObjectRef> attrs,
		   UninterpFun axis_range_uf,
		   Dimension scan_dim,
		   Array<Tensor> init,
		   Array<Tensor> update,
		   Array<Tensor> state_placeholder,
		   Array<Tensor> inputs) {
		  return ScanOpNode::make(name, tag, attrs, axis_range_uf, scan_dim,
					  init, update, state_placeholder, inputs);
		});



Array<Tensor> scan(Dimension scan_dim,
                   Array<Tensor> init,
                   Array<Tensor> update,
                   Array<Tensor> state_placeholder,
                   Array<Tensor> inputs,
		   std::string name,
                   std::string tag,
                   Map<std::string, ObjectRef> attrs) {
  // IterVar scan_axis =
  //     IterVarNode::make(
  //         Range::make_by_min_extent(
  //             init[0]->shape[0], update[0]->shape[0] - init[0]->shape[0]),
  //         Var(name + ".idx"), kOrdered);
  // Operation op = ScanOpNode::make(
  //     name, tag, attrs, scan_axis, scan_dim,
  //     init, update, state_placeholder, inputs);

  PrimExpr max = update[0]->shape[0] - init[0]->shape[0];
  UninterpFun uf = UninterpFunNode::make("scan_extent", Range(max, max), {}, {}, max);
  Operation op = ScanOpNode::make(
      name, tag, attrs, uf, scan_dim,
      init, update, state_placeholder, inputs);

  Array<Tensor> res;
  for (int i = 0; i < op->num_outputs(); ++i) {
    res.push_back(op.output(i));
  }
  return res;
}

  Array<Tensor> ScanOpNode::InputTensors() const {
  Array<Tensor> ret;
  for (Tensor t : init) {
    ret.push_back(t);
  }
  for (Tensor t : update) {
    ret.push_back(t);
  }
  return ret;
}

Operation ScanOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  auto n = make_object<ScanOpNode>(*this);
  for (size_t i = 0; i < n->init.size(); ++i) {
    if (rmap.count(n->init[i])) {
      n->init.Set(i, rmap.at(n->init[i]));
    }
    if (rmap.count(n->update[i])) {
      n->update.Set(i, rmap.at(n->update[i]));
    }
  }
  if (!n->init.same_as(init) ||
      !n->update.same_as(update)) {
    return Operation(n);
  } else {
    return self;
  }
}

void ScanOpNode::PropBoundToInputs(
    const Operation& self,
    arith::Analyzer* analyzer,
    const std::unordered_map<const VarNode*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
  bool print = false;//(self->name == "child_sum");
  if (print) std::cout << "[PBI] Op " << self->name << std::endl;
  for (size_t i = 0, sp_idx = 0; i < this->init.size(); ++i) {
    TensorDom* init_dom = nullptr;
    TensorDom* update_dom = nullptr;
    if (out_dom_map->count(this->init[i])) {
      init_dom = &out_dom_map->at(this->init[i]);
    }
    if (out_dom_map->count(this->update[i])) {
      update_dom = &out_dom_map->at(this->update[i]);
    }
    // first dimension, always needed.
    // if (init_dom) {
    //   std::vector<IntSet> this_scan_data;
    //   this_scan_data.push_back(IntSet::range(
    // 	     Range::make_by_min_extent(0, this->init[i]->shape[0])));
    //   init_dom->scan_axis_data.push_back(this_scan_data);
    // }
    // if (update_dom) {
    //   std::vector<IntSet> this_scan_data;
    //   this_scan_data.push_back(dom_map.at(this->scan_axis->var.get()));
    //   update_dom->scan_axis_data.push_back(this_scan_data);
    // }

    // The update dimensions
    for (size_t k = 0; k < this->update[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = this->spatial_axis_[sp_idx];
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      if (init_dom) {
        init_dom->data[k].push_back(dom_map.at(sp_ax->var.get()));
      }

      if (update_dom) {
	Tensor t = update[i];
	PrimExpr inlined_arg;
	if (sp_dim->type <= DimensionNode::kRangeDim) {
	  inlined_arg = sp_ax->var;
	}
	else {
	  Array<Dimension> loop_dims;
	  Array<PrimExpr> axis_vars;
	  for (auto it: dim2var_map) {
	    if (it.first->type <= DimensionNode::kRangeDim) {
	      loop_dims.push_back(GetRef<Dimension>(it.first));
	      axis_vars.push_back(it.second.iv->var);
	    }
	  }
	  CHECK(dim2var_map.count(sp_dim.as<DimensionNode>())) << sp_dim->name;
	  auto ufun = dim2var_map.at(sp_dim.as<DimensionNode>()).value_expr;
	  auto dtype = DataType::Int(32);
	  auto fun_name = ufun->func_name();
	  inlined_arg = CallNode::make(dtype, fun_name,
				       axis_vars, CallNode::CallType::PureExtern,
				       loop_dims, ufun, 0);
	}

	IntSet arg_intset = EvalSet(inlined_arg, dom_map);
	// if (print) std::cout << "[PBI]   Arg intset for " << inlined_arg << " " << arg_intset << std::endl;

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
	  update_dom->data[k].push_back(IntSet::interval(min_value, max_value));
	} else {
	  update_dom->data[k].push_back(arg_intset);
	}
      }
    }
  }
}

DimVarEntry ScanOpNode::GetDimVarEntry(Dimension dim, bool only_loop_dims) const {
  auto it = this->dim2var_map.find(dim.as<DimensionNode>());
  CHECK(it != this->dim2var_map.end()) << "No such dimension " << dim->name;
  return it->second;
}

IterVar ScanOpNode::GetIterVarFromDim(Dimension dim, bool only_loop_dims) const {
  return GetDimVarEntry(dim, only_loop_dims).iv;
}

void ScanOpNode::GatherBound(
    const Operation& self,
    const std::unordered_map<Tensor, TensorDom>& tensor_dom,
    std::unordered_map<IterVar, Range>* out_dom_map) const {
  bool print = false;//(self->name == "child_sum");
  CHECK_EQ(self.operator->(), this);
  CHECK(!out_dom_map->count(this->scan_axis));
  std::vector<Tensor> output(this->num_outputs());
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = self.output(i);
  }

  // Update for time axis.
  // std::vector<IntSet> time_dom;
  // for (size_t i = 0; i < output.size(); ++i) {
  //   const TensorDom& d = tensor_dom.at(output[i]);
  //   time_dom.insert(time_dom.end(), d.data[0].begin(), d.data[0].end());
  // }
  // CHECK(!out_dom_map->count(this->scan_axis));
  // Range sdom = this->scan_axis->dom;
  // Range r = arith::Union(time_dom).cover_range(sdom);
  // (*out_dom_map)[this->scan_axis] = Range::make_by_min_extent(
  //     sdom->min, tir::Simplify(r->extent + r->min - sdom->min));

  Map<IterVar, PrimExpr> fix_pt = ScanFixPointAnalysis(self);


  // Update for spatial axis.
  size_t sp_idx = 0;
  if (print) std::cout << "[GBS] Op " << self->name << std::endl;
  for (size_t i = 0; i < output.size(); ++i) {
    const TensorDom& d = tensor_dom.at(output[i]);
    Map<IterVar, IntSet> lv_sets_map;
    for (size_t k = 0; k < this->update[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = this->spatial_axis_[sp_idx];
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];

      // if (print) std::cout << "[GBS]  Dim " << sp_dim->name << " " << sp_ax->var->name_hint << " " << fix_pt[sp_ax] << std::endl;
      CHECK(fix_pt.count(sp_ax));
      if (fix_pt[sp_ax].as<tir::IntImmNode>()->value) {
        // fix point, we can slice it.

	IntSet iv_set = arith::Union(d.data[k]);
	// if (print) std::cout << "[GBS]  Dim0Set " << sp_dim->name << " " << iv_set << std::endl;
      	if (sp_dim->type <= DimensionNode::kRangeDim) {
	  // CHECK(/* Check if loop dim */)
	  IterVar lv = this->GetIterVarFromDim(sp_dim);
	  // if (print) std::cout << "[GBS]   Dim0.0 " << sp_dim->name << " " << lv << " " << iv_set << std::endl;
	  if (lv_sets_map.count(lv)) {
	    lv_sets_map.Set(lv, arith::Union({ lv_sets_map.at(lv), iv_set }));
	  }
	  else {
	    lv_sets_map.Set(lv, iv_set);
	  }
	}
	else {
	  Map<Dimension, IntSet> lv_sets =
	    arith::ProjectInverse(iv_set, dim2var_map.at(sp_dim.operator->()).value_expr);
	  // if (print) std::cout << "[GBS]  Dim0.1S " << sp_dim->name << " " << lv_sets << std::endl;
	  if (lv_sets.defined()) {
	    for (auto pair: lv_sets) {
	      Dimension dim = pair.first;
	      IntSet lv_set = pair.second;
	      IterVar lv = this->GetIterVarFromDim(dim);
	      // if (print) std::cout << "[GBS]   Dim0.1 " << sp_dim->name << " " << dim->name << " " << lv << " " << iv_set << std::endl;
	      if (lv_sets_map.count(lv)) {
		lv_sets_map.Set(lv, arith::Union({ lv_sets_map.at(lv), lv_set }));
	      }
	      else {
		lv_sets_map.Set(lv, lv_set);
	      }
	    }
	  }
	}
      } else {
	// if (print) std::cout << "[GBS] Dim0 " << sp_dim->name << " No fixed point" << std::endl;
        // not a fix point, need to include everything.
	if (sp_dim->type <= DimensionNode::kRangeDim) {
	  lv_sets_map.Set(sp_ax, IntSet::range(sp_ax->dom));
	}
	else {
	  for (auto arg_dim: dim2var_map.at(sp_dim.operator->()).value_expr->dimensions) {
	    IterVar loop_iv = this->GetIterVarFromDim(arg_dim);
	    IntSet set = IntSet::range(loop_iv->dom);
	    if (lv_sets_map.count(loop_iv)) {
	      lv_sets_map.Set(loop_iv, arith::Union({ lv_sets_map.at(loop_iv), set }));
	    }
	    else {
	      lv_sets_map.Set(loop_iv, set);
	    }
	  }
	}
      }
    }

    for (auto it: lv_sets_map) {
      if (out_dom_map->find(it.first) == out_dom_map->end()) {
	(*out_dom_map)[it.first] = it.second.cover_range(it.first->dom);
	// if (print) std::cout << "[GBS]  Dim1 " << it.first << " " <<
		     // UninterpFun::InlineUninterpFunCalls((*out_dom_map)[it.first]) << std::endl;
      }
    }

    for (size_t i = 0; i < this->spatial_axis_.size(); ++i) {
      if (out_dom_map->find(this->spatial_axis_[i]) == out_dom_map->end()) {
	(*out_dom_map)[this->spatial_axis_[i]] = this->spatial_axis_[i]->dom;
	// if (print) std::cout << "[GBS]  Dim2 " << this->spatial_axis_[i] << " " <<
		     // UninterpFun::InlineUninterpFunCalls((*out_dom_map)[this->spatial_axis_[i]]) << std::endl;
      }
    }

    for (auto it: dim2var_map) {
      if (it.first->type <= DimensionNode::kRangeDim) {
	if (print) std::cout << "[GBS]   DimF " << it.first->name << " " << it.second.iv->var->name_hint << " " <<
		     UninterpFun::InlineUninterpFunCalls((*out_dom_map)[it.second.iv]) << std::endl;
      }
    }
  }
}

Stmt ScanOpNode::BuildRealize(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    const Stmt& body) const {
  CHECK_EQ(stage->op.get(), this);
  Range sdom = dom_map.at(this->scan_axis);
  Range tdom = Range::make_by_min_extent(
      0, tir::Simplify(sdom->extent + sdom->min));
  Stmt ret = body;
  size_t sp_idx = 0;
  for (size_t i = 0; i < update.size(); ++i) {
    Tensor t = stage->op.output(i);
    CHECK_EQ(static_cast<size_t>(t->value_index), i);
    Region bounds;
    bounds.push_back(tdom);
    for (size_t k = 1; k < this->update[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = this->spatial_axis_[sp_idx];
      bounds.push_back(dom_map.count(sp_ax) ? dom_map.at(sp_ax) : sp_ax->dom);
    }
    ret = tir::RealizeNode::make(t->op, t->value_index, t->dtype,
                            bounds, const_true(), ret);
  }
  return ret;
}

Stmt ScanOpNode::BuildProvide(
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
