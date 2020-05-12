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
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>

#include "../../arith/interval_set.h"
#include "../../tir/ir/var_replacer.h"
#include "../schedule/graph.h"
#include "op_util.h"

namespace tvm {
namespace te {
using namespace tir;

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SpecializationEnvelopeOpNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SpecializationEnvelopeOpNode*>(node.get());
      p->stream << "specialization(" << op->name << ", " << op << ")";
    });
TVM_REGISTER_NODE_TYPE(SpecializationEnvelopeOpNode);

int SpecializationEnvelopeOpNode::num_outputs() const { return static_cast<int>(inputs[0].size()); }
inline bool prove_equal(PrimExpr lhs, PrimExpr rhs) { return is_zero(tir::Simplify(lhs - rhs)); }
Array<IterVar> SpecializationEnvelopeOpNode::root_iter_vars() const {
  Array<IterVar> ret;
  for (auto dim2var_map : dim2var_maps) {
    for (auto it : dim2var_map) {
      if (it.first->type <= DimensionNode::kRangeDim) {
        ret.push_back(it.second.iv);
      }
    }
  }
  return ret;
}

DataType SpecializationEnvelopeOpNode::output_dtype(size_t i) const { return inputs[0][i]->dtype; }

Array<PrimExpr> SpecializationEnvelopeOpNode::output_shape(size_t i) const {
  return inputs[0][i]->shape;
}

Dimension SpecializationEnvelopeOpNode::GetBaseIndexDimension(size_t val_idx,
                                                              size_t dim_idx) const {
  return input_ops[0]->GetBaseIndexDimension(val_idx, dim_idx);
}

Operation SpecializationEnvelopeOpNode::make(std::string name, std::string tag,
                                             Map<std::string, ObjectRef> attrs,
                                             Array<Array<Tensor>> inputs) {
  if (!attrs.defined()) {
    attrs = Map<std::string, ObjectRef>();
  }
  auto n = make_object<SpecializationEnvelopeOpNode>();

  std::vector<const BaseVarDimOpNode*> input_ops;
  for (auto input : inputs) {
    if (input[0]->op.as<ScanOpNode>())
      input_ops.push_back(input[0]->op.as<ScanOpNode>());
    else if (input[0]->op.as<ComputeOpNode>())
      input_ops.push_back(input[0]->op.as<ComputeOpNode>());
    else
      CHECK(false) << "All participating ops should be scans or computes " << input[0]->op;

    for (auto t : input) {
      CHECK_EQ(input[0]->op, t->op) << "Tensors belong to different operations";
    }
  }

  for (auto input : inputs) {
    CHECK_EQ(inputs[0].size(), input.size())
        << "All participating inputs should have the same number of outputs";
  }

  auto num_outputs = inputs[0].size();

  for (size_t i = 0; i < num_outputs; ++i) {
    for (auto input : inputs) {
      CHECK_EQ(input[i]->dtype, inputs[0][i]->dtype);
      CHECK_EQ(input[i].ndim(), inputs[0][i].ndim());
      for (size_t k = 0; k < input[i].ndim(); ++k) {
        CHECK(prove_equal(input[i]->shape[k], inputs[0][i]->shape[k]));
      }
    }
  }

  for (auto input_op : input_ops) {
    // CHECK_EQ(input_op->input_dim, input_ops[0]->input_dim);
    for (size_t i = 0; i < num_outputs; ++i) {
      for (size_t j = 0; j < inputs[0][i].ndim(); ++j) {
        CHECK_EQ(input_op->GetBaseIndexDimension(i, j), input_ops[0]->GetBaseIndexDimension(i, j));
      }
    }
  }

  std::unordered_map<const VarNode*, PrimExpr> vmap;
  for (const auto& dim2var_map : input_ops[0]->dim2var_maps) {
    for (const auto& it : dim2var_map) {
      vmap[it.second.iv->var.as<VarNode>()] = it.second.iv->var.copy_with_suffix(".env");
    }
  }

  n->dim2var_maps = std::vector<std::unordered_map<const DimensionNode*, DimVarEntry>>(num_outputs);

  VarReplacer var_replacer(vmap);
  for (size_t i = 0; i < input_ops[0]->dim2var_maps.size(); ++i) {
    auto dim2var_map = input_ops[0]->dim2var_maps[i];
    for (auto it : dim2var_map) {
      Dimension dim = GetRef<Dimension>(it.first);
      auto entry = it.second;

      IterVar iv = IterVarNode::make(Range::make_by_min_extent(var_replacer(entry.iv->dom->min),
                                                               var_replacer(entry.iv->dom->extent)),
                                     Downcast<Var>(vmap[entry.iv->var.as<VarNode>()]), kOpaque);
      n->dim2var_maps[i][it.first] = {dim, iv, entry.value_expr};
    }
  }

  if (auto s_op = inputs[0][0]->op.as<ScanOpNode>()) {
    for (auto dim : s_op->spatial_dimensions_) {
      n->spatial_dimensions_.push_back(dim);
    }
  } else if (auto c_op = inputs[0][0]->op.as<ComputeOpNode>()) {
    for (size_t i = 0; i < num_outputs; ++i) {
      for (auto dim : c_op->root_index_dimensions) {
        n->spatial_dimensions_.push_back(dim);
      }
    }
  }

  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->inputs = std::move(inputs);
  n->input_ops = std::move(input_ops);
  return Operation(n);
}

TVM_REGISTER_GLOBAL("te.SpecializationEnvelopeOp")
    .set_body_typed([](std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                       Array<Array<Tensor>> input_tensors) {
      return SpecializationEnvelopeOpNode::make(name, tag, attrs, input_tensors);
    });

Array<Tensor> SpecializationEnvelopeOpNode::InputTensors() const {
  Array<Tensor> ret;
  for (auto input : inputs) {
    for (auto t : input) {
      ret.push_back(t);
    }
  }
  return ret;
}

Operation SpecializationEnvelopeOpNode::ReplaceInputs(
    const Operation& self, const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  Array<Array<Tensor>> new_inputs;
  bool replaced = false;
  for (size_t i = 0; i < this->inputs.size(); ++i) {
    Array<Tensor> new_input;
    for (size_t j = 0; j < this->inputs[i].size(); ++j) {
      if (rmap.count(this->inputs[i][j])) {
        new_input.push_back(rmap.at(this->inputs[i][j]));
        replaced = true;
      } else {
        new_input.push_back(this->inputs[i][j]);
      }
    }
    new_inputs.push_back(new_input);
  }

  if (replaced) {
    std::cout << "New op" << std::endl;
    return SpecializationEnvelopeOpNode::make(this->name, this->tag, this->attrs, new_inputs);
  } else {
    return self;
  }
}

void SpecializationEnvelopeOpNode::PropBoundToInputs(
    const Operation& self, arith::Analyzer* analyzer,
    const std::unordered_map<const VarNode*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
  for (int i = 0, sp_idx = 0; i < this->num_outputs(); ++i) {
    // std::cout << "[ScPBI] DimIdx " << i << std::endl;
    // The update dimensions
    for (size_t k = 0; k < this->inputs[0][i]->shape.size(); ++k, ++sp_idx) {
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      IterVar sp_ax = this->dim2var_maps[i].at(sp_dim.as<DimensionNode>()).iv;

      // std::cout << "[ScPBI]   Dim " << sp_idx << " " << sp_dim->name << std::endl;

      PrimExpr inlined_arg;
      if (sp_dim->type <= DimensionNode::kRangeDim) {
        inlined_arg = sp_ax->var;
      } else {
        CHECK(dim2var_maps[i].count(sp_dim.as<DimensionNode>())) << sp_dim->name;
        auto ufun = dim2var_maps[i].at(sp_dim.as<DimensionNode>()).value_expr;
        Array<Dimension> loop_dims;
        Array<PrimExpr> axis_vars;
        for (auto it : dim2var_maps[i]) {
          if (it.first->type <= DimensionNode::kRangeDim) {
            loop_dims.push_back(GetRef<Dimension>(it.first));
            axis_vars.push_back(it.second.iv->var);
          }
        }
        inlined_arg = UninterpFun::MakeCallTo(ufun, axis_vars, loop_dims);
      }

      IntSet arg_intset = EvalSet(inlined_arg, dom_map);
      // std::cout << "[ScPBI]     Arg intset " << inlined_arg << " " << arg_intset << std::endl;

      const arith::IntervalSetNode* arg_interval = arg_intset.as<arith::IntervalSetNode>();
      if (arg_interval) {
        PrimExpr shape_i_min_value = make_zero(inputs[0][i]->shape[k].dtype());
        PrimExpr shape_i_max_value = inputs[0][i]->shape[k] - 1;
        PrimExpr min_value = arg_interval->min_value;
        PrimExpr max_value = arg_interval->max_value;
        // Prefer the shape bounds only when we can prove they are tighter.
        if (arith::is_neg_inf(min_value) || analyzer->CanProve(shape_i_min_value >= min_value)) {
          min_value = shape_i_min_value;
        }
        if (arith::is_pos_inf(max_value) || analyzer->CanProve(shape_i_max_value <= max_value)) {
          max_value = shape_i_max_value;
        }

        for (size_t j = 0; j < this->inputs.size(); ++j) {
          Tensor t = inputs[j][i];
          if (out_dom_map->count(t)) {
            TensorDom* update_dom = &out_dom_map->at(t);
            update_dom->data[k].push_back(IntSet::interval(min_value, max_value));
            // std::cout << "[ScPBI]     Passing to " << t->op->name << " " <<
            // IntSet::interval(min_value, max_value) << std::endl;
          }
        }
      } else {
        for (size_t j = 0; j < this->inputs.size(); ++j) {
          Tensor t = inputs[j][i];
          if (out_dom_map->count(t)) {
            TensorDom* update_dom = &out_dom_map->at(t);
            update_dom->data[k].push_back(arg_intset);
            // std::cout << "[ScPBI]     Passing to " << t->op->name << " " << arg_intset <<
            // std::endl;
          }
        }
      }
    }
  }
}

void SpecializationEnvelopeOpNode::GatherBound(
    const Operation& self, const std::unordered_map<Tensor, TensorDom>& tensor_dom,
    std::unordered_map<IterVar, Range>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
  std::vector<Tensor> output(this->num_outputs());
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = self.output(i);
  }

  // Update for spatial axis.
  size_t sp_idx = 0;
  for (int i = 0; i < num_outputs(); ++i) {
    const TensorDom& d = tensor_dom.at(output[i]);
    Map<IterVar, IntSet> lv_sets_map;
    for (size_t k = 0; k < this->inputs[0][i]->shape.size(); ++k, ++sp_idx) {
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      IterVar sp_ax = this->dim2var_maps[i].at(sp_dim.as<DimensionNode>()).iv;

      IntSet iv_set = arith::Union(d.data[k]);
      if (sp_dim->type <= DimensionNode::kRangeDim) {
        // CHECK(/* Check if loop dim */)
        IterVar lv = this->GetIterVarFromDim(i, sp_dim);
        if (lv_sets_map.count(lv)) {
          lv_sets_map.Set(lv, arith::Union({lv_sets_map.at(lv), iv_set}));
        } else {
          lv_sets_map.Set(lv, iv_set);
        }
      } else {
        Map<Dimension, IntSet> lv_sets =
            arith::ProjectInverse(iv_set, dim2var_maps[i].at(sp_dim.operator->()).value_expr);
        if (lv_sets.defined()) {
          for (auto pair : lv_sets) {
            Dimension dim = pair.first;
            IntSet lv_set = pair.second;
            IterVar lv = this->GetIterVarFromDim(i, dim);
            if (lv_sets_map.count(lv)) {
              lv_sets_map.Set(lv, arith::Union({lv_sets_map.at(lv), lv_set}));
            } else {
              lv_sets_map.Set(lv, lv_set);
            }
          }
        }
      }
    }

    for (auto it : lv_sets_map) {
      if (out_dom_map->find(it.first) == out_dom_map->end()) {
        std::cout << "[GBSc] " << it.first->var << " " << it.second.cover_range(it.first->dom)
                  << std::endl;
        (*out_dom_map)[it.first] = it.second.cover_range(it.first->dom);
      }
    }

    for (auto sp_dim : this->spatial_dimensions_) {
      IterVar sp_ax = this->dim2var_maps[i].at(sp_dim.as<DimensionNode>()).iv;
      if (out_dom_map->find(sp_ax) == out_dom_map->end()) {
        std::cout << "[GBSc] " << sp_ax->var << " " << sp_ax->dom << std::endl;
        (*out_dom_map)[sp_ax] = sp_ax->dom;
      }
    }
  }
}

Stmt SpecializationEnvelopeOpNode::BuildRealize(const Stage& stage,
                                                const std::unordered_map<IterVar, Range>& dom_map,
                                                const Stmt& body) const {
  CHECK_EQ(stage->op.get(), this);
  Stmt ret = body;
  size_t sp_idx = 0;
  for (int i = 0; i < num_outputs(); ++i) {
    Tensor t = stage->op.output(i);
    CHECK_EQ(static_cast<size_t>(t->value_index), i);
    Region bounds;
    for (size_t k = 0; k < this->inputs[0][i]->shape.size(); ++k, ++sp_idx) {
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      IterVar sp_ax = this->GetDimVarEntry(i, sp_dim).iv;
      bounds.push_back(dom_map.count(sp_ax) ? dom_map.at(sp_ax) : sp_ax->dom);
    }
    ret = tir::RealizeNode::make(t->op, t->value_index, t->dtype, bounds, const_true(), ret);
  }
  return ret;
}

Stmt SpecializationEnvelopeOpNode::BuildProvide(const Stage& stage,
                                                const std::unordered_map<IterVar, Range>& dom_map,
                                                bool debug_keep_trivial_loop) const {
  // CHECK_EQ(stage->op.operator->(), this);
  // Stmt provide = AttrStmtNode::make(
  //     stage->op, attr::scan_envelope_input_scope, 0,
  //     EvaluateNode::make(0));

  // std::unordered_map<IterVar, PrimExpr> vmap;
  // std::unordered_set<IterVar> empty;
  // auto nest = MakeLoopNest(
  //     stage, dom_map, 0, false, empty, &vmap, debug_keep_trivial_loop);
  // nest.push_back(
  //     MakeIfNest(
  //         MakeBoundCheck(stage, dom_map, vmap, false, empty)));
  // return MergeNest(nest, provide);
  return EvaluateNode::make(0);
}
}  // namespace te
}  // namespace tvm
