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
#include <tvm/tir/uf_equality.h>

#include "../../arith/interval_set.h"
#include "../../tir/ir/var_replacer.h"
#include "../schedule/graph.h"
#include "op_util.h"

namespace tvm {
namespace te {
using namespace tir;

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SingleKernelEnvelopeOpNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SingleKernelEnvelopeOpNode*>(node.get());
      p->stream << "single_kernel(" << op->name << ", " << op << ")";
    });
TVM_REGISTER_NODE_TYPE(SingleKernelEnvelopeOpNode);

int SingleKernelEnvelopeOpNode::num_outputs() const { return static_cast<int>(inputs.size()); }
inline bool prove_equal(PrimExpr lhs, PrimExpr rhs) { return is_zero(tir::Simplify(lhs - rhs)); }
Array<IterVar> SingleKernelEnvelopeOpNode::root_iter_vars() const {
  Array<IterVar> ret;
  std::unordered_set<const Object*> explicit_dims;
  for (const auto& di : explicit_dimensions) {
    ret.push_back(di->iv);
    explicit_dims.insert(di->dim.get());
  }

  for (const auto& dim2var_map : dim2var_maps) {
    for (const auto& it : dim2var_map) {
      if (!ret.Contains(it.second.iv) && !explicit_dims.count(it.first)) {
        CHECK(it.first->isLoopDim());
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

Array<Dimension> SingleKernelEnvelopeOpNode::GetRootIndexDimensions(size_t val_idx) const {
  Tensor t = inputs[val_idx];
  auto op = t->op.as<BaseVarDimOpNode>();
  CHECK(op);
  return op->GetRootIndexDimensions(t->value_index);
}

std::vector<const BaseVarDimOpNode*> GetInputOps(Array<Tensor> inputs) {
  std::vector<const BaseVarDimOpNode*> input_ops;
  for (auto input : inputs) {
    if (input->op.as<ScanOpNode>())
      input_ops.push_back(input->op.as<ScanOpNode>());
    else if (input->op.as<ComputeOpNode>())
      input_ops.push_back(input->op.as<ComputeOpNode>());
    else if (input->op.as<SpecializationEnvelopeOpNode>())
      input_ops.push_back(input->op.as<SpecializationEnvelopeOpNode>());
    else
      CHECK(false) << "All participating ops should be scans or computes but instead we have a "
                   << input->op;
  }
  return input_ops;
}

Operation SingleKernelEnvelopeOpNode::make(std::string name, std::string tag,
                                           Map<std::string, ObjectRef> attrs,
                                           Array<Dimension> explicit_dims, Array<Tensor> inputs) {
  if (!attrs.defined()) {
    attrs = Map<std::string, ObjectRef>();
  }
  auto n = make_object<SingleKernelEnvelopeOpNode>();

  std::vector<const BaseVarDimOpNode*> input_ops = GetInputOps(inputs);

  auto num_outputs = inputs.size();

  std::unordered_map<const VarNode*, PrimExpr> vmap;
  std::unordered_set<const BaseVarDimOpNode*> input_ops_set(input_ops.begin(), input_ops.end());
  for (const auto& op : input_ops_set) {
    for (const auto& dim2var_map : op->dim2var_maps) {
      for (const auto& it : dim2var_map) {
        vmap[it.second.iv->var.as<VarNode>()] = it.second.iv->var.copy_with_suffix(".env");
      }
    }
  }

  n->dim2var_maps = std::vector<std::unordered_map<const DimensionNode*, DimVarEntry>>(num_outputs);
  std::unordered_map<const DimensionNode*, DimVarEntry> explicit_dim_entries;

  VarReplacer var_replacer(vmap);
  for (size_t i = 0; i < num_outputs; ++i) {
    const auto& op = input_ops[i];
    for (const auto& dim2var_map : op->dim2var_maps) {
      for (const auto& it : dim2var_map) {
        Dimension dim = GetRef<Dimension>(it.first);
        if (explicit_dims.Contains(dim)) {
          auto dim_node = dim.as<DimensionNode>();
          if (explicit_dim_entries.count(dim_node)) {
            auto entry = explicit_dim_entries.at(dim_node);
            // std::cout << "[SK] Dim " << i << " " << it.first->name << std::endl;
            n->dim2var_maps[i][it.first] = {dim, entry.iv};
          } else {
            auto entry = it.second;
            IterVar iv = IterVarNode::make(
                Range::make_by_min_extent(var_replacer(entry.iv->dom->min),
                                          var_replacer(entry.iv->dom->extent)),
                Downcast<Var>(vmap[entry.iv->var.as<VarNode>()]), entry.iv->iter_type);
            explicit_dim_entries[dim_node] = {dim, iv};
            // std::cout << "[SK] Dim1 " << dim << " " << iv << " " << entry.iv->iter_type
            // << std::endl;
            n->dim2var_maps[i][it.first] = {dim, iv};
          }
        } else {
          auto entry = it.second;
          IterVar iv =
              IterVarNode::make(Range::make_by_min_extent(var_replacer(entry.iv->dom->min),
                                                          var_replacer(entry.iv->dom->extent)),
                                Downcast<Var>(vmap[entry.iv->var.as<VarNode>()]), kLoopNestOpaque);
          // std::cout << "[SK] Dim2 " << dim << " " << iv << std::endl;
          n->dim2var_maps[i][it.first] = {dim, iv};
        }
      }
    }
  }

  for (auto dim : explicit_dims) {
    auto e = explicit_dim_entries.at(dim.as<DimensionNode>());
    n->explicit_dimensions.push_back(DimInfoNode::make(e.dim, e.iv));
  }

  for (size_t j = 0; j < inputs.size(); ++j) {
    auto op = input_ops[j];
    for (size_t i = 0; i < inputs[j].ndim(); ++i) {
      auto dim = op->GetBaseIndexDimension(inputs[j]->value_index, i);
      // std::cout << "[SK] SpDim " << dim << std::endl;
      n->spatial_dimensions_.push_back(dim);
    }
  }

  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->inputs = std::move(inputs);
  n->input_ops = std::move(input_ops);
  return Operation(n);
}

Array<Tensor> InputTensorsInternal(const SingleKernelEnvelopeOpNode* op, bool includeAll) {
  bool print = false;  //(op->name == "l_unified");
  if (print) std::cout << "[IT1] Op " << op->name << " " << includeAll << std::endl;
  std::unordered_set<const Object*> explicit_set;
  for (const auto& di : op->explicit_dimensions) {
    explicit_set.insert(di->dim.get());
    if (print) std::cout << "[IT1]  exp dim " << di->dim << std::endl;
  }

  Array<Tensor> ret;
  for (const auto& t : op->inputs) {
    ret.push_back(t);
  }

  Array<PrimExpr> toCollectIn;
  for (auto dim2var_map : op->dim2var_maps) {
    for (auto it : dim2var_map) {
      if (includeAll || explicit_set.count(it.first)) {
        CHECK(!it.first->isFunDim());
        toCollectIn.push_back(UninterpFun::InlineUninterpFunCalls(it.second.iv->dom->min));
        toCollectIn.push_back(UninterpFun::InlineUninterpFunCalls(it.second.iv->dom->extent));
      }
    }
  }
  CollectTensors(ret, toCollectIn);
  return ret;
}

Array<Tensor> SingleKernelEnvelopeOpNode::InputTensors() const {
  return InputTensorsInternal(this, false);
}

Array<Tensor> SingleKernelEnvelopeOpNode::InputTensorsWithUnemitted() const {
  return InputTensorsInternal(this, true);
}

Operation SingleKernelEnvelopeOpNode::ReplaceInputs(
    const Operation& self, const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  bool replaced = false;
  Array<Tensor> new_inputs;
  for (size_t i = 0; i < this->inputs.size(); ++i) {
    if (rmap.count(this->inputs[i])) {
      new_inputs.push_back(rmap.at(this->inputs[i]));
      replaced = true;
      // std::cout << "Replaced " << this->inputs[i]->op << " " << rmap.at(this->inputs[i])->op <<
      // std::endl;
    } else {
      new_inputs.push_back(this->inputs[i]);
    }
  }

  std::vector<std::unordered_map<const DimensionNode*, DimVarEntry>> new_dim2var_maps;
  for (auto& dim2var_map : this->dim2var_maps) {
    auto it = dim2var_map.begin();
    std::unordered_map<const DimensionNode*, DimVarEntry> new_dim2var_map;
    for (; it != dim2var_map.end(); ++it) {
      CHECK(!it->first->isFunDim());
      new_dim2var_map[it->first] = {it->second.dim, it->second.iv};

      IterVar iv = it->second.iv;
      PrimExpr old_extent = iv->dom->extent;
      PrimExpr new_extent = te::ReplaceTensor(old_extent, rmap);
      if (!new_extent.same_as(old_extent)) {
        const_cast<RangeNode*>(iv->dom.as<RangeNode>())->extent = new_extent;
        replaced = true;
      }
    }
    new_dim2var_maps.push_back(new_dim2var_map);
  }

  if (replaced) {
    auto thisNode = self.as<SingleKernelEnvelopeOpNode>();
    auto n = make_object<SingleKernelEnvelopeOpNode>();
    n->name = thisNode->name;
    n->tag = thisNode->tag;
    n->attrs = thisNode->attrs;
    n->inputs = new_inputs;
    n->input_ops = GetInputOps(new_inputs);
    n->dim2var_maps = new_dim2var_maps;
    n->spatial_dimensions_ = thisNode->spatial_dimensions_;
    return Operation(n);
  } else {
    return self;
  }
}

void SingleKernelEnvelopeOpNode::PropBoundToInputs(
    const Operation& self, arith::Analyzer* analyzer,
    const std::unordered_map<const VarNode*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
#define COUT \
  if (print) std::cout << "[PBI] "
  CHECK_EQ(self.operator->(), this);
  for (int i = 0, sp_idx = 0; i < this->num_outputs(); ++i) {
    Tensor t = inputs[i];
    bool print = false;  //(t->op->name == "scan");
    if (print) COUT << "Op " << self << " " << t->op << std::endl;
    TensorDom* tdom = nullptr;
    if (out_dom_map->count(t)) {
      tdom = &out_dom_map->at(t);
    }
    // The update dimensions
    for (size_t k = 0; k < this->inputs[i]->shape.size(); ++k, ++sp_idx) {
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      IterVar sp_ax = this->dim2var_maps[i].at(sp_dim.as<DimensionNode>()).iv;

      PrimExpr inlined_arg;
      CHECK(sp_dim->type <= DimensionNode::kRangeDim);
      inlined_arg = sp_ax->var;

      IntSet arg_intset = EvalSet(inlined_arg, dom_map);
      COUT << "    Arg intset " << inlined_arg << " " << arg_intset << std::endl;

      const arith::IntervalSetNode* arg_interval = arg_intset.as<arith::IntervalSetNode>();
      if (arg_interval) {
        PrimExpr shape_i_min_value = make_zero(t->shape[k].dtype());
        PrimExpr shape_i_max_value = t->shape[k] - 1;
        PrimExpr min_value = arg_interval->min_value;
        PrimExpr max_value = arg_interval->max_value;
        // Prefer the shape bounds only when we can prove they are tighter.
        if (arith::is_neg_inf(min_value) || analyzer->CanProve(shape_i_min_value >= min_value)) {
          min_value = shape_i_min_value;
        }
        if (arith::is_pos_inf(max_value) || analyzer->CanProve(shape_i_max_value <= max_value)) {
          max_value = shape_i_max_value;
        }

        if (tdom) {
          tdom->data[k].push_back(IntSet::interval(min_value, max_value));
          COUT << "      Pushing " << IntSet::interval(min_value, max_value) << std::endl;
        }
      } else {
        if (tdom) {
          COUT << "      Pushing " << arg_intset << std::endl;
          tdom->data[k].push_back(arg_intset);
        }
      }
    }
  }
#undef COUT
}

void SingleKernelEnvelopeOpNode::GatherBound(
    const Operation& self, const std::unordered_map<Tensor, TensorDom>& tensor_dom,
    std::unordered_map<IterVar, Range>* out_dom_map,
    const Map<FunctionRef, CacheInfo> cacheTensorInfos) const {
  bool print = false;  //(self->name == "fused");
  CHECK_EQ(self.operator->(), this);
  std::vector<Tensor> output(this->num_outputs());
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = self.output(i);
  }

  // Update for spatial axis.
  size_t sp_idx = 0;
  if (print) std::cout << "[GBSc] Op " << self->name << std::endl;
  for (size_t i = 0; i < output.size(); ++i) {
    const TensorDom& d = tensor_dom.at(output[i]);
    Map<IterVar, IntSet> lv_sets_map;
    int sp_idx_start = sp_idx;
    for (size_t k = 0; k < this->inputs[i]->shape.size(); ++k, ++sp_idx) {
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      IterVar sp_ax = this->dim2var_maps[i].at(sp_dim.as<DimensionNode>()).iv;
      if (print)
        std::cout << "[GBSc]  Dim " << sp_dim->name << " " << sp_ax->var->name_hint << " "
                  << std::endl;

      IntSet iv_set = arith::Union(d.data[k]);
      if (print) std::cout << "[GBSc]  Dim0Set " << sp_dim->name << " " << iv_set << std::endl;
      CHECK(sp_dim->type <= DimensionNode::kRangeDim);

      // CHECK(/* Check if loop dim */)
      IterVar lv = this->GetIterVarFromDim(i, sp_dim);
      if (print)
        std::cout << "[GBSc]   Dim0.0 " << sp_dim->name << " " << lv << " " << iv_set << std::endl;
      if (lv_sets_map.count(lv)) {
        lv_sets_map.Set(lv, arith::Union({lv_sets_map.at(lv), iv_set}));
      } else {
        lv_sets_map.Set(lv, iv_set);
      }
    }

    for (auto it : lv_sets_map) {
      if (out_dom_map->find(it.first) == out_dom_map->end()) {
        if (print) {
          auto iset = it.second.as<arith::IntervalSetNode>();
          if (iset)
            std::cout << "[GBSc] " << it.first->var << " "
                      << UninterpFun::InlineUninterpFunCalls(it.second.min()) << " "
                      << UninterpFun::InlineUninterpFunCalls(it.second.max()) << " "
                      << UninterpFun::InlineUninterpFunCalls((it.first->dom)) << std::endl;
        }
        (*out_dom_map)[it.first] = it.second.cover_range(it.first->dom);
      }
    }

    for (size_t j = sp_idx_start; j < this->inputs[i]->shape.size(); ++j) {
      auto sp_dim = this->spatial_dimensions_[j];
      IterVar sp_ax = this->dim2var_maps[i].at(sp_dim.as<DimensionNode>()).iv;
      if (out_dom_map->find(sp_ax) == out_dom_map->end()) {
        (*out_dom_map)[sp_ax] = sp_ax->dom;
      }
    }
    for (auto it : dim2var_maps[i]) {
      if (it.first->type <= DimensionNode::kRangeDim) {
        if ((!out_dom_map->count(it.second.iv))) {
          (*out_dom_map)[it.second.iv] = it.second.iv->dom;
        }
      }
    }
    for (auto it : dim2var_maps[i]) {
      if (it.first->type <= DimensionNode::kRangeDim) {
        if (print) {
          // std::cout << "[GBSc]   DimF " << it.first->name << " " << it.second.iv->var->name_hint
          //           << " " << UninterpFun::InlineUninterpFunCalls((*out_dom_map)[it.second.iv])
          //           << std::endl;

          std::cout << "[GBSc]   DimF " << it.first->name << " " << it.second.iv->var->name_hint
                    << " " << (*out_dom_map).count(it.second.iv) << std::endl;
        }
      }
    }
  }
}  // namespace te

Stmt SingleKernelEnvelopeOpNode::BuildRealize(const Stage& stage,
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
      // Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      Dimension sp_dim = stage->dim_relation_graph->leaf_dimensions[k];
      // std::cout << "[BRSi] SpIdx " << this->inputs[i] << " " << k << " " << sp_dim << std::endl;
      IterVar sp_ax = this->GetDimVarEntry(i, sp_dim).iv;
      bounds.push_back(dom_map.count(sp_ax) ? dom_map.at(sp_ax) : sp_ax->dom);
    }
    ret = tir::RealizeNode::make(t->op, t->value_index, t->dtype, bounds, const_true(), ret);
  }
  return ret;
}

Stmt SingleKernelEnvelopeOpNode::BuildProvide(
    const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
    const std::unordered_map<std::string, Range>& env_dom_map,
    const std::unordered_map<std::string, IterVar>& env_var_map,
    const std::unordered_map<const VarNode*, std::string>& bind_map,
    bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  Stmt provide =
      AttrStmtNode::make(stage->op, attr::single_kernel_input_scope, 0, EvaluateNode::make(0));

  std::unordered_map<IterVar, PrimExpr> vmap;
  std::unordered_set<IterVar> empty;
  auto nest = MakeLoopNest(stage, dom_map, 0, false, empty, &vmap, debug_keep_trivial_loop);
  nest.push_back(MakeIfNest(
      MakeBoundCheck(stage, dom_map, env_dom_map, env_var_map, bind_map, vmap, false, empty)));
  // return MergeNest(nest, provide);
  return Substitute(MergeNest(nest, provide), vmap);
  // return EvaluateNode::make(0);
}
}  // namespace te
}  // namespace tvm
