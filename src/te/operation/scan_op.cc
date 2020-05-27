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
#include <tvm/tir/stmt_functor.h>

#include "../../arith/interval_set.h"
#include "../../tir/ir/var_replacer.h"
#include "../schedule/graph.h"
#include "op_util.h"

namespace tvm {
namespace te {
using namespace tir;

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ScanOpNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ScanOpNode*>(node.get());
      p->stream << "scan(" << op->name << ", " << op << ")";
    });
TVM_REGISTER_NODE_TYPE(ScanOpNode);

inline bool prove_equal(PrimExpr lhs, PrimExpr rhs) { return is_zero(tir::Simplify(lhs - rhs)); }

int ScanOpNode::num_outputs() const { return static_cast<int>(update.size()); }
Array<IterVar> ScanOpNode::root_iter_vars() const {
  Array<IterVar> ret;
  for (const auto& iv : explicit_loop_ivs) {
    ret.push_back(iv);
  }
  ret.push_back(scan_axis);

  for (const auto& dim2var_map : dim2var_maps) {
    for (const auto& it : dim2var_map) {
      if (it.first->type <= DimensionNode::kRangeDim && !ret.Contains(it.second.iv)) {
        ret.push_back(it.second.iv);
      }
    }
  }
  return ret;
}

DataType ScanOpNode::output_dtype(size_t i) const { return update[i]->dtype; }

Array<PrimExpr> ScanOpNode::output_shape(size_t i) const {
  CHECK_LT(i, state_placeholder.size());
  return state_placeholder[i]->shape;
}

Dimension ScanOpNode::GetBaseIndexDimension(size_t val_idx, size_t dim_idx) const {
  CHECK_LT(val_idx, num_outputs());
  for (size_t i = 0, sp_idx = 0; i < this->init.size(); ++i) {
    if (i < val_idx)
      sp_idx += update[i].ndim();
    else if (i == val_idx) {
      return this->spatial_dimensions_[sp_idx + dim_idx];
    }
  }
  return {};
}

Operation ScanOpNode::make(std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                           UninterpFun range_min_uf, UninterpFun range_max_uf, Dimension scan_dim,
                           Array<Tensor> init, Array<Tensor> update,
                           Array<Tensor> state_placeholder, Array<Tensor> inputs,
                           Array<Dimension> explicit_dims, Array<UninterpFun> explicit_extent_ufs) {
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
    for (size_t k = 0; k < update[i].ndim(); ++k) {
      CHECK(prove_equal(update[i]->shape[k], state_placeholder[i]->shape[k]));
    }

    for (size_t k = 1; k < init[i].ndim(); ++k) {
      CHECK(prove_equal(init[i]->shape[k], state_placeholder[i]->shape[k]));
    }
  }

  n->dim2var_maps =
      std::vector<std::unordered_map<const DimensionNode*, DimVarEntry>>(update.size());

  // Handle explicit and scan dimensions
  Array<PrimExpr> args;
  Array<Dimension> arg_dims;
  for (size_t i = 0; i < explicit_dims.size(); ++i) {
    auto dim = explicit_dims[i];
    std::ostringstream os;
    os << "exp" << i;
    PrimExpr min = UninterpFun::MakeCallTo(explicit_extent_ufs[i], Array<PrimExpr>(args),
                                           Array<Dimension>(arg_dims));
    PrimExpr extent = UninterpFun::MakeCallTo(explicit_extent_ufs[i], Array<PrimExpr>(args),
                                              Array<Dimension>(arg_dims));
    auto iv = IterVarNode::make(Range::make_by_min_extent(0, extent),
                                Var(os.str(), DataType::Int(32)), kDataPar);
    n->explicit_loop_ivs.push_back(iv);
    args.push_back(iv->var);
    arg_dims.push_back(dim);
    for (size_t j = 0; j < update.size(); ++j) {
      n->dim2var_maps[j][dim.as<DimensionNode>()] = {dim, iv, explicit_extent_ufs[i]};
    }
  }

  // Now the scan dimension
  PrimExpr range_min = UninterpFun::MakeCallTo(range_min_uf, args, arg_dims);
  PrimExpr range_max = UninterpFun::MakeCallTo(range_max_uf, args, arg_dims);
  IterVar axis = IterVarNode::make(Range(range_min, range_max), Var(name + ".idx"), kOrdered, "");

  // In the following code, we collect, for each input (update, for
  // now) tensor, the dimensions corresponding to it's operation, and
  // create IterVars for them. Even if two input tensor ops share a
  // dimension, we create separate IterVars for them (unless this
  // dimension is an explicit dimension, which also includes the scan
  // dimension; these are handled above), as dimensions do not
  // correspond to any specific ranges, and the two operations sharing
  // the dimension may nevertheles have different bounds on the
  // corresponding IterVars, which we want to faithfully replicate
  // here to avoid any loss of precision during bounds inference.
  for (size_t i = 0; i < update.size(); ++i) {
    Tensor t = update[i];
    auto update_op = t->op.as<ComputeOpNode>();
    CHECK(update_op) << "Only ComputeOp allowed to be the update for a scan";

    for (size_t k = 0; k < update_op->root_index_dimensions.size(); ++k) {
      auto dim = update_op->root_index_dimensions[k];
      if (!n->dim2var_maps[i].count(dim.as<DimensionNode>())) {
        IterVar iv = axis;
        if (dim != scan_dim) {
          // setup spatial axis
          std::ostringstream spatial_name;
          spatial_name << name << ".out" << i << ".i" << k;
          iv = IterVarNode::make(Range::make_by_min_extent(0, update[i]->shape[k]),
                                 Var(spatial_name.str()),
                                 dim->type == DimensionNode::kScanDim ? kOrdered : kLoopNestOpaque);
        }

        auto entry = update_op->GetDimVarEntry(0, dim);
        n->dim2var_maps[i][dim.as<DimensionNode>()] = {entry.dim, iv, entry.value_expr};
      }
    }

    std::unordered_map<const VarNode*, PrimExpr> vsub;
    for (auto dim : update_op->loop_dimensions) {
      auto entry = update_op->GetDimVarEntry(0, dim);
      if (!n->dim2var_maps[i].count(dim.as<DimensionNode>())) {
        IterVar iv = axis;
        if (dim != scan_dim) {
          VarReplacer replacer(vsub);

          iv = IterVarNode::make(Range::make_by_min_extent(replacer(entry.iv->dom->min),
                                                           replacer(entry.iv->dom->extent)),
                                 entry.iv->var.copy_with_suffix(".sc"),
                                 dim->type == DimensionNode::kScanDim ? kOrdered : kLoopNestOpaque,
                                 entry.iv->thread_tag);
        }
        n->dim2var_maps[i][dim.as<DimensionNode>()] = {entry.dim, iv, entry.value_expr};
      }
      vsub[entry.iv->var.as<VarNode>()] = n->dim2var_maps[i][dim.as<DimensionNode>()].iv->var;
    }

    for (size_t k = 0; k < update_op->root_index_dimensions.size(); ++k) {
      auto dim = update_op->root_index_dimensions[k];
      n->spatial_dimensions_.push_back(dim);
      n->spatial_axis_.push_back(n->dim2var_maps[i].at(dim.as<DimensionNode>()).iv);
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
  n->explicit_dims = std::move(explicit_dims);
  return Operation(n);
}

TVM_REGISTER_GLOBAL("te.ScanOp")
    .set_body_typed([](std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                       UninterpFun axis_range_min_uf, UninterpFun axis_range_max_uf,
                       Dimension scan_dim, Array<Tensor> init, Array<Tensor> update,
                       Array<Tensor> state_placeholder, Array<Tensor> inputs,
                       Array<Dimension> explicit_dims, Array<UninterpFun> explicit_extent_ufs) {
      return ScanOpNode::make(name, tag, attrs, axis_range_min_uf, axis_range_max_uf, scan_dim,
                              init, update, state_placeholder, inputs, explicit_dims,
                              explicit_extent_ufs);
    });

Array<Tensor> scan(Dimension scan_dim, Array<Tensor> init, Array<Tensor> update,
                   Array<Tensor> state_placeholder, Array<Dimension> explicit_dims,
                   Array<UninterpFun> explicit_extent_ufs, Array<Tensor> inputs, std::string name,
                   std::string tag, Map<std::string, ObjectRef> attrs) {
  PrimExpr max = update[0]->shape[0] - init[0]->shape[0];
  UninterpFun max_uf = UninterpFunNode::make("scan_extent", Range(max, max), {}, {}, max);
  UninterpFun min_uf = UninterpFunNode::make("scan_extent", Range(0, 0), {}, {}, 0);
  Operation op = ScanOpNode::make(name, tag, attrs, min_uf, max_uf, scan_dim, init, update,
                                  state_placeholder, inputs, explicit_dims, explicit_extent_ufs);

  Array<Tensor> res;
  for (int i = 0; i < op->num_outputs(); ++i) {
    res.push_back(op.output(i));
  }
  return res;
}

Array<Tensor> ScanOpNode::InputTensors(bool includeAll) const {
  Array<Tensor> ret;
  for (Tensor t : init) {
    ret.push_back(t);
  }
  for (Tensor t : update) {
    ret.push_back(t);
  }

  Array<PrimExpr> toCollectIn;
  for (auto dim2var_map : dim2var_maps) {
    for (auto it : dim2var_map) {
      if (it.first->type == DimensionNode::kFunDim) {
        UninterpFun ufun = it.second.value_expr;
        if (includeAll || it.second.iv->iter_type != kLoopNestOpaque) {
          toCollectIn.push_back(UninterpFun::InlineUninterpFunCalls(ufun->body));
        }
      }
    }
  }
  CollectTensors(ret, toCollectIn);
  return ret;
}

Array<Tensor> ScanOpNode::InputTensors() const { return this->InputTensors(false); }

Array<Tensor> ScanOpNode::InputTensorsWithUnemitted() const { return this->InputTensors(true); }

Operation ScanOpNode::ReplaceInputs(const Operation& self,
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
  if (!n->init.same_as(init) || !n->update.same_as(update)) {
    return Operation(n);
  } else {
    return self;
  }
}

void ScanOpNode::PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                                   const std::unordered_map<const VarNode*, IntSet>& dom_map,
                                   std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
#define COUT \
  if (print) std::cout << "[PBI] "

  CHECK_EQ(self.operator->(), this);
  for (int i = 0, sp_idx = 0; i < this->num_outputs(); ++i) {
    TensorDom* init_dom = nullptr;
    TensorDom* update_dom = nullptr;
    if (out_dom_map->count(this->init[i])) {
      init_dom = &out_dom_map->at(this->init[i]);
    }
    if (out_dom_map->count(this->update[i])) {
      update_dom = &out_dom_map->at(this->update[i]);
    }

    // The update dimensions
    for (size_t k = 0; k < this->update[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = this->spatial_axis_[sp_idx];
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      if (init_dom) {
        init_dom->data[k].push_back(dom_map.at(sp_ax->var.get()));
      }

      if (update_dom) {
        Tensor t = update[i];
        bool print = false;  //(t->op->name == "css_update");
        if (print) COUT << "Op " << self << " " << t->op << std::endl;
        PrimExpr inlined_arg;
        if (sp_dim->type <= DimensionNode::kRangeDim) {
          inlined_arg = sp_ax->var;
        } else {
          Array<Dimension> loop_dims;
          Array<PrimExpr> axis_vars;
          for (auto it : dim2var_maps[i]) {
            if (it.first->type <= DimensionNode::kRangeDim) {
              loop_dims.push_back(GetRef<Dimension>(it.first));
              axis_vars.push_back(it.second.iv->var);
            }
          }
          CHECK(dim2var_maps[i].count(sp_dim.as<DimensionNode>())) << sp_dim->name;
          auto ufun = dim2var_maps[i].at(sp_dim.as<DimensionNode>()).value_expr;
          inlined_arg = UninterpFun::MakeCallTo(ufun, axis_vars, loop_dims);
        }

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
          update_dom->data[k].push_back(IntSet::interval(min_value, max_value));
          COUT << "      Pushing " << IntSet::interval(min_value, max_value) << std::endl;
        } else {
          update_dom->data[k].push_back(arg_intset);
          COUT << "      Pushing " << arg_intset << std::endl;
        }
      }
    }
  }
}

void ScanOpNode::GatherBound(const Operation& self,
                             const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                             std::unordered_map<IterVar, Range>* out_dom_map) const {
  bool print = false;  // (self->name == "child_sum");
  CHECK_EQ(self.operator->(), this);
  CHECK(!out_dom_map->count(this->scan_axis));
  std::vector<Tensor> output(this->num_outputs());
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = self.output(i);
  }

  Map<IterVar, PrimExpr> fix_pt = ScanFixPointAnalysis(self);

  // Update for spatial axis.
  size_t sp_idx = 0;
  if (print) std::cout << "[GBS] Op " << self->name << std::endl;
  for (int i = 0; i < this->num_outputs(); ++i) {
    const TensorDom& d = tensor_dom.at(output[i]);
    Map<IterVar, IntSet> lv_sets_map;
    for (size_t k = 0; k < this->update[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = this->spatial_axis_[sp_idx];
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];

      // if (print) std::cout << "[GBS]  Dim " << sp_dim->name << " " << sp_ax->var->name_hint <<
      // " " << fix_pt[sp_ax] << std::endl;
      CHECK(fix_pt.count(sp_ax));
      if (fix_pt[sp_ax].as<tir::IntImmNode>()->value) {
        // fix point, we can slice it.

        IntSet iv_set = arith::Union(d.data[k]);
        // if (print) std::cout << "[GBS]  Dim0Set " << sp_dim->name << " " << iv_set <<
        // std::endl;
        if (sp_dim->type <= DimensionNode::kRangeDim) {
          // CHECK(/* Check if loop dim */)
          IterVar lv = this->GetIterVarFromDim(i, sp_dim);
          // if (print) std::cout << "[GBS]   Dim0.0 " << sp_dim->name << " " << lv << " " <<
          // iv_set
          // << std::endl;
          if (lv_sets_map.count(lv)) {
            lv_sets_map.Set(lv, arith::Union({lv_sets_map.at(lv), iv_set}));
          } else {
            lv_sets_map.Set(lv, iv_set);
          }
        } else {
          Map<Dimension, IntSet> lv_sets =
              arith::ProjectInverse(iv_set, dim2var_maps[i].at(sp_dim.operator->()).value_expr);
          // if (print) std::cout << "[GBS]  Dim0.1S " << sp_dim->name << " " << lv_sets <<
          // std::endl;
          if (lv_sets.defined()) {
            for (auto pair : lv_sets) {
              Dimension dim = pair.first;
              IntSet lv_set = pair.second;
              IterVar lv = this->GetIterVarFromDim(i, dim);
              // if (print) std::cout << "[GBS]   Dim0.1 " << sp_dim->name << " " << dim->name <<
              // " " << lv << " " << iv_set << std::endl;
              if (lv_sets_map.count(lv)) {
                lv_sets_map.Set(lv, arith::Union({lv_sets_map.at(lv), lv_set}));
              } else {
                lv_sets_map.Set(lv, lv_set);
              }
            }
          }
        }
      } else {
        // if (print) std::cout << "[GBS] Dim0 " << sp_dim->name << " No fixed point" <<
        // std::endl; not a fix point, need to include everything.
        if (sp_dim->type <= DimensionNode::kRangeDim) {
          lv_sets_map.Set(sp_ax, IntSet::range(sp_ax->dom));
        } else {
          for (auto arg_dim : dim2var_maps[i].at(sp_dim.operator->()).value_expr->dimensions) {
            IterVar loop_iv = this->GetIterVarFromDim(i, arg_dim);
            IntSet set = IntSet::range(loop_iv->dom);
            if (lv_sets_map.count(loop_iv)) {
              lv_sets_map.Set(loop_iv, arith::Union({lv_sets_map.at(loop_iv), set}));
            } else {
              lv_sets_map.Set(loop_iv, set);
            }
          }
        }
      }
    }

    for (auto it : lv_sets_map) {
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
        // UninterpFun::InlineUninterpFunCalls((*out_dom_map)[this->spatial_axis_[i]]) <<
        // std::endl;
      }
    }

    for (auto it : dim2var_maps[i]) {
      if (it.first->type <= DimensionNode::kRangeDim) {
        if (print)
          std::cout << "[GBS]   DimF " << it.first->name << " " << it.second.iv->var->name_hint
                    << " " << UninterpFun::InlineUninterpFunCalls((*out_dom_map)[it.second.iv])
                    << std::endl;
      }
    }
  }
}

Stmt ScanOpNode::BuildRealize(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                              const Stmt& body) const {
  Stmt ret = body;
  size_t sp_idx = 0;
  std::cout << "[BR] Build realize for " << stage->op << " " << std::endl;
  for (int i = 0; i < num_outputs(); ++i) {
    Tensor t = stage->op.output(i);
    Region bounds;
    for (size_t k = 0; k < this->update[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = spatial_axis_[sp_idx];
      std::cout << "[BR]     " << spatial_dimensions_[sp_idx] << " " << sp_ax << " "
                << dom_map.count(sp_ax) << " "
                << (dom_map.count(sp_ax) ? dom_map.at(sp_ax) : sp_ax->dom) << " " << std::endl;
      bounds.push_back(dom_map.count(sp_ax) ? dom_map.at(sp_ax) : sp_ax->dom);
    }
    ret = tir::RealizeNode::make(t->op, t->value_index, t->dtype, bounds, const_true(), ret);
  }
  return ret;

  // CHECK_EQ(stage->op.get(), this);
  // Range sdom = dom_map.at(this->scan_axis);
  // Range tdom = Range::make_by_min_extent(0, tir::Simplify(sdom->extent + sdom->min));
  // Stmt ret = body;
  // size_t sp_idx = 0;
  // std::cout << "[BR] Build realize for " << stage->op << " " << std::endl;
  // for (int i = 0; i < num_outputs(); ++i) {
  //   Tensor t = stage->op.output(i);
  //   CHECK_EQ(static_cast<size_t>(t->value_index), i);
  //   Region bounds;
  //   bounds.push_back(tdom);
  //   std::cout << "[BR]   t " << tdom << " " << std::endl;
  //   for (size_t k = 1; k < this->update[i]->shape.size(); ++k, ++sp_idx) {
  //     IterVar sp_ax = this->spatial_axis_[sp_idx];
  //     std::cout << "[BR]     " << sp_ax << " " << dom_map.count(sp_ax) << " "
  //               << (dom_map.count(sp_ax) ? dom_map.at(sp_ax) : sp_ax->dom) << " " << std::endl;
  //     bounds.push_back(dom_map.count(sp_ax) ? dom_map.at(sp_ax) : sp_ax->dom);
  //   }
  //   ret = tir::RealizeNode::make(t->op, t->value_index, t->dtype, bounds, const_true(), ret);
  // }
  // return ret;
}

Stmt ScanOpNode::BuildProvide(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                              bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  Stmt provide = AttrStmtNode::make(stage->op, attr::scan_update_scope, this->scan_axis->var,
                                    EvaluateNode::make(0));
  Stmt init = AttrStmtNode::make(stage->op, attr::scan_init_scope, 0, EvaluateNode::make(0));
  size_t begin_scan = 0;
  for (size_t i = 0; i < stage->leaf_iter_vars.size(); ++i) {
    if (stage->leaf_iter_vars[i]->iter_type == kThreadIndex) {
      CHECK_EQ(begin_scan, i);
      begin_scan = i + 1;
    }
  }
  std::unordered_map<IterVar, PrimExpr> vmap;
  std::unordered_set<IterVar> empty;
  auto nest = MakeLoopNest(stage, dom_map, 0, false, empty, &vmap, debug_keep_trivial_loop);
  nest[begin_scan].push_back(init);
  nest.push_back(MakeIfNest(MakeBoundCheck(stage, dom_map, vmap, false, empty)));
  Stmt ret = MergeNest(nest, provide);
  ret = Substitute(ret, vmap);
  return ret;
}
}  // namespace te
}  // namespace tvm
