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
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/uf_equality.h>

#include "../../arith/interval_set.h"
#include "../../tir/ir/var_replacer.h"
#include "../schedule/graph.h"
#include "op_util.h"

namespace tvm {
namespace te {
using namespace tir;

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ConditionalOpNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ConditionalOpNode*>(node.get());
      p->stream << "conditional(" << op->name << ", " << op << ")";
    });
TVM_REGISTER_NODE_TYPE(ConditionalOpNode);

inline bool prove_equal(PrimExpr lhs, PrimExpr rhs) { return is_zero(tir::Simplify(lhs - rhs)); }

int ConditionalOpNode::num_outputs() const { return static_cast<int>(then_case.size()); }
Array<IterVar> ConditionalOpNode::root_iter_vars() const {
  Array<IterVar> ret;

  for (size_t i = 0; i < explicit_dims.size(); ++i) {
    if (explicit_dims[i]->isLoopDim()) ret.push_back(explicit_loop_ivs[i]);
  }

  for (const auto& dim2var_map : dim2var_maps) {
    for (const auto& it : dim2var_map) {
      if (it.first->isLoopDim() && !ret.Contains(it.second.iv)) {
        ret.push_back(it.second.iv);
      }
    }
  }
  return ret;
}

DataType ConditionalOpNode::output_dtype(size_t i) const { return then_case[i]->dtype; }

Array<PrimExpr> ConditionalOpNode::output_shape(size_t i) const {
  CHECK_LT(i, then_case.size());
  return then_case[i]->shape;
}

Dimension ConditionalOpNode::GetBaseIndexDimension(size_t val_idx, size_t dim_idx) const {
  CHECK_LT(val_idx, num_outputs());
  for (size_t i = 0, sp_idx = 0; i < this->then_case.size(); ++i) {
    if (i < val_idx)
      sp_idx += then_case[i].ndim();
    else if (i == val_idx) {
      return this->spatial_dimensions_[sp_idx + dim_idx];
    }
  }
  return {};
}

Array<Dimension> ConditionalOpNode::GetRootIndexDimensions(size_t val_idx) const {
  Tensor t = then_case[val_idx];
  auto op = t->op.as<BaseVarDimOpNode>();
  CHECK(op);
  return op->GetRootIndexDimensions(t->value_index);
}

Operation ConditionalOpNode::make(std::string name, std::string tag,
                                  Map<std::string, ObjectRef> attrs, UninterpFun condition_uf,
                                  Array<Tensor> from_then, Array<Tensor> then_case,
                                  Array<Tensor> from_else, Array<Tensor> else_case,
                                  Array<Dimension> explicit_dims,
                                  Array<UninterpFun> explicit_min_ufs,
                                  Array<UninterpFun> explicit_max_ufs) {
  if (!attrs.defined()) {
    attrs = Map<std::string, ObjectRef>();
  }
  auto n = make_object<ConditionalOpNode>();
  CHECK_EQ(then_case.size(), else_case.size());

  for (size_t i = 0; i < then_case.size(); ++i) {
    CHECK_EQ(then_case[i]->dtype, else_case[i]->dtype);
    CHECK_EQ(else_case[i].ndim(), then_case[i].ndim())
        << "The then.ndim need to be equal to else.ndim";
    for (size_t k = 0; k < then_case[i].ndim(); ++k) {
      CHECK(prove_equal(else_case[i]->shape[k], then_case[i]->shape[k]));
    }
  }

  n->dim2var_maps =
      std::vector<std::unordered_map<const DimensionNode*, DimVarEntry>>(then_case.size());

  // Handle explicit dimensions
  Array<PrimExpr> args;
  Array<Dimension> arg_dims;
  for (size_t i = 0; i < explicit_dims.size(); ++i) {
    auto dim = explicit_dims[i];
    std::ostringstream os;
    os << "exp" << i;
    IterVar iv;
    if (dim->isLoopDim()) {
      PrimExpr min = UninterpFun::MakeCallTo(explicit_min_ufs[i], Array<PrimExpr>(args),
                                             Array<Dimension>(arg_dims));
      PrimExpr max = UninterpFun::MakeCallTo(explicit_max_ufs[i], Array<PrimExpr>(args),
                                             Array<Dimension>(arg_dims));
      iv = IterVarNode::make(Range(min, max), Var(os.str(), DataType::Int(32)), kDataPar);
      for (size_t j = 0; j < then_case.size(); ++j) {
        n->dim2var_maps[j][dim.as<DimensionNode>()] = {dim, iv, NullValue<UninterpFun>()};
        // std::cout << "[DIMAMP] " << dim << " " << std::endl;
      }
    } else {
      iv =
          IterVarNode::make(explicit_max_ufs[i]->range, Var(os.str(), DataType::Int(32)), kDataPar);
      for (size_t j = 0; j < then_case.size(); ++j) {
        n->dim2var_maps[j][dim.as<DimensionNode>()] = {dim, iv, explicit_max_ufs[i]};
        // std::cout << "[DIMAMP] " << dim << " " << std::endl;
      }
    }
    // std::cout << "[SCAN] Exp " << dim << " " << iv << std::endl;
    n->explicit_loop_ivs.push_back(iv);
    n->explicit_dims.push_back(dim);
    args.push_back(iv->var);
    arg_dims.push_back(dim);
  }

  n->condition = UninterpFun::MakeCallTo(condition_uf, args, arg_dims);

  // In the following code, we collect, for each input (update, for
  // now) tensor, the dimensions corresponding to its operation, and
  // create IterVars for them. Even if two input tensor ops share a
  // dimension, we create separate IterVars for them (unless this
  // dimension is an explicit dimension, which also includes the scan
  // dimension; these are handled above), as dimensions do not
  // correspond to any specific ranges, and the two operations sharing
  // the dimension may nevertheles have different bounds on the
  // corresponding IterVars, which we want to faithfully replicate
  // here to avoid any loss of precision during bounds inference.
  for (size_t i = 0; i < then_case.size(); ++i) {
    Tensor t = then_case[i];
    auto then_case_op = t->op.as<ComputeOpNode>();
    CHECK(then_case_op) << "Only ComputeOp allowed to be the then_case for a scan";

    std::unordered_map<const VarNode*, PrimExpr> vsub;
    for (size_t k = 0; k < then_case_op->all_dimensions.size(); ++k) {
      auto di = then_case_op->all_dimensions[k];
      auto dim = di->dim;
      auto entry = then_case_op->GetDimVarEntry(0, dim);
      if (!n->dim2var_maps[i].count(dim.as<DimensionNode>())) {
        VarReplacer replacer(vsub);
        IterVar iv =
            IterVarNode::make(Range::make_by_min_extent(replacer(entry.iv->dom->min),
                                                        replacer(entry.iv->dom->extent)),
                              entry.iv->var.copy_with_suffix(".sc"),
                              dim->type == DimensionNode::kScanDim ? kOrdered : kLoopNestOpaque,
                              entry.iv->thread_tag);
        n->dim2var_maps[i][dim.as<DimensionNode>()] = {entry.dim, iv, entry.value_expr};
      }
      vsub[entry.iv->var.as<VarNode>()] = n->dim2var_maps[i][dim.as<DimensionNode>()].iv->var;
    }

    for (size_t k = 0; k < then_case_op->root_index_dimensions.size(); ++k) {
      auto dim = then_case_op->root_index_dimensions[k];
      // std::cout << "[C_OP] Dim " << dim << std::endl;
      n->spatial_dimensions_.push_back(dim);
      n->spatial_axis_.push_back(n->dim2var_maps[i].at(dim.as<DimensionNode>()).iv);
    }
  }

  int sp_idx = 0;
  for (size_t i = 0; i < else_case.size(); ++i) {
    Tensor t = else_case[i];
    auto else_case_op = t->op.as<ComputeOpNode>();
    CHECK(else_case_op) << "Only ComputeOp allowed to be the else_case for a scan";

    std::unordered_map<const VarNode*, PrimExpr> vsub;
    // for (size_t k = 0; k < else_case_op->all_dimensions.size(); ++k) {
    //   auto di = else_case_op->all_dimensions[k];
    //   CHECK(n->dim2var_maps[i].count(di->dim.as<DimensionNode>())) << di->dim;
    // }
    for (size_t k = 0; k < else_case_op->all_dimensions.size(); ++k) {
      auto di = else_case_op->all_dimensions[k];
      auto dim = di->dim;
      auto entry = else_case_op->GetDimVarEntry(0, dim);
      if (!n->dim2var_maps[i].count(dim.as<DimensionNode>())) {
        VarReplacer replacer(vsub);
        IterVar iv =
            IterVarNode::make(Range::make_by_min_extent(replacer(entry.iv->dom->min),
                                                        replacer(entry.iv->dom->extent)),
                              entry.iv->var.copy_with_suffix(".sc"),
                              dim->type == DimensionNode::kScanDim ? kOrdered : kLoopNestOpaque,
                              entry.iv->thread_tag);
        n->dim2var_maps[i][dim.as<DimensionNode>()] = {entry.dim, iv, entry.value_expr};
      }
      vsub[entry.iv->var.as<VarNode>()] = n->dim2var_maps[i][dim.as<DimensionNode>()].iv->var;
    }
    for (size_t k = 0; k < else_case_op->root_index_dimensions.size(); ++k, sp_idx++) {
      auto dim = else_case_op->root_index_dimensions[k];
      CHECK_EQ(dim, n->spatial_dimensions_[sp_idx]);
    }
  }

  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->from_then = std::move(from_then);
  n->from_else = std::move(from_else);
  n->then_case = std::move(then_case);
  n->else_case = std::move(else_case);
  n->explicit_dims = std::move(explicit_dims);
  auto ret = Operation(n);
  // std::cout << "[CREATED] " << ret << std::endl;
  return ret;
}

TVM_REGISTER_GLOBAL("te.ConditionalOp")
    .set_body_typed([](std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                       UninterpFun condition_uf, Array<Tensor> from_then, Array<Tensor> then_case,
                       Array<Tensor> from_else, Array<Tensor> else_case,
                       Array<Dimension> explicit_loops, Array<UninterpFun> explicit_min_ufs,
                       Array<UninterpFun> explicit_max_ufs) {
      return ConditionalOpNode::make(name, tag, attrs, condition_uf, from_then, then_case,
                                     from_else, else_case, explicit_loops, explicit_min_ufs,
                                     explicit_max_ufs);
    });

Array<Tensor> ConditionalOpNode::InputTensors(bool includeAll) const {
  Array<Tensor> ret;
  for (Tensor t : then_case) {
    ret.push_back(t);
  }
  for (Tensor t : else_case) {
    ret.push_back(t);
  }

  Array<PrimExpr> toCollectIn;
  toCollectIn.push_back(condition);
  for (auto dim2var_map : dim2var_maps) {
    for (auto it : dim2var_map) {
      if (it.first->isFunDim()) {
        UninterpFun ufun = it.second.value_expr;
        if (includeAll || it.second.iv->iter_type != kLoopNestOpaque) {
          toCollectIn.push_back(UninterpFun::InlineUninterpFunCalls(ufun->body));
        }
      } else {
        toCollectIn.push_back(UninterpFun::InlineUninterpFunCalls(it.second.iv->dom->min));
        toCollectIn.push_back(UninterpFun::InlineUninterpFunCalls(it.second.iv->dom->extent));
      }
    }
  }
  CollectTensors(ret, toCollectIn);
  // for (auto it : ret) std::cout << "[INPUIT] " << it << std::endl;
  return ret;
}

Array<Tensor> ConditionalOpNode::InputTensors() const { return this->InputTensors(false); }

Array<Tensor> ConditionalOpNode::InputTensorsWithUnemitted() const {
  return this->InputTensors(true);
}

Operation ConditionalOpNode::ReplaceInputs(const Operation& self,
                                           const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  auto n = make_object<ConditionalOpNode>(*this);
  for (size_t i = 0; i < n->then_case.size(); ++i) {
    if (rmap.count(n->then_case[i])) {
      // std::cout << "[C_OP] Replacing " << n->then_case[i]->op << " " <<
      // rmap.at(n->then_case[i])->op << std::endl;
      n->then_case.Set(i, rmap.at(n->then_case[i]));
    }
    if (rmap.count(n->else_case[i])) {
      // std::cout << "[C_OP] Replacing " << n->else_case[i]->op << " " <<
      // rmap.at(n->else_case[i])->op << std::endl;
      n->else_case.Set(i, rmap.at(n->else_case[i]));
    }
  }

  bool changed = false;
  if (!n->then_case.same_as(then_case) || !n->else_case.same_as(else_case)) {
    changed = true;
  }

  PrimExpr new_condition = te::ReplaceTensor(condition, rmap);
  if (!new_condition.same_as(condition)) {
    changed = true;
    n->condition = new_condition;
  }

  std::vector<std::unordered_map<const DimensionNode*, DimVarEntry>> new_dim2var_maps;
  for (auto& dim2var_map : n->dim2var_maps) {
    auto it = dim2var_map.begin();
    std::unordered_map<const DimensionNode*, DimVarEntry> new_dim2var_map;
    for (; it != dim2var_map.end(); ++it) {
      if (it->first->isFunDim()) {
        UninterpFun old_fun = it->second.value_expr;

        PrimExpr old_fun_body = old_fun->body;
        PrimExpr new_fun_body = te::ReplaceTensor(old_fun_body, rmap);
        if (!new_fun_body.same_as(old_fun_body)) {
          changed = true;
          // std::cout << "Replaced " << new_fun_body << " " << old_fun_body << std::endl;
          new_dim2var_map[it->first] = {
              it->second.dim, it->second.iv,
              UninterpFunNode::make(old_fun->fname, old_fun->range, old_fun->dimensions,
                                    old_fun->parameters, new_fun_body)};
        } else {
          new_dim2var_map[it->first] = {it->second.dim, it->second.iv, it->second.value_expr};
        }
      } else {
        new_dim2var_map[it->first] = {it->second.dim, it->second.iv, it->second.value_expr};
      }

      IterVar iv = it->second.iv;
      PrimExpr old_extent = iv->dom->extent;
      PrimExpr new_extent = te::ReplaceTensor(old_extent, rmap);
      if (!new_extent.same_as(old_extent)) {
        const_cast<RangeNode*>(iv->dom.as<RangeNode>())->extent = new_extent;
        changed = true;
      }
    }
    new_dim2var_maps.push_back(new_dim2var_map);
  }

  n->dim2var_maps = new_dim2var_maps;

  if (changed) {
    return Operation(n);
  } else {
    return self;
  }
}

void ConditionalOpNode::PropBoundToInputs(
    const Operation& self, arith::Analyzer* analyzer,
    const std::unordered_map<const VarNode*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
#define COUT \
  if (print) std::cout << "[PBIcond] "

  CHECK_EQ(self.operator->(), this);
  for (int i = 0, sp_idx = 0; i < this->num_outputs(); ++i) {
    TensorDom* then_case_dom = nullptr;
    TensorDom* else_case_dom = nullptr;
    if (out_dom_map->count(this->then_case[i])) {
      then_case_dom = &out_dom_map->at(this->then_case[i]);
    }
    if (out_dom_map->count(this->else_case[i])) {
      else_case_dom = &out_dom_map->at(this->else_case[i]);
    }

    for (size_t k = 0; k < this->then_case[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = this->spatial_axis_[sp_idx];
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];
      auto fun = [&](TensorDom* dom, Tensor t) {
        bool print = false;
        // bool print = (t->op->name == "next_v");
        if (print)
          COUT << "Op " << self << " " << t->op << " " << GetRef<Operation>(this) << " "
               << this->dim2var_maps.size() << std::endl;
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

        IntSet arg_intset;
        arg_intset = EvalSet(inlined_arg, dom_map);

        COUT << "    Arg intset " << inlined_arg << " " << arg_intset << std::endl;
        ////////////////////////////// PPF: DEBUG
        // arg_intset = TranslateIterVarsFromConsumerToProducer(arg_intset, self, t);
        ////////////////////////////// PPF: DEBUG
        COUT << "       translated " << arg_intset << std::endl;

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
          dom->data[k].push_back(IntSet::interval(min_value, max_value));
          COUT << "      Pushing " << IntSet::interval(min_value, max_value) << std::endl;
        } else {
          dom->data[k].push_back(arg_intset);
          COUT << "      Pushing " << arg_intset << std::endl;
        }
      };

      if (then_case_dom) {
        fun(then_case_dom, then_case[i]);
      }

      if (else_case_dom) {
        fun(else_case_dom, else_case[i]);
      }
    }
  }
#undef COUT
}

void ConditionalOpNode::GatherBound(const Operation& self,
                                    const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                                    std::unordered_map<IterVar, Range>* out_dom_map,
                                    const Map<FunctionRef, CacheInfo> cacheTensorInfos) const {
  bool print = false;  //(self->name == "c_next_h");
  CHECK_EQ(self.operator->(), this);
  std::vector<Tensor> output(this->num_outputs());
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = self.output(i);
  }

  // Update for spatial axis.
  size_t sp_idx = 0;
  if (print) std::cout << "[GBS] Op " << self->name << std::endl;
  for (int i = 0; i < this->num_outputs(); ++i) {
    const TensorDom& d = tensor_dom.at(output[i]);
    Map<IterVar, IntSet> lv_sets_map;
    for (size_t k = 0; k < this->then_case[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = this->spatial_axis_[sp_idx];
      Dimension sp_dim = this->spatial_dimensions_[sp_idx];

      if (print)
        std::cout << "[GBS]  Dim " << sp_dim->name << " " << sp_ax->var->name_hint << std::endl;
      // CHECK(fix_pt.count(sp_ax));
      // if (fix_pt[sp_ax].as<tir::IntImmNode>()->value) {
      // fix point, we can slice it.

      IntSet iv_set = arith::Union(d.data[k]);
      if (print) std::cout << "[GBS]  Dim0Set " << sp_dim->name << " " << iv_set << std::endl;
      if (sp_dim->type <= DimensionNode::kRangeDim) {
        // CHECK(/* Check if loop dim */)
        IterVar lv = this->GetIterVarFromDim(i, sp_dim);
        if (print)
          std::cout << "[GBS]   Dim0.0 " << sp_dim->name << " " << lv << " " << iv_set << std::endl;
        if (lv_sets_map.count(lv)) {
          lv_sets_map.Set(lv, arith::Union({lv_sets_map.at(lv), iv_set}));
        } else {
          lv_sets_map.Set(lv, iv_set);
        }
      } else {
        Map<Dimension, IntSet> lv_sets =
            tir::ProjectInverse(iv_set, dim2var_maps[i].at(sp_dim.operator->()).value_expr);
        if (print)
          std::cout << "[GBS]  Dim0.1S " << sp_dim->name << " " << lv_sets << " "
                    << dim2var_maps[i].at(sp_dim.operator->()).value_expr->body << std::endl;
        if (lv_sets.defined()) {
          for (auto pair : lv_sets) {
            Dimension dim = pair.first;
            IntSet lv_set = pair.second;
            IterVar lv = this->GetIterVarFromDim(i, dim);
            if (print)
              std::cout << "[GBS]   Dim0.1 " << sp_dim->name << " " << dim->name << " " << lv << " "
                        << iv_set << std::endl;
            if (lv_sets_map.count(lv)) {
              lv_sets_map.Set(lv, arith::Union({lv_sets_map.at(lv), lv_set}));
            } else {
              lv_sets_map.Set(lv, lv_set);
            }
          }
        }
      }
      // } else {
      //   if (print) std::cout << "[GBS] Dim0 " << sp_dim->name << " No fixed point" << std::endl;
      //   // not a fix point, need to include everything.
      //   if (sp_dim->isLoopDim()) {
      //     lv_sets_map.Set(sp_ax, IntSet::range(sp_ax->dom));
      //   } else {
      //     for (auto arg_dim : dim2var_maps[i].at(sp_dim.operator->()).value_expr->dimensions) {
      //       IterVar loop_iv = this->GetIterVarFromDim(i, arg_dim);
      //       IntSet set = IntSet::range(loop_iv->dom);
      //       if (lv_sets_map.count(loop_iv)) {
      //         lv_sets_map.Set(loop_iv, arith::Union({lv_sets_map.at(loop_iv), set}));
      //       } else {
      //         lv_sets_map.Set(loop_iv, set);
      //       }
      //     }
      //   }
      // }
    }

    for (auto it : lv_sets_map) {
      if (out_dom_map->find(it.first) == out_dom_map->end()) {
        (*out_dom_map)[it.first] = it.second.cover_range(it.first->dom);
        if (print)
          std::cout << "[GBS]  Dim1 " << it.first << " "
                    << UninterpFun::InlineUninterpFunCalls((*out_dom_map)[it.first]) << std::endl;
      }
    }

    for (size_t i = 0; i < this->spatial_axis_.size(); ++i) {
      if (out_dom_map->find(this->spatial_axis_[i]) == out_dom_map->end()) {
        (*out_dom_map)[this->spatial_axis_[i]] = this->spatial_axis_[i]->dom;
        if (print)
          std::cout << "[GBS]  Dim2 " << this->spatial_axis_[i] << " "
                    << UninterpFun::InlineUninterpFunCalls((*out_dom_map)[this->spatial_axis_[i]])
                    << std::endl;
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

Stmt ConditionalOpNode::BuildRealize(const Stage& stage,
                                     const std::unordered_map<IterVar, Range>& dom_map,
                                     const Stmt& body) const {
  Stmt ret = body;
  size_t sp_idx = 0;
  // std::cout << "[BR] Build realize for " << stage->op << " " << std::endl;
  for (int i = 0; i < num_outputs(); ++i) {
    Tensor t = stage->op.output(i);
    Region bounds;
    for (size_t k = 0; k < this->then_case[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = spatial_axis_[sp_idx];

      Range r;
      r = dom_map.count(sp_ax) ? dom_map.at(sp_ax) : sp_ax->dom;
      // N.B.: Here, in order to ensure that we don't allocate a
      // buffer with a variable size, we relax the extent of the
      // realize range to no include any calls to complex uninterp
      // functions. This is more of a hack as the bounds of the
      // realize node migfht be used of purposes other than just
      // deciding the size of the buffer to allocate. But by the time
      // we create the AllocateNode in storage_flatten.cc, we have
      // inlined all calls to uninterp functions and can no longer
      // effectively relax them. Ideally, we should hold off on
      // inlining uninterp function calls to as late a stage as
      // possible.
      Range relaxed =
          Range::make_by_min_extent(r->min, UninterpFun::RelaxComplexUninterpCalls(r->extent));
      bounds.push_back(relaxed);
    }
    ret = tir::RealizeNode::make(t->op, t->value_index, t->dtype, bounds, const_true(), ret);
  }
  return ret;
}

Stmt ConditionalOpNode::BuildProvide(
    const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
    const std::unordered_map<std::string, Range>& env_dom_map,
    const std::unordered_map<std::string, IterVar>& env_var_map,
    const std::unordered_map<const VarNode*, std::string>& bind_map,
    bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);

  Stmt provide = IfThenElseNode::make(
      condition,
      AttrStmtNode::make(stage->op, attr::conditional_then_scope, 0, EvaluateNode::make(0)),
      AttrStmtNode::make(stage->op, attr::conditional_else_scope, 0, EvaluateNode::make(0)));

  std::unordered_map<IterVar, PrimExpr> vmap;
  std::unordered_set<IterVar> empty;
  auto nest = MakeScanOpLoopNest(stage, dom_map, 0, false, empty, &vmap, debug_keep_trivial_loop,
                                 explicit_dims);
  auto if_nest = MakeIfNest(
      MakeBoundCheck(stage, dom_map, env_dom_map, env_var_map, bind_map, vmap, false, empty));
  auto loops_and_preds = MergeWhileHoisting(stage, nest, if_nest);
  Stmt ret = MergeNest(loops_and_preds, provide);
  ret = Substitute(ret, vmap);
  return ret;
}
}  // namespace te
}  // namespace tvm
