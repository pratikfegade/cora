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
 * \brief Compute Op.
 * \file compute_op.cc
 */
#include "compute_op.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/modes.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/uf_equality.h>

#include <string>
#include <unordered_set>
#include <utility>

#include "../../arith/compute_expr.h"
#include "../../arith/interval_set.h"
#include "../../tir/ir/var_replacer.h"
#include "../schedule/message_passing.h"
#include "op_util.h"

namespace tvm {
namespace te {
using namespace tir;

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ComputeOpNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ComputeOpNode*>(node.get());
      p->stream << "compute(" << op->name << ", " << op << ")";
    });

TVM_REGISTER_NODE_TYPE(ComputeOpNode);

/// Verify if ComputeOp is valid with respect to Reduce operations.
static void VerifyComputeOp(const ComputeOpNode* op);

inline bool ReduceEqual(const tir::ReduceNode* a, const tir::ReduceNode* b) {
  return (a->combiner.same_as(b->combiner)) && (a->source.same_as(b->source)) &&
         (a->axis.same_as(b->axis)) && (a->condition.same_as(b->condition));
}

int ComputeOpNode::num_outputs() const { return body.size(); }

Array<IterVar> BaseComputeOpNode::root_iter_vars() const {
  if (reduce_axis.size() == 0) return axis;
  Array<IterVar> ret = axis;
  for (IterVar iv : reduce_axis) {
    ret.push_back(iv);
  }
  return ret;
}

DataType ComputeOpNode::output_dtype(size_t idx) const {
  CHECK_LT(idx, num_outputs());
  return body[idx].dtype();
}

Array<PrimExpr> BaseComputeOpNode::output_shape(size_t idx) const {
  CHECK_LT(idx, num_outputs());
  // for now, all outputs of a BaseComputeOp have the same shape
  Array<PrimExpr> shape;
  for (const auto& extent : this->output_shape_storage) {
    shape.push_back(extent);
  }
  return shape;
}

Array<DimInfo> BaseComputeOpNode::GetAllDimensions() const { return this->all_dimensions; }

Array<Dimension> BaseComputeOpNode::GetRootIndexDimensions(size_t val_idx) const {
  return this->root_index_dimensions;
}

Dimension BaseComputeOpNode::GetBaseIndexDimension(size_t val_idx, size_t dim_idx) const {
  CHECK_LT(val_idx, num_outputs());
  return this->root_index_dimensions[dim_idx];
}

Modes BaseComputeOpNode::output_layout(size_t i) const {
  if (storage_layouts.size() > 0) {
    return storage_layouts[i];
  } else {
    return NullValue<Modes>();
  }
}

Modes BaseComputeOpNode::loop_layout() const { return loop_layout_object; }

Tensor compute(Array<PrimExpr> shape, FCompute fcompute, std::string name, std::string tag,
               Map<std::string, ObjectRef> attrs) {
  auto op_node = make_object<ComputeOpNode>();
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  Array<Dimension> root_dims;
  Array<DimInfo> dim_infos;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    auto iv = IterVarNode::make(Range(0, shape[i]), Var(os.str(), shape[i].dtype()), kDataPar);
    auto dim = DimensionNode::make("dummy_dim", DimensionNode::kRangeDim);
    axis.emplace_back(iv);
    root_dims.push_back(dim);
    dim_infos.push_back(DimInfoNode::make(dim, iv));
    args.push_back(axis.back()->var);
  }

  auto loop_layout = ModesNode::make(name, shape, true);

  auto n = make_object<ComputeOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->axis = std::move(axis);
  n->loop_layout_object = std::move(loop_layout);
  n->output_shape_storage = std::move(shape);

  n->root_index_dimensions = std::move(root_dims);
  n->body = {fcompute(args)};
  if (n->body[0]->IsInstance<tir::ReduceNode>()) {
    const tir::ReduceNode* reduce = n->body[0].as<tir::ReduceNode>();
    n->reduce_axis = reduce->axis;
    n->reduction_dimensions = reduce->dimensions;
  }
  n->all_dimensions = std::move(dim_infos);

  VerifyComputeOp(n.get());
  n->RefreshDimVarMappings();
  Operation op(n);

  return op.output(0);
}

Array<Tensor> compute(Array<PrimExpr> shape, FBatchCompute fcompute, std::string name,
                      std::string tag, Map<std::string, ObjectRef> attrs) {
  auto op_node = make_object<ComputeOpNode>();
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  Array<Dimension> root_dims;
  Array<DimInfo> dim_infos;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    auto iv = IterVarNode::make(Range(0, shape[i]), Var(os.str(), shape[i].dtype()), kDataPar);
    axis.emplace_back(iv);
    args.push_back(axis.back()->var);
    auto dim = DimensionNode::make("dummy_dim", DimensionNode::kRangeDim);
    root_dims.push_back(dim);
    dim_infos.push_back(DimInfoNode::make(dim, iv));
  }

  if (!attrs.defined()) {
    attrs = Map<std::string, ObjectRef>();
  }
  auto n = make_object<ComputeOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->axis = std::move(axis);
  n->output_shape_storage = std::move(shape);

  n->root_index_dimensions = std::move(root_dims);
  n->body = fcompute(args);
  if (n->body[0]->IsInstance<tir::ReduceNode>()) {
    const tir::ReduceNode* reduce = n->body[0].as<tir::ReduceNode>();
    n->reduce_axis = reduce->axis;
    n->reduction_dimensions = reduce->dimensions;
  }
  n->all_dimensions = std::move(dim_infos);

  VerifyComputeOp(n.get());
  n->RefreshDimVarMappings();
  Operation op(n);

  Array<Tensor> outputs;
  for (int idx = 0; idx < op->num_outputs(); ++idx) {
    outputs.push_back(op.output(idx));
  }
  return outputs;
}

Operation ComputeOpNode::make(std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                              Array<IterVar> axis, Array<PrimExpr> body) {
  Array<PrimExpr> shape;
  Array<Var> parameters;
  for (size_t i = 0; i < axis.size(); ++i) {
    shape.push_back(axis[i]->dom->extent);
    parameters.push_back(Var("arg" + std::to_string(i), DataType::Int(32)));
  }
  return ComputeOpNode::make(name, tag, attrs, axis, {}, shape, {}, {}, body, {});
}

void InitComputeOpFields(const Array<UninterpFun>& axis_min_ufs,
                         const Array<UninterpFun>& axis_extent_ufs,
                         const Array<Dimension>& dimensions, Array<IterVar>* axis) {
  Array<PrimExpr> args;
  Array<Dimension> arg_dims;
  // compute dimension.
  CHECK_EQ(axis_min_ufs.size(), axis_extent_ufs.size());
  size_t num_loops = axis_min_ufs.size();
  for (size_t i = 0; i < num_loops; ++i) {
    std::ostringstream os;
    os << "axlv" << i;
    PrimExpr min = axis_min_ufs[i].MakeCallTo(Array<PrimExpr>(args), Array<Dimension>(arg_dims));
    PrimExpr extent =
        axis_extent_ufs[i].MakeCallTo(Array<PrimExpr>(args), Array<Dimension>(arg_dims));
    auto iv = IterVarNode::make(Range::make_by_min_extent(min, extent),
                                Var(os.str(), DataType::Int(32)), kDataPar);
    axis->push_back(iv);
    args.push_back(iv->var);
    arg_dims.push_back(dimensions[i]);
  }
}

Array<Tensor> compute(Array<PrimExpr> shape, FBatchComputeMap fcompute, FBatchComputeMap fpred,
                      std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                      Array<IterVar> axis, Array<Dimension> dimensions) {
  Map<Dimension, Var> body_args;
  for (size_t i = 0; i < dimensions.size(); ++i) {
    body_args.Set(dimensions[i], axis[i]->var);
  }

  Operation op = ComputeOpNode::make(name, tag, attrs, axis, dimensions, shape, {}, {},
                                     fcompute(body_args), fpred(body_args));
  Array<Tensor> outputs;
  for (int idx = 0; idx < op->num_outputs(); ++idx) {
    outputs.push_back(op.output(idx));
  }
  return outputs;
}

Array<Tensor> compute(Array<PrimExpr> shape, FBatchCompute fcompute, FBatchCompute fpred,
                      std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                      Array<UninterpFun> axis_min_ufs, Array<UninterpFun> axis_extent_ufs,
                      Array<Dimension> dimensions) {
  Array<IterVar> axis;
  InitComputeOpFields(axis_min_ufs, axis_extent_ufs, dimensions, &axis);

  Array<Var> body_args;
  for (size_t i = 0; i < axis.size(); ++i) {
    body_args.push_back(axis[i]->var);
  }

  Operation op = ComputeOpNode::make(name, tag, attrs, axis, dimensions, shape, {}, {},
                                     fcompute(body_args), fpred(body_args));
  Array<Tensor> outputs;
  for (int idx = 0; idx < op->num_outputs(); ++idx) {
    outputs.push_back(op.output(idx));
  }
  return outputs;
}

Array<Tensor> compute(Array<PrimExpr> shape, FBatchComputeMap fcompute, FBatchComputeMap fpred,
                      std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                      Array<UninterpFun> axis_min_ufs, Array<UninterpFun> axis_extent_ufs,
                      Array<Dimension> dimensions) {
  Array<IterVar> axis;
  InitComputeOpFields(axis_min_ufs, axis_extent_ufs, dimensions, &axis);

  Map<Dimension, Var> body_args;
  for (size_t i = 0; i < dimensions.size(); ++i) {
    body_args.Set(dimensions[i], axis[i]->var);
  }

  Operation op = ComputeOpNode::make(name, tag, attrs, axis, dimensions, shape, {}, {},
                                     fcompute(body_args), fpred(body_args));
  Array<Tensor> outputs;
  for (int idx = 0; idx < op->num_outputs(); ++idx) {
    outputs.push_back(op.output(idx));
  }
  return outputs;
}

Operation ComputeOpNode::make(std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                              Array<IterVar> axis, Array<Dimension> dimensions,
                              Array<PrimExpr> output_shape_storage, Array<Modes> storage_layouts,
                              Modes loop_layout_object, Array<PrimExpr> body,
                              Array<PrimExpr> pred) {
  bool print = false;  //(name == "B.shared1" || name == "A.local");
  if (print) {
    std::cout << "[COP] Creating COP " << name << " " << storage_layouts[0] << std::endl;
  }
  if (!attrs.defined()) {
    attrs = Map<std::string, ObjectRef>();
  }
  auto n = make_object<ComputeOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->axis = std::move(axis);
  n->output_shape_storage = std::move(output_shape_storage);
  n->storage_layouts = std::move(storage_layouts);
  n->loop_layout_object = std::move(loop_layout_object);

  n->root_index_dimensions = std::move(dimensions);
  n->body = std::move(body);
  n->pred = std::move(pred);
  if (n->body[0]->IsInstance<tir::ReduceNode>()) {
    const tir::ReduceNode* reduce = n->body[0].as<tir::ReduceNode>();
    n->reduce_axis = reduce->axis;
    n->reduction_dimensions = reduce->dimensions;
  }

  CHECK_EQ(n->axis.size(), n->root_index_dimensions.size()) << n->name;

  for (size_t i = 0; i < n->axis.size(); ++i) {
    CHECK(n->root_index_dimensions[i]->type != DimensionNode::kFunDim);
    n->all_dimensions.push_back(DimInfoNode::make(n->root_index_dimensions[i], n->axis[i]));
    if (print) {
      std::cout << "[COP]  Axis " << n->axis[i] << std::endl;
    }
  }

  // CHECK(n->reduce_axis.size() == n->reduction_dimensions.size() ||
  // n->reduction_dimensions.size() == 0);
  CHECK_EQ(n->reduce_axis.size(), n->reduction_dimensions.size()) << n->name;
  if (n->reduction_dimensions.size() > 0) {
    for (size_t i = 0; i < n->reduce_axis.size(); ++i) {
      CHECK(n->reduction_dimensions[i]->type != DimensionNode::kFunDim);
      n->all_dimensions.push_back(DimInfoNode::make(n->reduction_dimensions[i], n->reduce_axis[i]));
      // if (print) {
      // std::cout << "[COP]  RAxs " << n->reduce_axis[i] << std::endl;
      // }
    }
  }

  VerifyComputeOp(n.get());
  n->RefreshDimVarMappings();
  auto ret = Operation(n);
  // if (n->storage_layouts.size() > 0) {
  //   std::cout << "[COP] " << n->name << std::endl;
  //   for (auto lf: n->storage_layouts[0]->l_funs) {
  //     std::cout << "[COP]   " << lf << std::endl;
  //   }
  // }
  return ret;
}

void ComputeOpNode::RefreshDimVarMappings() {
  this->dim2var_maps.clear();
  std::unordered_map<const DimensionNode*, DimVarEntry> dim2var_map;
  for (const auto dim_info : all_dimensions) {
    dim2var_map[dim_info->dim.as<DimensionNode>()] = {dim_info->dim, dim_info->iv};
    this->var2dim_map[dim_info->iv->var.as<VarNode>()] = dim_info->dim.as<DimensionNode>();
  }
  for (size_t i = 0; i < this->num_outputs(); ++i) {
    this->dim2var_maps.push_back(std::move(dim2var_map));
  }
}

TVM_REGISTER_GLOBAL("te.ComputeOp")
    .set_body_typed([](std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                       Array<IterVar> axis, Array<Dimension> dimensions,
                       Array<PrimExpr> output_shape_storage, Array<Modes> storage_layouts,
                       Modes loop_layout_object, Array<PrimExpr> body, Array<PrimExpr> pred) {
      return ComputeOpNode::make(name, tag, attrs, axis, dimensions, output_shape_storage,
                                 storage_layouts, loop_layout_object, body, pred);
    });

// The schedule related logics
Array<Tensor> ComputeOpNode::InputTensors() const {
  bool print = false;  //(this->name == "lnext_v.ila");
  if (print) std::cout << "[IT] Input tensors for " << GetRef<Operation>(this) << std::endl;
  Array<Tensor> ret;
  Array<PrimExpr> toCollectIn;
  for (auto& e : body) {
    toCollectIn.push_back(e);

    if (print)
      std::cout << "[IT0] " << this->name << " " << UninterpFun::InlineUninterpFunCalls(e)
                << std::endl;
  }

  for (auto& e : pred) {
    toCollectIn.push_back(e);
    if (print)
      std::cout << "[IT01] " << this->name << " " << UninterpFun::InlineUninterpFunCalls(e)
                << std::endl;
  }

  for (const auto dim_info : all_dimensions) {
    // if (print) std::cout << "[IT0] Dim " << dim_info->dim << " " << std::endl;
    CHECK(!dim_info->dim->isFunDim());
    toCollectIn.push_back(UninterpFun::InlineUninterpFunCalls(dim_info->iv->dom->min));
    if (print)
      std::cout << "[IT2] " << this->name << " "
                << UninterpFun::InlineUninterpFunCalls(dim_info->iv->dom->min) << std::endl;
    toCollectIn.push_back(UninterpFun::InlineUninterpFunCalls(dim_info->iv->dom->extent));
    if (print)
      std::cout << "[IT3] " << this->name << " "
                << UninterpFun::InlineUninterpFunCalls(dim_info->iv->dom->extent) << std::endl;
  }
  CollectTensors(ret, toCollectIn);
  for (Tensor t : ret) {
    if (print) std::cout << "[IT]   Input " << t->op << std::endl;
  }
  return ret;
}

Array<Tensor> ComputeOpNode::InputTensorsOnlyBody() const {
  bool print = false;  //(this->name == "lnext_v.ila");
  if (print) std::cout << "[IT] Input tensors for " << GetRef<Operation>(this) << std::endl;
  Array<Tensor> ret;
  Array<PrimExpr> toCollectIn;
  for (auto& e : body) {
    toCollectIn.push_back(e);

    if (print)
      std::cout << "[IT0] " << this->name << " " << UninterpFun::InlineUninterpFunCalls(e)
                << std::endl;
  }

  for (auto& e : pred) {
    toCollectIn.push_back(e);
    if (print)
      std::cout << "[IT01] " << this->name << " " << UninterpFun::InlineUninterpFunCalls(e)
                << std::endl;
  }

  CollectTensors(ret, toCollectIn);
  for (Tensor t : ret) {
    if (print) std::cout << "[IT]   Input " << t->op << std::endl;
  }
  return ret;
}

Operation ComputeOpNode::ReplaceInputs(const Operation& self,
                                       const std::unordered_map<Tensor, Tensor>& rmap) const {
  // if (self->name == "r_mv.local" || self->name == "c_prev")
  // std::cout << "REPL " << self << std::endl;
  CHECK_EQ(self.operator->(), this);
  VerifyComputeOp(this);
  Array<PrimExpr> arr;
  if (this->body[0]->IsInstance<tir::ReduceNode>()) {
    // Specially handle reduce so the replaced op is shared across all
    // components (different output tensors)
    PrimExpr new_reduce = te::ReplaceTensor(this->body[0], rmap);
    if (!new_reduce.same_as(this->body[0])) {
      const tir::ReduceNode* r = new_reduce.as<tir::ReduceNode>();
      for (size_t k = 0; k < this->body.size(); ++k) {
        auto n = make_object<tir::ReduceNode>(*r);
        n->value_index = static_cast<int>(k);
        n->dtype = r->source[k].dtype();
        arr.push_back(PrimExpr(n));
      }
    } else {
      arr = this->body;
    }
  } else {
    arr =
        UpdateArray(this->body, [&rmap](const PrimExpr& e) { return te::ReplaceTensor(e, rmap); });
  }

  bool changed = false;
  if (!arr.same_as(this->body)) {
    changed = true;
  }

  Array<PrimExpr> pred_arr;
  pred_arr =
      UpdateArray(this->pred, [&rmap](const PrimExpr& e) { return te::ReplaceTensor(e, rmap); });
  if (!pred_arr.same_as(this->pred)) {
    changed = true;
  }

  Array<IterVar> new_axis;
  bool print = false;  //(self->name == "ii_s_h2h.ila");
  for (const auto& iv : axis) {
    PrimExpr old_extent = iv->dom->extent;
    PrimExpr new_extent = te::ReplaceTensor(old_extent, rmap);
    if (!old_extent.same_as(new_extent)) {
      changed = true;
    }

    // TODO(ppf): Mighty hack: As IterVars for this stage may already
    // be referred from IterVarRelations and the corresponding stage,
    // we would need to replace IterVars at all of those places if we
    // create new IterVars here. Instead we merely change the ranges
    // of the IterVars and reuse them.
    const_cast<IterVarNode*>(iv.as<IterVarNode>())
        ->set_dom(Range::make_by_min_extent(iv->dom->min, new_extent));
    new_axis.push_back(iv);
  }

  Array<Modes> new_storage_layouts;
  if (this->storage_layouts.defined()) {
    for (auto layout : this->storage_layouts) {
      auto new_layout = te::ReplaceTensor(layout, rmap);
      if (new_layout != layout) {
        changed = true;
      }
      new_storage_layouts.push_back(new_layout);
    }
  }

  Modes new_loop_layout = NullValue<Modes>();
  if (this->loop_layout_object.defined()) {
    Modes new_loop_layout = te::ReplaceTensor(this->loop_layout_object, rmap);
    if (new_loop_layout != this->loop_layout_object) {
      changed = true;
    }
  }

  if (changed) {
    Operation ret = ComputeOpNode::make(this->name, this->tag, this->attrs, new_axis,
                                        this->root_index_dimensions, this->output_shape_storage,
                                        new_storage_layouts, new_loop_layout, arr, pred_arr);
    auto mut_op = const_cast<ComputeOpNode*>(ret.as<ComputeOpNode>());
    mut_op->set_realize_bounds(this->realize_bounds, this->who_set_realize_bounds);
    mut_op->output_buffer = this->output_buffer;
    mut_op->output_buffer_dims = this->output_buffer_dims;
    mut_op->storage_layouts = this->storage_layouts;
    mut_op->loop_layout_object = this->loop_layout_object;
    return ret;
  } else {
    return self;
  }
}

PrimExpr ReplaceIndexVariables(PrimExpr expr, Array<DimInfo> dim_infos) {
  std::unordered_map<const VarNode*, PrimExpr> replace_map;
  Array<PrimExpr> args;
  Array<Dimension> arg_dims;
  for (size_t i = 0; i < dim_infos.size(); ++i) {
    auto dim_info = dim_infos[i];
    auto var = dim_info->iv->var;
    auto dim = dim_info->dim;
    CHECK(!dim->isFunDim());
    args.push_back(var);
    arg_dims.push_back(dim);
  }
  return VarReplacer(replace_map)(expr);
}

void ComputeOpNode::PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                                      const std::unordered_map<const VarNode*, IntSet>& dom_map,
                                      std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
  auto fvisit = [&dom_map, out_dom_map, analyzer, this](const ObjectRef& n) {
    auto* call = n.as<tir::CallNode>();
    if (call != nullptr && call->func.defined() && call->func.as<OperationNode>()) {
      Tensor t = Downcast<Operation>(call->func).output(call->value_index);

      if (t->op.defined() && out_dom_map->count(t)) {
        bool print = false;
        // bool print = (t->op->name == "A.shared");
        if (print) std::cout << "[PBIc] Op " << this->name << " " << t << " " << n << std::endl;

        if (print) {
          for (auto it : dom_map) {
            std::cout << "[PBIc]   DomMap " << it.first->name_hint << " " << it.second << std::endl;
          }
        }

        TensorDom& dom = out_dom_map->at(t);
        for (size_t i = 0; i < t.ndim(); ++i) {
          // print = print && (i == 0);

          // We assume that the value of the argument cannot be out of bounds (otherwise it is
          // undefined behaviour), so we can intersect the estimated set of the argument with the
          // range expected by the tensor. However, intersection may result in overly complex
          // expressions, so we perform a more relaxed form of intersection.

          PrimExpr inlined_arg = ReplaceIndexVariables(call->args[i], this->all_dimensions);
          IntSet arg_intset = EvalSet(inlined_arg, dom_map);

          if (print)
            std::cout << "[PBIc]   Repl " << i << " " << inlined_arg << " " << arg_intset
                      << std::endl;
          arg_intset =
              TranslateIterVarsFromConsumerToProducer(arg_intset, GetRef<Operation>(this), t);
          if (print) {
            std::cout << "[PBIc]    Arg intset for " << inlined_arg << " " << arg_intset
                      << std::endl;
          }

          const arith::IntervalSetNode* arg_interval = arg_intset.as<arith::IntervalSetNode>();
          if (arg_interval) {
            PrimExpr shape_i_min_value = make_zero(t->shape[i].dtype());
            PrimExpr shape_i_max_value = t->shape[i] - 1;
            PrimExpr min_value = arg_interval->min_value;
            PrimExpr max_value = arg_interval->max_value;

            if (auto bvd_op = t->op.as<BaseVarDimOpNode>()) {
              auto i_dim = bvd_op->GetRootIndexDimensions(t->value_index)[i];
              Range r = bvd_op->GetIterVarFromDim(t->value_index, i_dim)->dom;
              shape_i_min_value = r->min;
              shape_i_max_value = r->max_inclusive();
            }

            bool can_prove_min = analyzer->CanProve(shape_i_min_value >= min_value);
            bool can_prove_max = analyzer->CanProve(shape_i_max_value <= max_value);
            if (print) {
              std::cout << "[PBIc]   Min " << (shape_i_min_value >= min_value) << std::endl;
              std::cout << "[PBIc]     Result " << can_prove_min << std::endl;
              std::cout << "[PBIc]   Max " << (shape_i_max_value <= max_value) << std::endl;
              std::cout << "[PBIc]     Result " << can_prove_max << std::endl;
            }
            // exit(0);

            // Prefer the shape bounds only when we can prove they are tighter.
            if (arith::is_neg_inf(min_value) || can_prove_min) {
              if (print) std::cout << "[PBIc]     Approx 1" << std::endl;
              min_value = shape_i_min_value;
            }
            if (arith::is_pos_inf(max_value) || can_prove_max) {
              if (print) std::cout << "[PBIc]     Approx 2" << std::endl;
              max_value = shape_i_max_value;
            }
            dom.data[i].push_back(IntSet::interval(min_value, max_value));
            if (print)
              std::cout << "[PBIc]      Pushing1 " << i << " "
                        << IntSet::interval(min_value, max_value) << std::endl;
          } else {
            dom.data[i].push_back(arg_intset);
            if (print) std::cout << "[PBIc]      Pushing2 " << i << " " << arg_intset << std::endl;
          }
        }
      }
    }
  };
  for (auto& e : body) {
    tir::PostOrderVisit(e, fvisit);
  }
  {
    Array<PrimExpr> args;
    Array<Dimension> arg_dims;
    // std::cout << "[PBIc] Call " << self->name << std::endl;
    for (size_t i = 0; i < all_dimensions.size(); ++i) {
      auto dim_info = all_dimensions[i];
      auto var = dim_info->iv->var;
      auto dim = dim_info->dim;
      CHECK(!dim->isFunDim());
      tir::PostOrderVisit(dim_info->iv->dom->min, fvisit);
      tir::PostOrderVisit(UninterpFun::InlineUninterpFunCalls(dim_info->iv->dom->extent), fvisit);

      args.push_back(var);
      arg_dims.push_back(dim);
      // std::cout << "[PBIc]   Dim " << dim << std::endl;
    }
  }
}

void BaseComputeOpNode::GatherBound(const Operation& self,
                                    const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                                    std::unordered_map<IterVar, Range>* out_dom_map,
                                    const Map<FunctionRef, CacheInfo> cacheTensorInfos) const {
  auto compute_op = self.as<BaseComputeOpNode>();
  bool print = false;
  // bool print = (self->name == "O.2.1.local");
  if (print) std::cout << "[GBC] Op " << self->name << std::endl;

  CHECK_EQ(self.operator->(), this);
  const TensorDom& tdom = tensor_dom.at(self.output(0));

  Map<IterVar, IntSet> lv_sets_map;
  for (size_t i = 0; i < output_shape_storage.size(); ++i) {
    Dimension idx_dim = root_index_dimensions[i];
    for (auto iset : tdom.data.at(i)) {
      if (print) std::cout << "[GBC]    Dim0 " << iset << std::endl;
    }

    IntSet iv_set = arith::Union(tdom.data.at(i));
    if (print) std::cout << "[GBC]  Dim " << idx_dim->name << " " << iv_set << std::endl;
    CHECK(idx_dim->isLoopDim());
    IterVar lv = compute_op->GetIterVarFromDim(0, idx_dim);
    if (print)
      std::cout << "[GBC]   Dim0.0 " << idx_dim->name << " " << lv->var->name_hint << " "
                << Simplify(UninterpFun::InlineUninterpFunCalls(iv_set.max() - iv_set.min() + 1))
                << std::endl;
    if (lv_sets_map.count(lv)) {
      lv_sets_map.Set(lv, arith::Union({lv_sets_map.at(lv), iv_set}));
    } else {
      lv_sets_map.Set(lv, iv_set);
    }
  }

  for (auto it : lv_sets_map) {
    if (print) std::cout << "[GBC]  Dim1 " << it.first->var->name_hint << std::endl;
    if (out_dom_map->find(it.first) == out_dom_map->end()) {
      if (print)
        std::cout << "[GBC]     Covering range " << it.second << " " << it.first->dom << std::endl;
      (*out_dom_map)[it.first] = it.second.cover_range(it.first->dom);
      if (print)
        std::cout << "[GBC]     " << it.first->dom << " "
                  << UninterpFun::InlineUninterpFunCalls((*out_dom_map)[it.first]) << std::endl;
    }
  }

  for (size_t i = 0; i < this->axis.size(); ++i) {
    if (print) std::cout << "[GBC]  Dim2 " << this->axis[i]->var->name_hint << std::endl;
    if (out_dom_map->find(this->axis[i]) == out_dom_map->end()) {
      (*out_dom_map)[this->axis[i]] = this->axis[i]->dom;
      if (print)
        std::cout << "[GBC]     "
                  << UninterpFun::InlineUninterpFunCalls((*out_dom_map)[this->axis[i]])
                  << std::endl;
    }
  }

  // Handle reduce axes separately
  for (size_t i = 0; i < this->reduce_axis.size(); ++i) {
    CHECK(!out_dom_map->count(this->reduce_axis[i])) << this->reduce_axis[i];
    (*out_dom_map)[this->reduce_axis[i]] = this->reduce_axis[i]->dom;
    if (print)
      std::cout << "[GBC]  DimFReduce " << this->reduce_axis[i]->var << " "
                << this->reduce_axis[i].get() << " " << (*out_dom_map)[this->reduce_axis[i]]
                << std::endl;
  }

  for (const auto& di : this->all_dimensions) {
    CHECK(di->dim->isLoopDim());
    if (print) {
      std::cout << "[GBC]  DimF " << di->dim->name << " " << di->iv->var->name_hint << std::endl;
      std::cout << "[GBC]  DimF   " << UninterpFun::InlineUninterpFunCalls((*out_dom_map)[di->iv])
                << std::endl;
    }
  }
}

void BaseComputeOpNode::set_realize_bounds(Array<Range> bounds, std::string caller) {
  // std::cout << "[SRB]  " << GetRef<Operation>(this) << " " << bounds.size() << " " << caller
  //           << std::endl;
  this->realize_bounds = std::move(bounds);
  this->who_set_realize_bounds = caller;
}

void BaseComputeOpNode::set_storage_layout(int i, Modes layout) {
  ArrayNode* storage_layouts = this->storage_layouts.CopyOnWrite();
  storage_layouts->data[i] = layout;
}

void BaseComputeOpNode::set_all_dimensions(Array<DimInfo> dim_infos) {
  this->all_dimensions = std::move(dim_infos);
}

Region BaseComputeOpNode::GetRealizeBounds(
    const Stage& stage, const std::unordered_map<IterVar, Range>& realize_map) const {
  bool print = false;  //(stage->op->name == "O.local");
  CHECK_EQ(stage->op.get(), this);

  Region bounds;
  // bool to_relax = !stage.is_ancestor_attached_at_root();
  bool to_relax = false;  //! stage.is_ancestor_attached_at_root();

  if (print) std::cout << "[BR] Build realize for " << stage << " " << to_relax << std::endl;
  CHECK(realize_bounds.defined());
  CHECK_EQ(realize_bounds.size(), stage->dim_relation_graph->leaf_dimensions.size())
      << stage << " " << stage->op << " " << who_set_realize_bounds;
  for (size_t i = 0; i < stage->dim_relation_graph->leaf_dimensions.size(); ++i) {
    Dimension dim = stage->dim_relation_graph->leaf_dimensions[i];
    if (print)
      std::cout << "[BR]  Unrelaxed " << realize_bounds[i] << " " << dim << " " << dim->type
                << std::endl;

    Range r = realize_bounds[i];
    if (to_relax) {
      // N.B.: Here, in order to ensure that we don't allocate a
      // buffer with a variable size, we relax the extent of the
      // realize range to no include any calls to complex uninterp
      // functions. This is more of a hack as the bounds of the
      // realize node might be used for purposes other than just
      // deciding the size of the buffer to allocate. But by the time
      // we create the AllocateNode in storage_flatten.cc, we have
      // inlined all calls to uninterp functions and can no longer
      // effectively relax them. Ideally, we should hold off on
      // inlining uninterp function calls to as late a stage as
      // possible.
      r = Range::make_by_min_extent(r->min, UninterpFun::RelaxUninterpCallsMaxInclusive(r->extent));
      if (print) std::cout << "[BR]  Relaxed " << r << std::endl;
    }

    bounds.push_back(r);
  }
  return bounds;
}

Stmt BaseComputeOpNode::BuildRealize(const Stage& stage,
                                     const std::unordered_map<IterVar, Range>& realize_map,
                                     const Stmt& body) const {
  CHECK_EQ(stage->op.get(), this);

  Region bounds = GetRealizeBounds(stage, realize_map);
  Stmt realize = body;
  for (int i = this->num_outputs(); i > 0; --i) {
    Tensor t = stage->op.output(i - 1);
    realize = tir::RealizeNode::make(
        t->op, t->value_index, t->dtype, bounds, const_true(), realize,
        stage.is_ancestor_attached_at_root() ? output_layout(t->value_index) : NullValue<Modes>());
    // alignment requirement, only useful for compute
    for (size_t i = 0; i < stage->dim_relation_graph->leaf_dimensions.size(); ++i) {
      Dimension dim = stage->dim_relation_graph->leaf_dimensions[i];

      auto it = stage->align_info.find(dim.as<DimensionNode>());
      if (it != stage->align_info.end()) {
        auto pair = (*it).second;
        if (pair.first != 0) {
          Array<PrimExpr> tuple = {static_cast<int>(i), pair.first, pair.second};
          realize =
              tir::AttrStmtNode::make(t, tir::attr::buffer_dim_align,
                                      CallNode::make(DataType::Handle(), tir::intrinsic::tvm_tuple,
                                                     tuple, CallNode::Intrinsic),
                                      realize);
        }
      }
    }
  }
  return realize;
}

size_t ComputeOpNode::num_schedulable_dims() const { return axis.size(); }

// Build a reduction body.
void MakeReduction(const Stage s, const ComputeOpNode* op,
                   const std::unordered_map<IterVar, Range>& dom_map, const Array<Tensor>& tensors,
                   std::unordered_map<IterVar, PrimExpr> init_vmap,
                   std::unordered_map<IterVar, PrimExpr> main_vmap, Stmt* init, Stmt* provide) {
  bool print = false;  // op->name == "is_h2h.ila";
  std::unordered_map<const DimensionNode*, Range> dim_doms;
  for (auto dim : op->root_index_dimensions) {
    auto iv = op->GetIterVarFromDim(0, dim);
    if (dom_map.count(iv)) {
      dim_doms[dim.operator->()] = dom_map.at(op->GetIterVarFromDim(0, dim));
    } else {
      dim_doms[dim.operator->()] = iv->dom;
    }
  }

  DimensionPassDownDomain(s, op, &dim_doms, true);

  std::unordered_map<const DimensionNode*, PrimExpr> dim_vals;
  for (auto dim : op->root_index_dimensions) {
    dim_vals[dim.operator->()] = op->GetIterVarFromDim(0, dim)->var;
  }

  DimensionPassDownValues(s, op, dim_doms, &dim_vals, true);
  // DimensionPassDownValues(s, op, &dim_vals, true);

  Array<PrimExpr> args;
  for (auto dim : s->dim_relation_graph->leaf_dimensions) {
    if (print) std::cout << "[MP] Arg " << dim << " " << dim_vals[dim.operator->()] << std::endl;
    args.push_back(dim_vals[dim.operator->()]);
  }

  // Array<PrimExpr> args;
  // for (auto dim : op->root_index_dimensions) {
  //   args.push_back(op->GetIterVarFromDim(0, dim)->var);
  // }

  std::vector<Stmt> inits, provides;

  Array<Range> init_realize_bounds;
  Array<Range> main_realize_bounds;
  {
    Array<Range> realize_bounds = op->GetRealizeBounds(s, dom_map);

    std::unordered_map<const VarNode*, PrimExpr> main_vsub;
    for (auto it : main_vmap) {
      main_vsub[it.first->var.operator->()] = it.second;
    }
    std::unordered_map<const VarNode*, PrimExpr> init_vsub;
    for (auto it : init_vmap) {
      init_vsub[it.first->var.operator->()] = it.second;
    }

    VarReplacer main_replacer(main_vsub);
    VarReplacer init_replacer(init_vsub);
    for (auto r : realize_bounds) {
      init_realize_bounds.push_back(init_replacer.replace(r));
      main_realize_bounds.push_back(main_replacer.replace(r));
    }

    // std::cout << "[COP] Orignal Realize bounds for " << op->name << std::endl;
    // for (auto r : realize_bounds) {
    //   std::cout << "[COP]    " << r << std::endl;
    // }
    // std::cout << "[COP] Init Realize bounds for " << op->name << std::endl;
    // for (auto r : init_realize_bounds) {
    // std::cout << "[COP]    " << r << std::endl;
    // }
    // std::cout << "[COP] Main Realize bounds for " << op->name << std::endl;
    // for (auto r : main_realize_bounds) {
    //   std::cout << "[COP]    " << r << std::endl;
    // }
    // std::cout << "[COP] Main vmap for " << op->name << std::endl;
    // for (auto it : main_vmap) {
    //   std::cout << "[COP]    " << it.first->var << " " << it.second << std::endl;
    // }
  }

  size_t size = op->body.size();
  const ReduceNode* reduce = op->body[0].as<ReduceNode>();
  CHECK(reduce);
  const CommReducerNode* combiner = reduce->combiner.as<CommReducerNode>();
  CHECK(combiner);
  Array<PrimExpr> lhs;
  for (size_t i = 0; i < size; ++i) {
    lhs.push_back(CallNode::make(tensors[i]->dtype, tensors[i]->op->name, args, CallNode::Halide,
                                 {}, tensors[i]->op, tensors[i]->value_index, main_realize_bounds));
  }

  Array<PrimExpr> init_value = combiner->identity_element;
  Array<PrimExpr> update_value = (*combiner)(lhs, reduce->source);
  for (size_t i = 0; i < size; ++i) {
    Tensor t = tensors[i];
    inits.emplace_back(
        ProvideNode::make(t->op, t->value_index, init_value[i], args, init_realize_bounds));
    provides.emplace_back(
        ProvideNode::make(t->op, t->value_index, update_value[i], args, main_realize_bounds));
  }
  *init = SeqStmt::Flatten(inits);
  *provide = SeqStmt::Flatten(provides);
  if (!is_one(reduce->condition)) {
    *provide = IfThenElseNode::make(reduce->condition, *provide);
  }
}

// Normal computation.
Stmt MakeProvide(const Stage s, const ComputeOpNode* op,
                 const std::unordered_map<IterVar, Range>& dom_map,
                 std::unordered_map<IterVar, PrimExpr> vmap, const Tensor& t) {
  std::unordered_map<const DimensionNode*, Range> dim_doms;
  for (auto dim : op->root_index_dimensions) {
    auto iv = op->GetIterVarFromDim(0, dim);
    if (dom_map.count(iv)) {
      dim_doms[dim.operator->()] = dom_map.at(op->GetIterVarFromDim(0, dim));
    } else {
      dim_doms[dim.operator->()] = iv->dom;
    }
  }

  DimensionPassDownDomain(s, op, &dim_doms, true);

  std::unordered_map<const DimensionNode*, PrimExpr> dim_vals;
  for (auto dim : op->root_index_dimensions) {
    dim_vals[dim.operator->()] = op->GetIterVarFromDim(0, dim)->var;
  }

  DimensionPassDownValues(s, op, dim_doms, &dim_vals, true);

  Array<Range> realize_bounds;
  {
    Array<Range> unreplaced_realize_bounds = op->GetRealizeBounds(s, dom_map);

    std::unordered_map<const VarNode*, PrimExpr> vsub;
    for (auto it : vmap) {
      vsub[it.first->var.operator->()] = it.second;
    }

    VarReplacer replacer(vsub);
    for (auto r : unreplaced_realize_bounds) {
      realize_bounds.push_back(replacer.replace(r));
    }
  }

  Array<PrimExpr> args;
  for (auto dim : s->dim_relation_graph->leaf_dimensions) {
    args.push_back(dim_vals[dim.operator->()]);
  }
  auto provide = ProvideNode::make(t->op, t->value_index, arith::Simplify(op->body[t->value_index]),
                                   args, realize_bounds);
  if (op->output_buffer.defined()) {
    Array<PrimExpr> buf_args;
    for (auto dim : op->output_buffer_dims) {
      buf_args.push_back(op->GetIterVarFromDim(0, dim)->var);
    }
    Stmt output_buffer_write =
        op->output_buffer.vstore(buf_args, op->body[t->value_index], tir::kNone);
    return SeqStmt({provide, output_buffer_write});
  } else {
    return provide;
  }
}

const VarNode* get_defined_var(Stmt stmt) {
  if (const auto* for_ = stmt.as<ForNode>()) {
    return for_->loop_var.operator->();
  } else if (const auto* let = stmt.as<LetStmtNode>()) {
    return let->var.operator->();
  } else if (const auto* attr = stmt.as<AttrStmtNode>()) {
    if (attr->attr_key == attr::thread_extent || attr->attr_key == attr::virtual_thread ||
        attr->attr_key == attr::pipeline_exec_scope || attr->attr_key == attr::loop_scope) {
      return Downcast<IterVar>(attr->node)->var.operator->();
    } else {
      return nullptr;
    }
  } else if (const auto* ite = stmt.as<IfThenElseNode>()) {
    return nullptr;
  } else if (const auto* seq = stmt.as<SeqStmtNode>()) {
    return nullptr;
  } else if (const auto* assert_ = stmt.as<AssertStmtNode>()) {
    return nullptr;
  } else if (const auto* alloc = stmt.as<AllocateNode>()) {
    return alloc->buffer_var.operator->();
  } else {
    return nullptr;
  }
}

std::vector<Stmt> hoist_and_flatten(std::vector<std::vector<Stmt>> stmts) {
  std::vector<Stmt> flattened;
  std::unordered_map<const IfThenElseNode*, std::unordered_set<const VarNode*>> if_vars;

  for (auto vec : stmts) {
    for (auto stmt : vec) {
      if (auto ifn = stmt.as<IfThenElseNode>()) {
        if_vars[ifn] = VarCollector().collect(ifn->condition);
      }
      flattened.push_back(stmt);
    }
  }

  std::vector<Stmt> ret;
  std::unordered_set<const IfThenElseNode*> unadded_ifs;
  for (int i = flattened.size() - 1; i >= 0; --i) {
    auto stmt = flattened[i];
    if (auto ufn = stmt.as<IfThenElseNode>()) {
      unadded_ifs.insert(ufn);
    } else if (auto vn = get_defined_var(stmt)) {
      std::unordered_set<const IfThenElseNode*> to_remove;

      for (auto it = unadded_ifs.begin(); it != unadded_ifs.end(); ++it) {
        if (if_vars[*it].count(vn)) {
          ret.push_back(GetRef<Stmt>(*it));
          to_remove.insert(*it);
        }
      }
      for (auto ite : to_remove) {
        unadded_ifs.erase(ite);
      }
      ret.push_back(stmt);
    } else {
      ret.push_back(stmt);
    }
  }
  for (auto it : unadded_ifs) {
    ret.push_back(GetRef<Stmt>(it));
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

Stmt MakeComputeStmt(const ComputeOpNode* self, const Stage& stage,
                     const std::unordered_map<IterVar, Range>& dom_map,
                     const std::unordered_map<std::string, Range>& env_dom_map,
                     const std::unordered_map<std::string, IterVar>& env_var_map,
                     const std::unordered_map<const VarNode*, std::string>& bind_map,
                     const Map<Stage, Array<Stage>>& attach_stages,
                     const Map<Stage, Array<IterVar>>& attach_vars, bool debug_keep_trivial_loop) {
  // grab the nest structure
  ComputeLoopNest n =
      ComputeLoopNest::make(self, stage, dom_map, env_dom_map, env_var_map, bind_map, attach_stages,
                            attach_vars, debug_keep_trivial_loop);

  // Normal loop structure
  n.init_nest.emplace_back(MakeIfNest(n.init_predicates));
  n.main_nest.emplace_back(MakeIfNest(n.main_predicates));

  if (self->reduce_axis.size() != 0) {
    // make reduction.
    Stmt init, provide;
    Array<Tensor> source;
    for (size_t i = 0; i < self->body.size(); ++i) {
      source.push_back(stage->op.output(i));
    }
    MakeReduction(stage, self, dom_map, source, n.init_vmap, n.main_vmap, &init, &provide);

    init = MergeNest(n.init_nest, init);

    init = Substitute(init, n.init_vmap);
    // common nest
    std::vector<std::vector<Stmt>> common(n.main_nest.begin(),
                                          n.main_nest.begin() + n.num_common_loop + 1);
    std::vector<std::vector<Stmt>> reduce(n.main_nest.begin() + n.num_common_loop + 1,
                                          n.main_nest.end());

    provide = MergeNest(reduce, provide);
    if (debug_keep_trivial_loop) {
      provide = MergeNest(common, provide);
    } else {
      provide = MergeNest(common, SeqStmt::Flatten(init, provide));
    }

    // run substitution in the on the full nest, because loop condition
    // could depend on outer loops.
    return Substitute(provide, n.main_vmap);
  } else {
    std::vector<Stmt> provides;
    for (size_t i = 0; i < self->body.size(); ++i) {
      provides.emplace_back(MakeProvide(stage, self, dom_map, n.main_vmap, stage->op.output(i)));
    }
    Stmt provide = SeqStmt::Flatten(provides);

    provide = MergeNest(n.main_nest, provide);
    // provide = MergeNest(hoist_and_flatten(n.main_nest), provide);

    // run substitution in the on the full nest, because  loop condition
    // could depend on outer loops.
    provide = Substitute(provide, n.main_vmap);
    // if (self->name == "B.shared") {
    // std::cout << "[COP] ProvideStmt\n"  << provide<< std::endl;
    // }
    return provide;
  }
}

enum class ComputeType { kNormal, kCrossThreadReduction, kTensorize };

ComputeType DetectComputeType(const ComputeOpNode* self, const Stage& stage) {
  // Verify correctness of leaf nest.
  int normal_red = 0, thread_red = 0, tensorize = 0;

  for (IterVar iv : stage->leaf_iter_vars) {
    IterVarAttr attr;
    auto it = stage->iter_var_attrs.find(iv);
    if (it != stage->iter_var_attrs.end()) {
      attr = (*it).second;
    }
    if (attr.defined() && attr->iter_type == kTensorized) {
      ++tensorize;
    }
    if (iv->iter_type == kCommReduce) {
      if (attr.defined() && attr->bind_thread.defined()) {
        ++thread_red;
      } else {
        ++normal_red;
      }
    } else {
      CHECK_EQ(thread_red, 0) << "Cross thread reduce cannot swap with normal data axis";
    }
  }
  if (tensorize != 0) {
    CHECK(thread_red == 0) << "Cannot mix cross thread reduction with Tensorize";
    return ComputeType::kTensorize;
  }
  CHECK(normal_red == 0 || thread_red == 0) << "Cannot mix normal reduction with thread reduce";
  if (thread_red != 0) {
    return ComputeType::kCrossThreadReduction;
  } else {
    return ComputeType::kNormal;
  }
}

// implement the provide utility.
Stmt ComputeOpNode::BuildProvide(const Stage& stage,
                                 const std::unordered_map<IterVar, Range>& dom_map,
                                 const std::unordered_map<std::string, Range>& env_dom_map,
                                 const std::unordered_map<std::string, IterVar>& env_var_map,
                                 const std::unordered_map<const VarNode*, std::string>& bind_map,
                                 const Map<Stage, Array<Stage>>& attach_stages,
                                 const Map<Stage, Array<IterVar>>& attach_vars,
                                 bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  ComputeType ctype = DetectComputeType(this, stage);
  if (ctype == ComputeType::kCrossThreadReduction) {
    // specially handle cross thread reduction.
    return MakeCrossThreadReduction(this, stage, dom_map, env_dom_map, env_var_map, bind_map,
                                    attach_stages, attach_vars, debug_keep_trivial_loop);
  } else if (ctype == ComputeType::kTensorize) {
    return MakeTensorize(this, stage, dom_map, env_dom_map, env_var_map, bind_map, attach_stages,
                         attach_vars, debug_keep_trivial_loop);
  } else {
    Stmt ret = MakeComputeStmt(this, stage, dom_map, env_dom_map, env_var_map, bind_map,
                               attach_stages, attach_vars, debug_keep_trivial_loop);
    // if (this->name == "is_h2h.ila") std::cout << "[MP] Provide " << ret << std::endl;
    return ret;
  }
}

ComputeLoopNest ComputeLoopNest::make(
    const BaseComputeOpNode* self, const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    const std::unordered_map<std::string, Range>& env_dom_map,
    const std::unordered_map<std::string, IterVar>& env_var_map,
    const std::unordered_map<const VarNode*, std::string>& bind_map,
    const Map<Stage, Array<Stage>>& attach_stages, const Map<Stage, Array<IterVar>>& attach_vars,
    bool debug_keep_trivial_loop) {
  CHECK_EQ(stage->op.operator->(), self);
  ComputeLoopNest ret;
  // make main loop nest

  ret.main_nest =
      MakeComputeOpLoopNest(stage, dom_map, 0, false, std::unordered_set<IterVar>(), &ret.main_vmap,
                            debug_keep_trivial_loop, self->all_dimensions);

  ret.main_predicates =
      MakeBoundCheck(stage, dom_map, env_dom_map, env_var_map, bind_map, ret.main_vmap, false,
                     std::unordered_set<IterVar>(), attach_stages, attach_vars);

  if (self->name == "QKV.shared") {
    // exit(0);
  }
  for (auto& e : ret.main_predicates) {
    e = likely(e);
  }
  if (stage->store_predicate.defined()) {
    ret.main_predicates.push_back(stage->store_predicate);
  }
  if (self->reduce_axis.size() != 0) {
    // try to find the location to insert the initialization.
    // Fuse the initialization and provide loop when possible.
    std::unordered_map<IterVar, int> update_state;
    for (IterVar iv : self->reduce_axis) {
      update_state[iv] = 2;
    }
    for (size_t i = 0; i < self->num_schedulable_dims(); ++i) {
      update_state[self->axis[i]] = 1;
    }
    // find which iter var is related to reduction and which is related to axis.
    te::PassDownBitMaskOr(stage, &update_state);
    auto leaf_iter_vars = stage->leaf_iter_vars;
    // first first loop that is related to reduction.
    size_t begin_loop = leaf_iter_vars.size();
    for (size_t i = 0; i < leaf_iter_vars.size(); ++i) {
      auto iv = leaf_iter_vars[i];
      int flag = update_state.at(iv);
      if ((flag & 2) != 0) {
        begin_loop = i;
        break;
      }
      ret.init_vmap[iv] = ret.main_vmap.at(iv);
    }
    ret.num_common_loop = begin_loop;
    // skip loops that are related to reduction and are unrelated to axis.
    std::unordered_set<IterVar> skip_iter;
    for (auto kv : update_state) {
      int flag = kv.second;
      if (flag == 2) {
        skip_iter.insert(kv.first);
      }
    }
    ret.init_nest =
        MakeComputeOpLoopNest(stage, dom_map, begin_loop, true, skip_iter, &(ret.init_vmap),
                              debug_keep_trivial_loop, self->all_dimensions);

    // if (self->name == "Asum.repl") {
    //   for (auto it : ret.init_vmap) {
    //     std::cout << "[IVMAP] " << it.first->var << " " << it.second << std::endl;
    //   }
    //   // std::cout << "[BODY] " << static_cast<const ComputeOpNode*>(self)->body << std::endl;
    // }

    ret.init_predicates =
        MakeBoundCheck(stage, dom_map, env_dom_map, env_var_map, bind_map, ret.init_vmap, false,
                       skip_iter, attach_stages, attach_vars);
    for (auto& e : ret.init_predicates) {
      e = likely(e);
    }
  } else {
    CHECK_EQ(ret.main_nest.size(), stage->leaf_iter_vars.size() + 1);
    ret.num_common_loop = stage->leaf_iter_vars.size();
  }
  if (stage->op.as<ComputeOpNode>()) {
    for (const auto& p : static_cast<const ComputeOpNode*>(self)->pred) {
      // std::cout << "[PRED] " << self->name << " " << p << std::endl;
      ret.main_predicates.push_back(p);
      ret.init_predicates.push_back(p);
    }
  }
  // copy elison here.
  return ret;
}

namespace {
/*!
 * \brief Verify if ComputeOp is valid with respect to Reduce operations.
 *
 *  The following two properties are verified:
 *  (1) All Reduce operations must exist at top level.
 *  (2) For a list of operations, if one is Reduce, then the others
 *      must be Reduce as well; and their inputs should have the
 *      same attribute except value_index.
 */
class ComputeVerifier final : protected tir::ExprVisitor {
 public:
  /// Special member functions
  //@{
  explicit ComputeVerifier(const ComputeOpNode* compute)
      : compute_(compute), reduce_(compute->body[0].as<tir::ReduceNode>()) {}
  virtual ~ComputeVerifier() = default;
  ComputeVerifier(const ComputeVerifier&) = delete;
  ComputeVerifier(ComputeVerifier&&) = delete;
  ComputeVerifier& operator=(const ComputeVerifier&) = delete;
  ComputeVerifier& operator=(ComputeVerifier&&) = delete;
  //@}

  /// Interface to perform compute verification
  void Run() {
    for (const PrimExpr e : compute_->body) {
      // Check for consistency of top level reductions
      const tir::ReduceNode* reduce = e.as<tir::ReduceNode>();
      CHECK((reduce && reduce_) || (!reduce && !reduce_)) << "All ComputeOp should be consistent "
                                                          << "with being Reduce operation or not.";

      if (reduce && reduce_) {
        CHECK(ReduceEqual(reduce, reduce_)) << "The Reduce inputs of ComputeOp should "
                                            << "have the same attribute except value_index";
      }

      level_ = 0;
      // std::cout << "[VERIFq] " << compute_->name << " " << e  << std::endl;
      ExprVisitor::VisitExpr(e);
    }
  }

 protected:
  /// Visitor implementation
  //@{
  void VisitExpr(const PrimExpr& n) final {
    ++level_;
    ExprVisitor::VisitExpr(n);
    --level_;
  }

  void VisitExpr_(const tir::ReduceNode* op) final {
    // Check for non top level reductions
    CHECK(0 == level_) << "Reductions are only allowed at the top level of compute. "
                       << "Please create another tensor for further composition. "
                       << compute_->name;
  }
  //@}

 private:
  const ComputeOpNode* compute_{nullptr};   ///< ComputeOpNode to verify
  const tir::ReduceNode* reduce_{nullptr};  ///< Top level Reduce operation
  int level_{0};                            ///< Level of op being processed
};
}  // namespace

/// Verify if ComputeOp is valid with respect to Reduce operations.
static void VerifyComputeOp(const ComputeOpNode* op) {
  ComputeVerifier v(op);
  v.Run();
}

Stmt TransformUpdate(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                     const ComputeLoopNest& n, Stmt body, Stmt update) {
  Array<PrimExpr> conds;
  std::unordered_set<const VarNode*> banned;
  for (size_t i = 0; i < stage->leaf_iter_vars.size(); ++i) {
    IterVar iv = stage->leaf_iter_vars[i];
    auto iit = stage->iter_var_attrs.find(iv);
    if (iit != stage->iter_var_attrs.end()) {
      const IterVarAttr& attr = (*iit).second;
      if (attr->iter_type == kTensorized) {
        break;
      }
    }
    if (iv->iter_type == kCommReduce) {
      auto vit = dom_map.find(iv);
      CHECK(vit != dom_map.end());
      const Range& vrange = vit->second;
      conds.push_back(likely(iv->var > vrange->min));
      banned.insert(iv->var.get());
    }
  }
  for (const PrimExpr& pred : n.main_predicates) {
    if (tir::ExprUseVar(pred, banned)) {
      LOG(FATAL) << "Tensorize update transform failed, the condition " << pred
                 << " has a conflict with the reset condition";
    }
  }

  return IfThenElseNode::make(arith::ComputeReduce<tir::OrNode>(conds, const_true(1)), update,
                              body);
}

TVM_REGISTER_GLOBAL("te.ComputeOpSetOutputBuf")
    .set_body_typed([](Operation op, Buffer buf, Array<Dimension> buf_dims) {
      ComputeOpNode* c_op = const_cast<ComputeOpNode*>(op.as<ComputeOpNode>());
      CHECK(c_op);
      // std::cout << "[COP] Setting buf " << c_op->name << std::endl;
      c_op->output_buffer = buf;
      c_op->output_buffer_dims = buf_dims;
    });

TVM_REGISTER_GLOBAL("te.BaseComputeOpGetRootIndexDimensions")
    .set_body_typed([](Operation op, int value_index) {
      auto c_op = op.as<ComputeOpNode>();
      auto dimensions = c_op->GetRootIndexDimensions(value_index);
      if (dimensions.size() == 0) {
        return NullValue<Array<Dimension>>();
      }
      return dimensions;
    });

}  // namespace te
}  // namespace tvm
