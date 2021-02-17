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
 *  Compile executable modules.
 * \file driver_api.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/driver/driver_api.h>
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>
#include <tvm/te/dimension.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/ta_declarations.h>
#include <tvm/tir/te_capsule.h>

#include <algorithm>
#include <mutex>
#include <stack>

#include "../te/schedule/graph.h"
#include "../tir/ir/var_replacer.h"

namespace tvm {
using namespace tvm::te;
using namespace tvm::tir;

class OpBodyLowerer : public ExprMutator {
  class InputArgumentLowerer : public ExprMutator {
   public:
    InputArgumentLowerer(OpBodyLowerer* p_op_body_lowerer_, TADeclarations declarations_,
                         Map<Var, Var> var_var_mapping_, Map<Var, te::Tensor> var_tensor_mapping_,
                         std::unordered_set<const Object*> tas_tensorize_only_one_dim_,
                         Var orig_loop_var_, Var new_loop_var_)
        : op_body_lowerer(p_op_body_lowerer_),
          declarations(declarations_),
          var_tensor_mapping(var_tensor_mapping_),
          var_var_mapping(var_var_mapping_),
          tas_tensorize_only_one_dim(tas_tensorize_only_one_dim_) {
      rmap[orig_loop_var_.get()] = new_loop_var_;
    }

    bool is_orig_callee(const PrimExprNode* expr) { return this->orig_call_argument == expr; }

    PrimExpr lower_input_argument(PrimExpr argument, const CallNode* orig_call) {
      // std::cout << "[TE]    Lowering input argument " << argument << std::endl;
      this->orig_call_argument = (orig_call != nullptr) ? argument.get() : nullptr;
      this->orig_call = orig_call;
      return this->VisitExpr(argument);
    }

   private:
    PrimExpr VisitExpr_(const VarNode* var) override {
      if (is_orig_callee(var)) {
        CHECK(var->dtype.is_handle());
        if (var_tensor_mapping.count(GetRef<Var>(var))) {
          // This means we have a new tensor for this variable
          VarReplacer var_replacer(rmap);
          te::Tensor tensor = var_tensor_mapping.at(GetRef<Var>(var));
          Array<PrimExpr> args;
          for (auto arg : orig_call->args) {
            args.push_back(op_body_lowerer->VisitExpr(arg));
          }
          PrimExpr ret = var_replacer(
              CallNode::make(tensor->dtype, tensor->op->name, args, CallNode::Halide,
                             orig_call->argument_dimensions, tensor->op, tensor->value_index));
          // std::cout << "[TE]    Call in argument " << GetRef<PrimExpr>(var) << " " << ret << " "
          //           << ret->dtype << std::endl;
          return ret;
        } else {
          // We're to use the same old tensor for this as there was no
          // need to create a new one.
          return op_body_lowerer->VisitExpr_(orig_call);
        }
      } else {
        if (var_var_mapping.count(GetRef<Var>(var))) {
          return var_var_mapping.at(GetRef<Var>(var));
        } else {
          return GetRef<Var>(var);
        }
      }
    }

    PrimExpr VisitExpr_(const LoadNode* load) override {
      Var buffer_var = load->buffer_var;
      CHECK(var_tensor_mapping.count(buffer_var)) << buffer_var;
      te::Tensor tensor = var_tensor_mapping.at(buffer_var);
      Array<PrimExpr> args;
      args.push_back(ExprFunctor::VisitExpr(load->index));
      if (is_orig_callee(load)) {
        for (auto arg : orig_call->args) {
          args.push_back(op_body_lowerer->VisitExpr(arg));
        }
      }
      Array<Dimension> arg_dims;
      if (auto pl_op = tensor->op.as<PlaceholderOpNode>()) {
        arg_dims.push_back(pl_op->self_index_dimensions[0]);
      } else if (auto bvd_op = tensor->op.as<BaseVarDimOpNode>()) {
        arg_dims.push_back(bvd_op->GetBaseIndexDimension(tensor->value_index, 0));
      }
      if (is_orig_callee(load)) {
        arg_dims.push_back_all(orig_call->argument_dimensions);
      }
      VarReplacer var_replacer(rmap);
      PrimExpr ret =
          var_replacer(CallNode::make(tensor->dtype, tensor->op->name, args, CallNode::Halide,
                                      arg_dims, tensor->op, tensor->value_index));
      // std::cout << "[TE]    Call in argument " << GetRef<PrimExpr>(load) << " " << ret << " "
      // << ret->dtype << std::endl;
      return ret;
    }

    PrimExpr VisitExpr_(const RegionTALoadNode* load) override {
      Var ta_var = load->region_ta;
      CHECK(var_tensor_mapping.count(ta_var)) << ta_var << " " << ta_var.get();
      te::Tensor tensor = var_tensor_mapping.at(ta_var);
      Array<PrimExpr> args;
      // bool tensorize_one_dim = tas_tensorize_only_one_dim.count(
      // declarations.get_tensor_array(ta_var)->GetBaseTensorArray().get());
      bool tensorize_one_dim =
          tas_tensorize_only_one_dim.count(declarations.get_tensor_array(ta_var).get());
      if (tensorize_one_dim) {
        args.push_back(ExprFunctor::VisitExpr(load->indices[load->indices.size() - 1]));
      } else {
        for (auto index : load->indices) {
          args.push_back(ExprFunctor::VisitExpr(index));
        }
      }
      if (is_orig_callee(load)) {
        for (auto arg : orig_call->args) {
          args.push_back(op_body_lowerer->VisitExpr(arg));
        }
      }
      Array<Dimension> arg_dims;
      if (auto pl_op = tensor->op.as<PlaceholderOpNode>()) {
        if (tensorize_one_dim) {
          arg_dims.push_back(pl_op->self_index_dimensions[load->indices.size() - 1]);
        } else {
          for (size_t i = 0; i < load->indices.size(); ++i) {
            arg_dims.push_back(pl_op->self_index_dimensions[i]);
          }
        }
      } else if (auto bvd_op = tensor->op.as<BaseVarDimOpNode>()) {
        if (tensorize_one_dim) {
          arg_dims.push_back(
              bvd_op->GetBaseIndexDimension(tensor->value_index, load->indices.size() - 1));
        } else {
          for (size_t i = 0; i < load->indices.size(); ++i) {
            arg_dims.push_back(bvd_op->GetBaseIndexDimension(tensor->value_index, i));
          }
        }
      }
      if (is_orig_callee(load)) {
        arg_dims.push_back_all(orig_call->argument_dimensions);
      }
      VarReplacer var_replacer(rmap);
      PrimExpr ret =
          var_replacer(CallNode::make(tensor->dtype, tensor->op->name, args, CallNode::Halide,
                                      arg_dims, tensor->op, tensor->value_index));
      // std::cout << "[TE]    Call in argument " << GetRef<PrimExpr>(load) << " " << ret << " "
      // << ret->dtype << std::endl;
      return ret;
    }

    OpBodyLowerer* op_body_lowerer;
    TADeclarations declarations;
    Map<Var, te::Tensor> var_tensor_mapping;
    Map<Var, Var> var_var_mapping;
    std::unordered_set<const Object*> tas_tensorize_only_one_dim;
    std::unordered_map<const VarNode*, PrimExpr> rmap;
    const Object* orig_call_argument;
    const CallNode* orig_call;
  };

 public:
  OpBodyLowerer(Map<Var, PrimExpr> input_vars_, Map<te::Operation, PrimExpr> input_arguments_,
                TADeclarations declarations_, Map<Var, Var> var_var_mapping_,
                Map<Var, te::Tensor> var_tensor_mapping_,
                std::unordered_set<const Object*> tas_tensorize_only_one_dim_,
                Map<te::Operation, te::Operation>* p_new_op_mapping_, Var orig_loop_var_,
                Var new_loop_var_, Dimension new_loop_dim_)
      : input_vars(input_vars_),
        input_arguments(input_arguments_),
        declarations(declarations_),
        var_var_mapping(var_var_mapping_),
        var_tensor_mapping(var_tensor_mapping_),
        tas_tensorize_only_one_dim(tas_tensorize_only_one_dim_),
        new_op_mapping(*p_new_op_mapping_),
        new_loop_var(new_loop_var_),
        orig_loop_var(orig_loop_var_),
        new_loop_dim(new_loop_dim_) {
    InputArgumentLowerer input_lowerer(this, declarations, var_var_mapping, var_tensor_mapping,
                                       tas_tensorize_only_one_dim, orig_loop_var, new_loop_var);
    for (auto it : input_vars) {
      rmap_[it.first.as<VarNode>()] = input_lowerer.lower_input_argument(it.second, nullptr);
    }
  }

  PrimExpr lower_body(PrimExpr body) {
    // std::cout << "[TE]   Lowering body " << body << std::endl;
    PrimExpr expr = this->VisitExpr(body);
    VarReplacer replacer(rmap_);
    return replacer(expr);
  }

  PrimExpr VisitExpr_(const LoadNode* load) override {
    // std::cout << "[TE]   Lowering load " << GetRef<PrimExpr>(load) << std::endl;
    Var buffer_var = load->buffer_var;
    if (var_tensor_mapping.count(buffer_var)) {
      Tensor tensor = var_tensor_mapping.at(buffer_var);
      Array<Dimension> arg_dims;
      if (auto pl_op = tensor->op.as<PlaceholderOpNode>()) {
        arg_dims = pl_op->self_index_dimensions;
      } else if (auto bvd_op = tensor->op.as<BaseVarDimOpNode>()) {
        arg_dims = bvd_op->GetRootIndexDimensions(tensor->value_index);
      }
      CHECK_EQ(arg_dims.size(), 1);
      return CallNode::make(load->dtype, tensor->op->name, {ExprMutator::VisitExpr(load->index)},
                            CallNode::Halide, arg_dims, tensor->op, tensor->value_index);
    }
    return ExprMutator::VisitExpr_(load);
  }

  PrimExpr VisitExpr_(const CallNode* call) override {
    // std::cout << "[TE]  Call " << GetRef<PrimExpr>(call) << " " << call->func << std::endl;
    if (call->call_type == CallNode::Halide && call->func.defined() &&
        call->func.as<OperationNode>()) {
      auto op = Downcast<Operation>(call->func);
      if (auto pl_op = op.as<PlaceholderOpNode>()) {
        if (input_arguments.count(op)) {
          PrimExpr argument = input_arguments.at(op);
          InputArgumentLowerer input_arg_lowerer(this, declarations, var_var_mapping,
                                                 var_tensor_mapping, tas_tensorize_only_one_dim,
                                                 orig_loop_var, new_loop_var);
          auto ret = input_arg_lowerer.lower_input_argument(argument, call);
          // std::cout << "[TE]   Lowering argument " << argument << " " << ret << std::endl;
          // std::cout << "[TE]   Lowering argument " << ret << std::endl;
          return ret;
        } else {
          // std::cout << "[TE]   Skipping " << op << " " << call->value_index << std::endl;
          return ExprMutator::VisitExpr_(call);
        }
      } else if (new_op_mapping.count(op)) {
        auto new_op = new_op_mapping.at(op);
        Array<PrimExpr> args;
        args.push_back(new_loop_var);
        for (auto arg : call->args) {
          args.push_back(arg);
        }
        Array<Dimension> arg_dims;
        arg_dims.push_back(new_loop_dim);
        arg_dims.push_back_all(call->argument_dimensions);
        return CallNode::make(call->dtype, new_op->name, args, CallNode::Halide, arg_dims, new_op,
                              call->value_index);
      }
    }
    return ExprMutator::VisitExpr_(call);
  }

  PrimExpr VisitExpr_(const VarNode* var) override {
    // std::cout << "[TE]   Var argument " << var->name_hint << std::endl;
    if (var_var_mapping.count(GetRef<Var>(var))) {
      return var_var_mapping.at(GetRef<Var>(var));
    } else {
      return GetRef<Var>(var);
    }
  }

  Map<Var, PrimExpr> input_vars;
  Map<te::Operation, PrimExpr> input_arguments;
  TADeclarations declarations;
  Map<Var, Var> var_var_mapping;
  Map<Var, te::Tensor> var_tensor_mapping;
  std::unordered_set<const Object*> tas_tensorize_only_one_dim;
  Map<te::Operation, te::Operation> new_op_mapping;
  Var new_loop_var;
  Var orig_loop_var;
  Dimension new_loop_dim;
  std::unordered_map<const VarNode*, PrimExpr> rmap_;
};

te::Tensor MakeTensor(std::string name, Array<PrimExpr> orig_shape, Array<PrimExpr> tensor_shape,
                      DataType dtype, te::Tensor original, Array<PrimExpr> index_exprs,
                      Var orig_loop_var, Range loop_range, Dimension new_loop_dim,
                      bool only_one_dim = false) {
  Array<PrimExpr> shape;
  Array<Dimension> self_index_dimensions;
  Array<Dimension> dimensions;
  Array<tir::IterVar> itervars;
  Array<tir::UninterpFun> uninterpfuns;

  for (size_t i = (only_one_dim ? index_exprs.size() - 1 : 0); i < index_exprs.size(); ++i) {
    auto index = index_exprs[i];
    if (VarFinder::ContainsVariable(index, orig_loop_var)) {
      CHECK(!self_index_dimensions.Contains(new_loop_dim));
      self_index_dimensions.push_back(new_loop_dim);
      dimensions.push_back(new_loop_dim);
      itervars.push_back(
          IterVarNode::make(loop_range, orig_loop_var.copy_with_suffix("iv"), kDataPar, ""));
    } else {
      Dimension dim = DimensionNode::make("dim" + std::to_string(i), DimensionNode::kRangeDim);
      self_index_dimensions.push_back(dim);
      dimensions.push_back(dim);
      itervars.push_back(IterVarNode::make(Range::make_by_min_extent(0, orig_shape[i]),
                                           Var("var" + std::to_string(i)), kDataPar, ""));
    }
    shape.push_back(orig_shape[i]);
    uninterpfuns.push_back(NullValue<UninterpFun>());
  }

  shape.push_back_all(tensor_shape);

  if (original.defined()) {
    auto p_op = original->op.as<PlaceholderOpNode>();
    self_index_dimensions.push_back_all(p_op->self_index_dimensions);

    for (size_t i = 0; i < p_op->all_dimensions.size(); ++i) {
      dimensions.push_back(p_op->all_dimensions[i]->dim);
      itervars.push_back(p_op->all_dimensions[i]->iv);
      uninterpfuns.push_back(p_op->all_dimensions[i]->ufun);
    }
  }

  Operation op = PlaceholderOpNode::make(name, shape, dtype, self_index_dimensions, dimensions,
                                         itervars, uninterpfuns);
  return op.output(original.defined() ? original->value_index : 0);
}

te::Tensor MakeTensor(const RegionTensorArrayNode* rta, te::Tensor original,
                      Array<PrimExpr> index_exprs, Var orig_loop_var, Range loop_range,
                      Dimension new_loop_dim, bool only_one_dim = false) {
  std::string name = rta->name;
  if (original.defined()) {
    name = original->op->name + "_nt";
  }
  return MakeTensor(name, rta->shape, rta->tensor_shape, rta->dtype, original, index_exprs,
                    orig_loop_var, loop_range, new_loop_dim, only_one_dim);
}

te::Tensor MakeTensor(Buffer buf, te::Tensor original, PrimExpr index_expr, Var orig_loop_var,
                      Range loop_range, Dimension new_loop_dim, bool only_one_dim = false) {
  return MakeTensor(buf->name, buf->shape, {}, buf->dtype, original, {index_expr}, orig_loop_var,
                    loop_range, new_loop_dim, only_one_dim);
}

class ProcessInputArgument : public ExprVisitor {
 public:
  ProcessInputArgument(TADeclarations declarations_, const ForNode* loop_, Dimension new_loop_dim_,
                       std::unordered_set<const Object*> tas_tensorize_only_one_dim_,
                       Map<tir::Var, tir::Var>* p_var_var_mapping_,
                       Map<tir::Var, te::Tensor>* p_var_tensor_mapping_,
                       Map<PrimExpr, Var>* p_new_input_var_tensor_mapping_,
                       Map<PrimExpr, te::Tensor>* p_new_input_tensor_mapping_,
                       Map<TensorArray, TensorArray>* p_new_reshaped_tensor_arrays_,
                       Map<TensorArray, te::Tensor>* p_old_ta_new_tensor_mapping_,
                       std::unordered_map<te::Tensor, te::Tensor>* p_old_tensor_new_tensor_mapping_)
      : declarations(declarations_),
        loop(loop_),
        new_loop_dim(new_loop_dim_),
        tas_tensorize_only_one_dim(tas_tensorize_only_one_dim_),
        var_var_mapping(*p_var_var_mapping_),
        var_tensor_mapping(*p_var_tensor_mapping_),
        new_input_var_mapping(*p_new_input_var_tensor_mapping_),
        new_input_tensor_mapping(*p_new_input_tensor_mapping_),
        new_reshaped_tensor_arrays(*p_new_reshaped_tensor_arrays_),
        old_ta_new_tensor_mapping(*p_old_ta_new_tensor_mapping_),
        old_tensor_new_tensor_mapping(*p_old_tensor_new_tensor_mapping_) {}

  void process_dependent_argument(PrimExpr input, te::Tensor original_tensor) {
    if (auto load = input.as<RegionTALoadNode>()) {
      // std::cout << "[TE]  RegionTALoad " << input << " " << original_tensor << std::endl;
      auto ta = declarations.get_tensor_array(load->region_ta);
      auto tensor = MakeTensor(ta.as<RegionTensorArrayNode>(), original_tensor, load->indices,
                               loop->loop_var, Range::make_by_min_extent(loop->min, loop->extent),
                               new_loop_dim, tas_tensorize_only_one_dim.count(ta.get()));
      // std::cout << "[TE]   Made tensor for inputs " << ta << " " << tensor << std::endl;
      var_tensor_mapping.Set(load->region_ta, tensor);
      old_ta_new_tensor_mapping.Set(ta, tensor);
      if (original_tensor.defined()) {
        old_tensor_new_tensor_mapping[original_tensor] = tensor;
      }

      // Create a new reshaped tensor array
      TensorArray reshaped;
      {
        auto rta = ta.as<RegionTensorArrayNode>();
        Array<PrimExpr> new_tensor_shape;
        Array<PrimExpr> new_tensor_array_shape;
        if (tas_tensorize_only_one_dim.count(ta.get())) {
          for (size_t i = 0; i < rta->shape.size() - 1; ++i) {
            new_tensor_array_shape.push_back(rta->shape[i]);
          }
          new_tensor_shape.push_back(rta->shape[rta->shape.size() - 1]);
        } else {
          new_tensor_shape.push_back_all(rta->shape);
        }
        new_tensor_shape.push_back_all(rta->tensor_shape);

        reshaped = RegionTensorArrayNode::make(rta->ta_var.copy_with_suffix("_reshaped"),
                                               rta->dtype, new_tensor_array_shape, new_tensor_shape,
                                               rta->name + "_reshaped", rta->base_region_ta);
        new_reshaped_tensor_arrays.Set(ta, reshaped);
        declarations.add_tensor_array(reshaped);
      }

      Array<PrimExpr> new_indices;
      if (tas_tensorize_only_one_dim.count(ta.get())) {
        for (size_t i = 0; i < load->indices.size() - 1; ++i) {
          new_indices.push_back(load->indices[i]);
        }
      }

      new_input_tensor_mapping.Set(
          RegionTALoadNode::make(reshaped->ta_var, new_indices, load->dtype), tensor);

      if (tas_tensorize_only_one_dim.count(ta.get())) {
        this->VisitExpr(load->indices[load->indices.size() - 1]);
      } else {
        for (auto index : load->indices) {
          this->VisitExpr(index);
        }
      }
    } else if (auto load = input.as<PointerTALoadNode>()) {
    } else if (auto varnode = input.as<VarNode>()) {
      if (original_tensor.defined()) {
        // std::cout << "[TE]  BufferVarArg " << input << std::endl;
        new_input_tensor_mapping.Set(input, original_tensor);
      } else {
        // std::cout << "[TE]  VarArg " << input << std::endl;
      }
    } else if (auto load = input.as<LoadNode>()) {
      // std::cout << "[TE]  BufferLoad " << input << std::endl;
      auto buf = declarations.get_buffer(load->buffer_var);
      auto tensor = MakeTensor(buf, {}, load->index, loop->loop_var,
                               Range::make_by_min_extent(loop->min, loop->extent), new_loop_dim);
      // std::cout << "[TE]   Made tensor for inputs " << buf << " " << tensor << std::endl;
      var_tensor_mapping.Set(load->buffer_var, tensor);
      new_input_tensor_mapping.Set(load->buffer_var, tensor);
      this->VisitExpr(load->index);
    } else {
      ExprVisitor::VisitExpr(input);
      // CHECK(false) << input;
    }
  }

  void process_argument(PrimExpr input, te::Tensor input_tensor) {
    // std::cout << "[TE] Input " << input << std::endl;
    process_dependent_argument(input, input_tensor);
  }

  void process_argument(PrimExpr input, Var input_var) {
    // std::cout << "[TE] Input " << input << std::endl;
    process_dependent_argument(input, {});
  }

  void VisitExpr_(const RegionTALoadNode* load) override {
    // std::cout << "[TE]  RegionTALoad " << GetRef<PrimExpr>(load) << std::endl;
    auto ta = declarations.get_tensor_array(load->region_ta);
    auto tensor = MakeTensor(ta.as<RegionTensorArrayNode>(), {}, load->indices, loop->loop_var,
                             Range::make_by_min_extent(loop->min, loop->extent), new_loop_dim,
                             tas_tensorize_only_one_dim.count(ta.get()));
    // std::cout << "[TE]   Made tensor for inputs " << ta << " " << tensor << " "
    // << load->region_ta.get() << std::endl;
    var_tensor_mapping.Set(load->region_ta, tensor);
    new_input_tensor_mapping.Set(load->region_ta, tensor);
  }

  void VisitExpr_(const VarNode* varnode) override {
    Var var = GetRef<Var>(varnode);
    if (!new_input_var_mapping.count(var) && !var.same_as(loop->loop_var)) {
      Var param_var = var.copy_with_suffix("_p");
      new_input_var_mapping.Set(var, param_var);
      var_var_mapping.Set(var, param_var);
    }
  }

  void VisitExpr_(const LoadNode* load) override {
    // std::cout << "[TE]  Load " << GetRef<PrimExpr>(load) << std::endl;
    auto buf = declarations.get_buffer(load->buffer_var);
    auto tensor = MakeTensor(buf, {}, load->index, loop->loop_var,
                             Range::make_by_min_extent(loop->min, loop->extent), new_loop_dim);
    // std::cout << "[TE]   Made tensor for inputs " << buf << " " << tensor << std::endl;
    var_tensor_mapping.Set(load->buffer_var, tensor);
    new_input_tensor_mapping.Set(load->buffer_var, tensor);
  }

  TADeclarations declarations;
  const ForNode* loop;
  Dimension new_loop_dim;
  std::unordered_set<const Object*> tas_tensorize_only_one_dim;
  Map<tir::Var, tir::Var>& var_var_mapping;
  Map<tir::Var, te::Tensor>& var_tensor_mapping;
  Map<PrimExpr, Var>& new_input_var_mapping;
  Map<PrimExpr, te::Tensor>& new_input_tensor_mapping;
  Map<TensorArray, TensorArray>& new_reshaped_tensor_arrays;
  Map<TensorArray, te::Tensor>& old_ta_new_tensor_mapping;
  std::unordered_map<te::Tensor, te::Tensor>& old_tensor_new_tensor_mapping;
};

Stmt LiftLoopToComputeOp(TADeclarations declarations, const ForNode* loop,
                         Map<TensorArray, te::Tensor> scan_io_mapping) {
  auto store = loop->body.as<RegionTAStoreNode>();
  auto loop_var = loop->loop_var;
  auto capsule = TECapsule::capsules.at(store->te_graph_name);

  Dimension new_loop_dim =
      DimensionNode::make(loop_var->name_hint + "_dim", DimensionNode::kRangeDim);

  Map<tir::Var, tir::Var> var_var_mapping;
  Map<tir::Var, te::Tensor> var_tensor_mapping;
  Map<PrimExpr, te::Tensor> new_input_tensor_mapping;
  Map<PrimExpr, Var> new_input_var_mapping;
  Map<TensorArray, TensorArray> new_reshaped_tensor_arrays;
  std::unordered_set<const Object*> tas_tensorize_only_one_dim;
  Map<TensorArray, te::Tensor> old_ta_new_tensor_mapping;
  std::unordered_map<te::Tensor, te::Tensor> old_tensor_new_tensor_mapping;
  for (auto it : scan_io_mapping) {
    // std::cout << "[TE] ScanIO " << it.first << " " << it.second << std::endl;
    tas_tensorize_only_one_dim.insert(it.first.get());
  }
  ProcessInputArgument arg_processor(declarations, loop, new_loop_dim, tas_tensorize_only_one_dim,
                                     &var_var_mapping, &var_tensor_mapping, &new_input_var_mapping,
                                     &new_input_tensor_mapping, &new_reshaped_tensor_arrays,
                                     &old_ta_new_tensor_mapping, &old_tensor_new_tensor_mapping);
  for (size_t i = 0; i < capsule->input_vars.size(); ++i) {
    arg_processor.process_argument(store->inputs[i], capsule->input_vars[i]);
  }
  for (size_t i = 0; i < capsule->inputs.size(); ++i) {
    arg_processor.process_argument(store->inputs[i + capsule->input_vars.size()],
                                   capsule->inputs[i]);
  }

  Array<te::Tensor> all_inputs(capsule->inputs);
  all_inputs.push_back_all(capsule->non_external_inputs);
  Array<te::Operation> operations = GetSubGraph(capsule->outputs, all_inputs, false);
  Map<Var, PrimExpr> input_var_mapping;
  for (size_t i = 0; i < capsule->input_vars.size(); ++i) {
    input_var_mapping.Set(capsule->input_vars[i], store->inputs[i]);
  }
  Map<te::Operation, PrimExpr> input_argument_mapping;
  for (size_t i = 0; i < capsule->inputs.size(); ++i) {
    input_argument_mapping.Set(capsule->inputs[i]->op,
                               store->inputs[i + capsule->input_vars.size()]);
  }

  Array<te::Tensor> non_external_input_tensors;
  Map<te::Operation, te::Operation> new_op_mapping;
  for (auto op : operations) {
    if (auto pl_op = op.as<te::PlaceholderOpNode>()) {
    } else if (auto c_op = op.as<te::ComputeOpNode>()) {
      IterVar new_loop_iv =
          IterVarNode::make(Range::make_by_min_extent(loop->min, loop->extent),
                            loop_var.copy_with_suffix("_liv_" + c_op->name), kDataPar, "");
      Map<tir::Var, tir::Var> updated_var_var_mapping(var_var_mapping);
      updated_var_var_mapping.Set(loop_var, new_loop_iv->var);
      OpBodyLowerer body_lowerer(input_var_mapping, input_argument_mapping, declarations,
                                 var_var_mapping, var_tensor_mapping, tas_tensorize_only_one_dim,
                                 &new_op_mapping, loop_var, new_loop_iv->var, new_loop_dim);
      Array<PrimExpr> new_body_exprs;

      for (auto body_expr : c_op->body) {
        PrimExpr new_body = body_lowerer.lower_body(body_expr);
        new_body_exprs.push_back(new_body);
        // std::cout << "[TE] New body " << new_body << std::endl;
      }
      Array<PrimExpr> new_pred_exprs;
      for (auto pred_expr : c_op->pred) {
        PrimExpr new_pred = body_lowerer.lower_body(pred_expr);
        new_pred_exprs.push_back(new_pred);
      }

      Array<Dimension> root_index_dimensions;
      Array<PrimExpr> output_shape_storage;
      Array<IterVar> itervars;
      Array<Dimension> dimensions;
      Array<UninterpFun> uninterpfuns;
      {
        root_index_dimensions.push_back(new_loop_dim);
        root_index_dimensions.push_back_all(c_op->GetRootIndexDimensions(0));

        output_shape_storage.push_back(loop->extent);
        output_shape_storage.push_back_all(c_op->output_shape_storage);

        dimensions.push_back(new_loop_dim);
        itervars.push_back(new_loop_iv);
        uninterpfuns.push_back(NullValue<UninterpFun>());

        for (auto dim_info : c_op->all_dimensions) {
          if (dim_info->dim->isRangeDim()) {
            dimensions.push_back(dim_info->dim);
            itervars.push_back(dim_info->iv);
            uninterpfuns.push_back(dim_info->ufun);
          }
        }
      }

      Operation new_op = ComputeOpNode::make(
          c_op->name + "_te", c_op->tag, c_op->attrs, itervars, root_index_dimensions,
          output_shape_storage, itervars, dimensions, uninterpfuns, new_body_exprs, new_pred_exprs);
      new_op_mapping.Set(op, new_op);
    } else if (auto s_op = op.as<te::ScanOpNode>()) {
      Array<te::Tensor> state_placeholder;
      Array<te::Tensor> init;
      Array<te::Tensor> update;

      // std::cout << "[SCAN]   Old state " << s_op->state_placeholder[0] << std::endl;

      auto remap_tensor = [&](const te::Tensor& tensor) {
        if (new_op_mapping.count(tensor->op)) {
          return new_op_mapping.at(tensor->op).output(tensor->value_index);
        } else {
          return tensor;
        }
      };

      CHECK_EQ(op->num_outputs(), 1);
      auto new_update = remap_tensor(s_op->update[0]);
      auto bvd_op = new_update->op.as<BaseVarDimOpNode>();
      CHECK(bvd_op);

      for (auto it : old_tensor_new_tensor_mapping) {
      }
      CHECK(old_tensor_new_tensor_mapping.count(s_op->state_placeholder[0]));
      auto new_state = old_tensor_new_tensor_mapping.at(s_op->state_placeholder[0]);
      auto new_init =
          PlaceholderOpNode::make(s_op->name + "_init", new_update->shape, new_update->dtype,
                                  bvd_op->GetRootIndexDimensions(0), bvd_op->GetAllDimensions())
              .output(0);
      state_placeholder.push_back(new_state);
      init.push_back(new_init);
      non_external_input_tensors.push_back(new_state);
      non_external_input_tensors.push_back(new_init);
      update.push_back(new_update);

      Array<Dimension> explicit_loops;
      Array<UninterpFun> explicit_min_ufs;
      Array<UninterpFun> explicit_max_ufs;

      {
        for (auto di : s_op->explicit_dimensions) {
          explicit_loops.push_back(di->dim);
          explicit_min_ufs.push_back(UninterpFunNode::from_constant("min", di->iv->dom->min));
          explicit_max_ufs.push_back(
              UninterpFunNode::from_constant("max", di->iv->dom->min + di->iv->dom->extent));
        }
        explicit_loops.push_back(new_loop_dim);
        explicit_min_ufs.push_back(UninterpFunNode::from_constant("min", loop->min));
        explicit_max_ufs.push_back(UninterpFunNode::from_constant("max", loop->min + loop->extent));
      }

      Operation scan = ScanOpNode::make(
          s_op->name, "", {}, UninterpFunNode::from_constant("min", s_op->scan_axis->dom->min),
          UninterpFunNode::from_constant("max",
                                         s_op->scan_axis->dom->min + s_op->scan_axis->dom->extent),
          s_op->scan_dim, false, init, update, state_placeholder, s_op->inputs, explicit_loops,
          explicit_min_ufs, explicit_max_ufs);
      new_op_mapping.Set(op, scan);
      // std::cout << "[SCAN]   New scan op " << op << " " << scan << std::endl;
    } else {
      // CHECK(false) << "Lifting " << op << " not yet supported";
    }
  }

  // New TECapsule fields
  Array<te::Tensor> output_tensors;
  if (loop->for_type == ForType::Sequential) {
    Array<te::Tensor> state_placeholder;
    Array<te::Tensor> init;
    Array<te::Tensor> update;

    CHECK_EQ(capsule->outputs.size(), 1);
    CHECK_EQ(scan_io_mapping.size(), 1);
    CHECK(new_op_mapping.count(capsule->outputs[0]->op)) << capsule->outputs[0]->op;
    for (auto it : scan_io_mapping) {
      CHECK(old_ta_new_tensor_mapping.count(it.first)) << it.first;
      auto new_state_op = old_ta_new_tensor_mapping.at(it.first)->op;
      auto new_state = new_state_op.output(it.second->value_index);
      auto pl_op = new_state_op.as<PlaceholderOpNode>();
      state_placeholder.push_back(new_state);
      auto new_init = PlaceholderOpNode::make("init", new_state->shape, new_state->dtype,
                                              pl_op->self_index_dimensions, pl_op->all_dimensions)
                          .output(0);
      init.push_back(new_init);
      non_external_input_tensors.push_back(new_init);
    }
    update.push_back(
        new_op_mapping.at(capsule->outputs[0]->op).output(capsule->outputs[0]->value_index));

    std::cout << "[SCAN] " << state_placeholder[0] << " " << update[0] << std::endl;

    Operation scan = ScanOpNode::make(
        loop_var->name_hint + "_scan", "", {}, UninterpFunNode::from_constant("min", loop->min),
        UninterpFunNode::from_constant("max", loop->min + loop->extent), new_loop_dim, false, init,
        update, state_placeholder, {}, {}, {}, {});

    CHECK_EQ(capsule->outputs.size(), 1);
    output_tensors.push_back(scan.output(0));
  } else {
    for (auto tensor : capsule->outputs) {
      output_tensors.push_back(new_op_mapping.at(tensor->op).output(tensor->value_index));
    }
  }

  Array<te::Tensor> input_tensors;
  Array<Var> input_vars;
  // New RegionTAStore fields
  Array<PrimExpr> op_inputs;
  Array<Array<PrimExpr>> region_ta_indices;
  {
    for (auto it : new_input_var_mapping) {
      input_vars.push_back(it.second);
      op_inputs.push_back(it.first);
    }

    for (auto it : new_input_tensor_mapping) {
      input_tensors.push_back(it.second);
      op_inputs.push_back(it.first);
    }

    for (auto indices : store->region_ta_indices) {
      CHECK(indices[indices.size() - 1].same_as(loop_var));
      Array<PrimExpr> new_indices;
      for (size_t i = 0; i < indices.size() - 1; ++i) {
        new_indices.push_back(indices[i]);
      }
      region_ta_indices.push_back(new_indices);
    }
  }

  // Mutate the TECapsule
  {
    TECapsuleNode* mut_capsule = const_cast<TECapsuleNode*>(capsule);

    // std::cout << "[TE] OUTPUT " << output_tensors << std::endl;

    mut_capsule->input_vars = input_vars;
    mut_capsule->inputs = input_tensors;
    mut_capsule->non_external_inputs = non_external_input_tensors;
    mut_capsule->outputs = output_tensors;
    mut_capsule->all_ops_ = {};
  }

  Array<Var> store_tas;
  {
    for (auto var : store->region_tas) {
      auto old_ta = declarations.get_tensor_array(var);
      if (new_reshaped_tensor_arrays.count(old_ta)) {
        store_tas.push_back(new_reshaped_tensor_arrays.at(old_ta)->ta_var);
      } else {
        auto old_rta = old_ta.as<RegionTensorArrayNode>();
        Array<PrimExpr> new_ta_shape;
        for (size_t i = 0; i < old_ta->shape.size() - 1; ++i) {
          new_ta_shape.push_back(old_ta->shape[i]);
        }
        Array<PrimExpr> new_tensor_shape;
        new_tensor_shape.push_back(old_ta->shape[old_rta->shape.size() - 1]);
        new_tensor_shape.push_back_all(old_rta->tensor_shape);

        TensorArray new_ta = RegionTensorArrayNode::make(
            old_rta->ta_var.copy_with_suffix("_reshaped"), old_rta->dtype, new_ta_shape,
            new_tensor_shape, old_rta->name + "_reshaped", old_rta->base_region_ta);
        declarations.add_tensor_array(new_ta);
        store_tas.push_back(new_ta->ta_var);
        new_reshaped_tensor_arrays.Set(old_ta, new_ta);
      }
    }
  }

  Stmt new_store = RegionTAStoreNode::make(store_tas, region_ta_indices, capsule->name, op_inputs,
                                           store->direct_inputs);

  // Create reshape statements now
  Array<Stmt> stmts;
  for (auto it : new_reshaped_tensor_arrays) {
    stmts.push_back(ReshapeTANode::make(it.second->ta_var, it.first->ta_var));
  }
  stmts.push_back(new_store);
  return SeqStmt(stmts);
}

Stmt LiftLoopToScanOp(TADeclarations declarations, const ForNode* loop,
                      Map<TensorArray, te::Tensor> scan_io_mapping) {
  auto stmt = LiftLoopToComputeOp(declarations, loop, scan_io_mapping);
  return stmt;
}

class ReshapeHoister : public StmtMutator {
 public:
  Stmt hoist(Stmt stmt) {
    Stmt body = this->VisitStmt(stmt);
    reshapes.push_back(body);
    return SeqStmt(reshapes);
  }

 private:
  Stmt VisitStmt_(const ReshapeTANode* op) override {
    reshapes.push_back(GetRef<Stmt>(op));
    return EvaluateNode::make(0);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) override {
    Array<Stmt> stmts;
    for (auto stmt : op->seq) {
      stmt = this->VisitStmt(stmt);
      if (auto evaluate = stmt.as<EvaluateNode>()) {
        if (evaluate->value.as<IntImmNode>()) continue;
      }
      stmts.push_back(stmt);
    }
    if (stmts.size() == 1) {
      return stmts[0];
    } else {
      return SeqStmt(stmts);
    }
  }

  Array<Stmt> reshapes;
};

class LoopFinderAndLowerer : public tir::StmtExprMutator {
 public:
  LoopFinderAndLowerer(TADeclarations declarations_) : declarations(declarations_) {}

  bool CanBeLiftedToComputeOp(const ForNode* loop) {
    if (auto store = loop->body.as<RegionTAStoreNode>()) {
      Var loop_var = loop->loop_var;
      bool last_index_is_loop_var = true;
      for (auto indices : store->region_ta_indices) {
        if (!indices[indices.size() - 1].same_as(loop_var)) {
          last_index_is_loop_var = false;
        }
      }
      return last_index_is_loop_var && (store->direct_inputs.size() == 0) &&
             (loop->for_type == ForType::Parallelizable);
    }
    return false;
  }

  bool CanBeLiftedToScanOp(const ForNode* loop) {
    if (auto store = loop->body.as<RegionTAStoreNode>()) {
      // std::cout << "[TE]  Body of loop " << loop->body->GetTypeKey() << std::endl;
      Var loop_var = loop->loop_var;
      bool last_index_is_loop_var = true;
      for (auto indices : store->region_ta_indices) {
        if (!indices[indices.size() - 1].same_as(loop_var)) {
          last_index_is_loop_var = false;
        }
      }

      std::unordered_set<const Object*> read_tas;
      for (auto input : store->inputs) {
        Var loaded_var;
        if (auto load = input.as<RegionTALoadNode>()) {
          loaded_var = load->region_ta;
        } else if (auto load = input.as<PointerTALoadNode>()) {
          loaded_var = load->pointer_ta;
        } else {
          continue;
        }
        // std::cout << "[TE]    Inputs " << loaded_var << " " << loaded_var << std::endl;
        read_tas.insert(declarations.get_tensor_array(loaded_var)->GetBaseTensorArray().get());
      }

      bool outputs_are_inputs = true;
      for (auto region_ta : store->region_tas) {
        if (!read_tas.count(declarations.get_tensor_array(region_ta)->GetBaseTensorArray().get())) {
          // std::cout << "[TE]    Not found output " << region_ta << " " << region_ta.get()
          // << std::endl;
          outputs_are_inputs = false;
        }
      }

      // std::cout << "[TE]   " << last_index_is_loop_var << (store->direct_inputs.size() == 0)
      //           << (loop->for_type == ForType::Sequential) << outputs_are_inputs << std::endl;

      return last_index_is_loop_var && (store->direct_inputs.size() == 0) &&
             (loop->for_type == ForType::Sequential) && outputs_are_inputs &&
             store->region_tas.size() == 1;
    }
    return false;
  }

  Map<TensorArray, te::Tensor> GetScanIOMapping(const ForNode* loop) {
    auto store = loop->body.as<RegionTAStoreNode>();
    std::unordered_set<const Object*> stored_tas;
    for (auto region_ta : store->region_tas) {
      // stored_tas.insert(declarations.get_tensor_array(region_ta)->GetBaseTensorArray().get());
      stored_tas.insert(declarations.get_tensor_array(region_ta).get());
    }

    auto capsule = TECapsule::capsules.at(store->te_graph_name);

    Map<TensorArray, te::Tensor> ret;
    for (size_t i = 0; i < capsule->inputs.size(); ++i) {
      auto input = store->inputs[capsule->input_vars.size() + i];
      Var loaded_var;
      if (auto load = input.as<RegionTALoadNode>()) {
        loaded_var = load->region_ta;
      } else if (auto load = input.as<PointerTALoadNode>()) {
        loaded_var = load->pointer_ta;
      } else {
        continue;
      }
      // std::cout << "[TE]    Inputs " << loaded_var << " " << loaded_var << std::endl;
      // auto read_ta = declarations.get_tensor_array(loaded_var)->GetBaseTensorArray();
      auto read_ta = declarations.get_tensor_array(loaded_var);
      if (stored_tas.count(read_ta.get())) {
        ret.Set(read_ta, capsule->inputs[i]);
      }
    }
    return ret;
  }

  Stmt VisitStmt_(const ForNode* loop) override {
    if (CanBeLiftedToComputeOp(loop)) {
      // std::cout << "[TE] Lifting to compute " << loop->loop_var << std::endl;
      return LiftLoopToComputeOp(declarations, loop, {});
    } else if (CanBeLiftedToScanOp(loop)) {
      // std::cout << "[TE] Lifting to scan " << loop->loop_var << std::endl;
      return LiftLoopToScanOp(declarations, loop, GetScanIOMapping(loop));
    } else {
      return StmtExprMutator::VisitStmt_(loop);
    }
  }

 private:
  TADeclarations declarations;
};

tir::Stmt lift_to_ter(TADeclarations declarations, const tir::Stmt& input_program) {
  check_ta_uses(declarations, input_program);

  // std::cout << "[TE] Lifting" << std::endl;
  Stmt stmt = input_program;
  for (size_t i = 0; i < 2; ++i) {
    LoopFinderAndLowerer lowerer(declarations);
    stmt = lowerer(stmt);
    ReshapeHoister hoister;
    stmt = hoister(stmt);
    // std::cout << "[TE] Lifted to\n " << stmt << std::endl;
  }
  // std::cout << "[TE] Lifted" << std::endl;
  check_ta_uses(declarations, stmt);
  return stmt;
}

TVM_REGISTER_GLOBAL("tir.lift_to_te").set_body_typed(lift_to_ter);

}  // namespace tvm
