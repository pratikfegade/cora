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
#include <tvm/tir/te_capsule.h>

#include <algorithm>
#include <mutex>
#include <stack>

#include "../te/schedule/graph.h"
#include "../tir/ir/var_replacer.h"

namespace tvm {
using namespace tvm::te;
using namespace tvm::tir;

class InputArgumentLowerer : public ExprMutator {
 public:
  InputArgumentLowerer(std::unordered_map<const Object*, tir::TensorArray> var_ta_mapping_,
                       std::unordered_map<const Object*, tir::Buffer> var_buf_mapping_,
                       Map<Var, Var> var_var_mapping_, Map<Var, te::Tensor> var_tensor_mapping_,
                       Var orig_loop_var_, Var new_loop_var_)
      : var_ta_mapping(var_ta_mapping_),
        var_buf_mapping(var_buf_mapping_),
        var_tensor_mapping(var_tensor_mapping_),
        var_var_mapping(var_var_mapping_) {
    rmap[orig_loop_var_.get()] = new_loop_var_;
  }

  bool is_orig_callee(const PrimExprNode* expr) { return this->orig_call_argument == expr; }

  PrimExpr lower_input_argument(PrimExpr argument, const CallNode* orig_call) {
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
        PrimExpr ret = var_replacer(CallNode::make(tensor->dtype, tensor->op->name, orig_call->args,
                                                   CallNode::Halide, orig_call->argument_dimensions,
                                                   tensor->op, tensor->value_index));
        // std::cout << "[TE]    Call in argument " << GetRef<PrimExpr>(var) << " " << ret << " "
        //           << ret->dtype << std::endl;
        return ret;
      } else {
        // We're to use the same old tensor for this as there was no
        // need to create a new one.
        return GetRef<PrimExpr>(orig_call);
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
      args.push_back_all(orig_call->args);
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
    //           << ret->dtype << std::endl;
    return ret;
  }

  PrimExpr VisitExpr_(const RegionTALoadNode* load) override {
    Var ta_var = load->region_ta;
    CHECK(var_tensor_mapping.count(ta_var)) << ta_var << " " << ta_var.get();
    te::Tensor tensor = var_tensor_mapping.at(ta_var);
    Array<PrimExpr> args;
    for (auto index : load->indices) {
      args.push_back(ExprFunctor::VisitExpr(index));
    }
    if (is_orig_callee(load)) {
      args.push_back_all(orig_call->args);
    }
    Array<Dimension> arg_dims;
    if (auto pl_op = tensor->op.as<PlaceholderOpNode>()) {
      for (size_t i = 0; i < load->indices.size(); ++i) {
        arg_dims.push_back(pl_op->self_index_dimensions[i]);
      }
    } else if (auto bvd_op = tensor->op.as<BaseVarDimOpNode>()) {
      for (size_t i = 0; i < load->indices.size(); ++i) {
        arg_dims.push_back(bvd_op->GetBaseIndexDimension(tensor->value_index, i));
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
    //           << ret->dtype << std::endl;
    return ret;
  }

  std::unordered_map<const Object*, tir::TensorArray> var_ta_mapping;
  std::unordered_map<const Object*, tir::Buffer> var_buf_mapping;
  Map<Var, te::Tensor> var_tensor_mapping;
  Map<Var, Var> var_var_mapping;
  std::unordered_map<const VarNode*, PrimExpr> rmap;
  const Object* orig_call_argument;
  const CallNode* orig_call;
};

class OpBodyLowerer : public ExprMutator {
 public:
  OpBodyLowerer(Map<Var, PrimExpr> input_vars_, Map<te::Operation, PrimExpr> input_arguments_,
                std::unordered_map<const Object*, tir::TensorArray> var_ta_mapping_,
                std::unordered_map<const Object*, tir::Buffer> var_buf_mapping_,
                Map<Var, Var> var_var_mapping_, Map<Var, te::Tensor> var_tensor_mapping_,
                Map<te::Operation, te::Operation>* p_new_op_mapping_, Var orig_loop_var_,
                Var new_loop_var_, Dimension new_loop_dim_)
      : input_vars(input_vars_),
        input_arguments(input_arguments_),
        var_ta_mapping(var_ta_mapping_),
        var_buf_mapping(var_buf_mapping_),
        var_var_mapping(var_var_mapping_),
        var_tensor_mapping(var_tensor_mapping_),
        new_op_mapping(*p_new_op_mapping_),
        new_loop_var(new_loop_var_),
        orig_loop_var(orig_loop_var_),
        new_loop_dim(new_loop_dim_) {}

  PrimExpr lower_body(PrimExpr body) {
    PrimExpr expr = this->VisitExpr(body);
    std::unordered_map<const VarNode*, PrimExpr> rmap;
    for (auto it : input_vars) {
      rmap[it.first.as<VarNode>()] = this->VisitExpr(it.second);
    }
    VarReplacer replacer(rmap);
    return replacer(expr);
  }

  PrimExpr VisitExpr_(const CallNode* call) override {
    if (call->call_type == CallNode::Halide && call->func.defined() &&
        call->func.as<OperationNode>()) {
      auto op = Downcast<Operation>(call->func);
      if (auto pl_op = op.as<PlaceholderOpNode>()) {
        if (input_arguments.count(op)) {
          PrimExpr argument = input_arguments.at(op);
          InputArgumentLowerer input_arg_lowerer(var_ta_mapping, var_buf_mapping, var_var_mapping,
                                                 var_tensor_mapping, orig_loop_var, new_loop_var);
          auto ret = input_arg_lowerer.lower_input_argument(argument, call);
          // std::cout << "[TE]   Lowering argument " << argument << " " << ret << std::endl;
          return ret;
        } else {
          // std::cout << "[TE]   Skipping " << op << " " << call->value_index << std::endl;
          return ExprMutator::VisitExpr_(call);
        }
      } else if (new_op_mapping.count(op)) {
        auto new_op = new_op_mapping.at(op);
        Array<PrimExpr> args;
        args.push_back(new_loop_var);
        args.push_back_all(call->args);
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
    if (var_var_mapping.count(GetRef<Var>(var))) {
      return var_var_mapping.at(GetRef<Var>(var));
    } else {
      return GetRef<Var>(var);
    }
  }

  Map<Var, PrimExpr> input_vars;
  Map<te::Operation, PrimExpr> input_arguments;
  std::unordered_map<const Object*, tir::TensorArray> var_ta_mapping;
  std::unordered_map<const Object*, tir::Buffer> var_buf_mapping;
  Map<Var, Var> var_var_mapping;
  Map<Var, te::Tensor> var_tensor_mapping;
  Map<te::Operation, te::Operation> new_op_mapping;
  Var new_loop_var;
  Var orig_loop_var;
  Dimension new_loop_dim;
};

te::Tensor MakeTensor(std::string name, Array<PrimExpr> orig_shape, Array<PrimExpr> tensor_shape,
                      DataType dtype, te::Tensor original, Array<PrimExpr> index_exprs,
                      Var orig_loop_var, Range loop_range, Dimension new_loop_dim) {
  Array<PrimExpr> shape;
  Array<Dimension> self_index_dimensions;
  Array<Dimension> dimensions;
  Array<tir::IterVar> itervars;
  Array<tir::UninterpFun> uninterpfuns;

  int i = 0;
  for (size_t i = 0; i < index_exprs.size(); ++i) {
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
                      Dimension new_loop_dim) {
  return MakeTensor(rta->name, rta->shape, rta->tensor_shape, rta->dtype, original, index_exprs,
                    orig_loop_var, loop_range, new_loop_dim);
}

te::Tensor MakeTensor(Buffer buf, te::Tensor original, PrimExpr index_expr, Var orig_loop_var,
                      Range loop_range, Dimension new_loop_dim) {
  return MakeTensor(buf->name, buf->shape, {}, buf->dtype, original, {index_expr}, orig_loop_var,
                    loop_range, new_loop_dim);
}

Stmt LiftLoop(const Array<tir::TensorArray> tensor_arrays, const Array<tir::Buffer> buffers,
              const ForNode* loop) {
  class ProcessInputArgument : public ExprVisitor {
   public:
    ProcessInputArgument(std::unordered_map<const Object*, tir::TensorArray> var_ta_mapping_,
                         std::unordered_map<const Object*, tir::Buffer> var_buf_mapping_,
                         const ForNode* loop_, Dimension new_loop_dim_,
                         Map<tir::Var, tir::Var>* p_var_var_mapping_,
                         Map<tir::Var, te::Tensor>* p_var_tensor_mapping_,
                         Map<PrimExpr, Var>* p_new_input_var_tensor_mapping_,
                         Map<PrimExpr, te::Tensor>* p_new_input_tensor_mapping_)
        : var_ta_mapping_inner(var_ta_mapping_),
          var_buf_mapping_inner(var_buf_mapping_),
          loop_inner(loop_),
          new_loop_dim_inner(new_loop_dim_),
          var_var_mapping_inner(*p_var_var_mapping_),
          var_tensor_mapping_inner(*p_var_tensor_mapping_),
          new_input_var_mapping_inner(*p_new_input_var_tensor_mapping_),
          new_input_tensor_mapping_inner(*p_new_input_tensor_mapping_) {}

    void process_argument(PrimExpr input, te::Tensor input_tensor) {
      // std::cout << "[TE] Input " << input << std::endl;

      if (!VarFinder::ContainsVariable(input, loop_inner->loop_var)) {
        // std::cout << "[TE]  Independent of loop variable" << std::endl;
        new_input_tensor_mapping_inner.Set(input, input_tensor);
      } else {
        if (auto load = input.as<RegionTALoadNode>()) {
          // std::cout << "[TE]  RegionTALoad " << input << std::endl;
          auto ta = var_ta_mapping_inner.at(load->region_ta.get());
          auto tensor = MakeTensor(
              ta.as<RegionTensorArrayNode>(), input_tensor, load->indices, loop_inner->loop_var,
              Range::make_by_min_extent(loop_inner->min, loop_inner->extent), new_loop_dim_inner);
          // std::cout << "[TE]   Made tensor for inputs " << ta << " " << tensor << std::endl;
          var_tensor_mapping_inner.Set(load->region_ta, tensor);
          // new_input_tensor_mapping_inner.Set(load->region_ta, tensor);
          new_input_tensor_mapping_inner.Set(
              RegionTALoadNode::make(load->region_ta, {}, load->dtype), tensor);

          for (auto index : load->indices) {
            this->VisitExpr(index);
          }
        } else if (auto load = input.as<PointerTALoadNode>()) {
        } else if (auto varnode = input.as<VarNode>()) {
        } else if (auto load = input.as<LoadNode>()) {
        } else {
          CHECK(false) << input;
        }
      }
    }

    void process_argument(PrimExpr input, Var input_var) {
      // std::cout << "[TE] Input " << input << std::endl;

      if (!VarFinder::ContainsVariable(input, loop_inner->loop_var)) {
        // std::cout << "[TE]  Independent of loop variable" << std::endl;
        new_input_var_mapping_inner.Set(input, input_var);
      } else {
        if (auto load = input.as<LoadNode>()) {
          // std::cout << "[TE]  BufferLoad " << input << std::endl;
          auto buf = var_buf_mapping_inner.at(load->buffer_var.get());
          auto tensor = MakeTensor(buf, {}, load->index, loop_inner->loop_var,
                                   Range::make_by_min_extent(loop_inner->min, loop_inner->extent),
                                   new_loop_dim_inner);
          // std::cout << "[TE]   Made tensor for inputs " << buf << " " << tensor << std::endl;
          var_tensor_mapping_inner.Set(load->buffer_var, tensor);
          new_input_tensor_mapping_inner.Set(load->buffer_var, tensor);
          this->VisitExpr(load->index);
        } else if (auto load = input.as<PointerTALoadNode>()) {
        } else if (auto varnode = input.as<VarNode>()) {
        } else if (auto load = input.as<RegionTALoadNode>()) {
        } else {
          CHECK(false) << input;
        }
      }
    }

    void VisitExpr_(const RegionTALoadNode* load) override {
      // std::cout << "[TE]  RegionTALoad " << GetRef<PrimExpr>(load) << std::endl;
      auto ta = var_ta_mapping_inner.at(load->region_ta.get());
      auto tensor = MakeTensor(
          ta.as<RegionTensorArrayNode>(), {}, load->indices, loop_inner->loop_var,
          Range::make_by_min_extent(loop_inner->min, loop_inner->extent), new_loop_dim_inner);
      // std::cout << "[TE]   Made tensor for inputs " << ta << " " << tensor << " "
      // << load->region_ta.get() << std::endl;
      var_tensor_mapping_inner.Set(load->region_ta, tensor);
      new_input_tensor_mapping_inner.Set(load->region_ta, tensor);
    }

    void VisitExpr_(const VarNode* varnode) override {
      Var var = GetRef<Var>(varnode);
      if (!new_input_var_mapping_inner.count(var) && !var.same_as(loop_inner->loop_var)) {
        Var param_var = var.copy_with_suffix("_p");
        new_input_var_mapping_inner.Set(var, param_var);
        var_var_mapping_inner.Set(var, param_var);
      }
    }

    void VisitExpr_(const LoadNode* load) override {
      // std::cout << "[TE]  Load " << GetRef<PrimExpr>(load) << std::endl;
      CHECK(var_buf_mapping_inner.count(load->buffer_var.get()))
          << load->buffer_var << " " << load->buffer_var.get();
      auto buf = var_buf_mapping_inner.at(load->buffer_var.get());
      auto tensor = MakeTensor(buf, {}, load->index, loop_inner->loop_var,
                               Range::make_by_min_extent(loop_inner->min, loop_inner->extent),
                               new_loop_dim_inner);
      // std::cout << "[TE]   Made tensor for inputs " << buf << " " << tensor << std::endl;
      var_tensor_mapping_inner.Set(load->buffer_var, tensor);
      new_input_tensor_mapping_inner.Set(load->buffer_var, tensor);
    }

    std::unordered_map<const Object*, tir::TensorArray> var_ta_mapping_inner;
    std::unordered_map<const Object*, tir::Buffer> var_buf_mapping_inner;
    const ForNode* loop_inner;
    Dimension new_loop_dim_inner;
    Map<tir::Var, tir::Var>& var_var_mapping_inner;
    Map<tir::Var, te::Tensor>& var_tensor_mapping_inner;
    Map<PrimExpr, Var>& new_input_var_mapping_inner;
    Map<PrimExpr, te::Tensor>& new_input_tensor_mapping_inner;
  };

  auto store = loop->body.as<RegionTAStoreNode>();
  auto loop_var = loop->loop_var;
  auto capsule = TECapsule::capsules.at(store->te_graph_name);

  std::unordered_map<const Object*, tir::TensorArray> var_ta_mapping;
  for (auto ta : tensor_arrays) {
    var_ta_mapping[ta->ta_var.get()] = ta;
  }
  std::unordered_map<const Object*, tir::Buffer> var_buf_mapping;
  for (auto buf : buffers) {
    var_buf_mapping[buf->data.get()] = buf;
  }

  Dimension new_loop_dim =
      DimensionNode::make(loop_var->name_hint + "_dim", DimensionNode::kRangeDim);

  Map<tir::Var, tir::Var> var_var_mapping;
  Map<tir::Var, te::Tensor> var_tensor_mapping;
  Map<PrimExpr, te::Tensor> new_input_tensor_mapping;
  Map<PrimExpr, Var> new_input_var_mapping;
  ProcessInputArgument arg_processor(var_ta_mapping, var_buf_mapping, loop, new_loop_dim,
                                     &var_var_mapping, &var_tensor_mapping, &new_input_var_mapping,
                                     &new_input_tensor_mapping);
  for (size_t i = 0; i < capsule->input_vars.size(); ++i) {
    // std::cout << "[TE] ArgProcessing " << store->inputs[i] << " " << capsule->input_vars[i]
    // << std::endl;
    arg_processor.process_argument(store->inputs[i], capsule->input_vars[i]);
  }

  for (size_t i = 0; i < capsule->inputs.size(); ++i) {
    // std::cout << "[TE] ArgProcessing " << store->inputs[i + capsule->input_vars.size()] << " "
    // << capsule->inputs[i] << std::endl;
    arg_processor.process_argument(store->inputs[i + capsule->input_vars.size()],
                                   capsule->inputs[i]);
  }

  Array<te::Operation> operations = GetSubGraph(capsule->outputs, capsule->inputs, false);
  Map<Var, PrimExpr> input_var_mapping;
  for (size_t i = 0; i < capsule->input_vars.size(); ++i) {
    input_var_mapping.Set(capsule->input_vars[i], store->inputs[i]);
    // std::cout << "[TE] Input var " << capsule->input_vars[i] << " " << store->inputs[i]
    // << std::endl;
  }
  Map<te::Operation, PrimExpr> input_argument_mapping;
  for (size_t i = 0; i < capsule->inputs.size(); ++i) {
    input_argument_mapping.Set(capsule->inputs[i]->op,
                               store->inputs[i + capsule->input_vars.size()]);
    // std::cout << "[TE] Input argument " << capsule->inputs[i]->op << " "
    // << capsule->inputs[i]->value_index << " "
    // << store->inputs[i + capsule->input_vars.size()] << std::endl;
  }

  Map<te::Operation, te::Operation> new_op_mapping;
  for (auto op : operations) {
    // std::cout << "[TE] OP " << op << std::endl;

    if (auto c_op = op.as<te::ComputeOpNode>()) {
      IterVar new_loop_iv = IterVarNode::make(Range::make_by_min_extent(loop->min, loop->extent),
                                              loop_var.copy_with_suffix("_liv"), kDataPar, "");
      Map<tir::Var, tir::Var> updated_var_var_mapping(var_var_mapping);
      updated_var_var_mapping.Set(loop_var, new_loop_iv->var);
      OpBodyLowerer body_lowerer(input_var_mapping, input_argument_mapping, var_ta_mapping,
                                 var_buf_mapping, var_var_mapping, var_tensor_mapping,
                                 &new_op_mapping, loop_var, new_loop_iv->var, new_loop_dim);
      Array<PrimExpr> new_body_exprs;

      for (auto body_expr : c_op->body) {
        PrimExpr new_body = body_lowerer.lower_body(body_expr);
        new_body_exprs.push_back(new_body);
        // std::cout << "[TE]  Lowering " << body_expr << "\n     " << new_body << std::endl;
        // std::cout << "[TE]  Lowered to " << new_body << std::endl;
      }
      Array<PrimExpr> new_pred_exprs;
      for (auto pred_expr : c_op->pred) {
        // std::cout << "[TE]  Lowering " << pred_expr << std::endl;
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
          c_op->name + "_mummy", c_op->tag, c_op->attrs, itervars, root_index_dimensions,
          output_shape_storage, itervars, dimensions, uninterpfuns, new_body_exprs, new_pred_exprs);
      new_op_mapping.Set(op, new_op);
    }
  }

  // New TECapsule fields
  Array<te::Tensor> output_tensors;
  Array<te::Tensor> input_tensors;
  Array<Var> input_vars;

  // New RegionTAStore fields
  Array<PrimExpr> op_inputs;
  Array<Array<PrimExpr>> region_ta_indices;
  {
    for (auto tensor : capsule->outputs) {
      output_tensors.push_back(new_op_mapping.at(tensor->op).output(tensor->value_index));
    }

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

    mut_capsule->input_vars = input_vars;
    mut_capsule->inputs = input_tensors;
    mut_capsule->outputs = output_tensors;
    mut_capsule->all_ops_ = {};
  }

  return RegionTAStoreNode::make(store->region_tas, region_ta_indices, capsule->name, op_inputs,
                                 store->direct_inputs);

  // return GetRef<Stmt>(loop);
}

class LoopFinderAndLowerer : public tir::StmtExprMutator {
 public:
  LoopFinderAndLowerer(const Array<tir::TensorArray> tensor_arrays_,
                       const Array<tir::Buffer> buffers_)
      : tensor_arrays(tensor_arrays_), buffers(buffers_) {}

  Stmt VisitStmt_(const ForNode* loop) override {
    if (auto store = loop->body.as<RegionTAStoreNode>()) {
      Var loop_var = loop->loop_var;
      bool last_index_is_loop_var = true;
      for (auto indices : store->region_ta_indices) {
        if (!indices[indices.size() - 1].same_as(loop_var)) {
          last_index_is_loop_var = false;
        }
      }
      if (last_index_is_loop_var && store->direct_inputs.size() == 0) {
        return LiftLoop(tensor_arrays, buffers, loop);
      } else {
        return StmtExprMutator::VisitStmt_(loop);
      }
    } else {
      return StmtExprMutator::VisitStmt_(loop);
    }
  }

 private:
  const Array<tir::TensorArray> tensor_arrays;
  const Array<tir::Buffer> buffers;
};

tir::Stmt lift_to_ter(const Array<tir::TensorArray> tensor_arrays, const Array<tir::Buffer> buffers,
                      const tir::Stmt& input_program) {
  std::cout << "[TE] Lifting" << std::endl;
  LoopFinderAndLowerer lowerer(tensor_arrays, buffers);
  Stmt stmt = lowerer(input_program);
  std::cout << "[TE] Lifted to\n " << stmt << std::endl;
  return stmt;
}

TVM_REGISTER_GLOBAL("tir.lift_to_te").set_body_typed(lift_to_ter);

}  // namespace tvm
