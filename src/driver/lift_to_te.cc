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
      auto ret = this->VisitExpr(argument);
      // std::cout << "[TE]    Lowered to " << ret << std::endl;
      return ret;
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
          // std::cout << "[TE]    Call in argument1 " << GetRef<PrimExpr>(var) << " " << ret << " "
          // << ret->dtype << std::endl;
          return ret;
        } else {
          // std::cout << "[TE]    ORig tensor frot " << GetRef<PrimExpr>(var) << std::endl;

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
      // std::cout << "[TE]    Call in argument2 " << GetRef<PrimExpr>(load) << " " << ret << " "
      // << ret->dtype << std::endl;
      return ret;
    }

    PrimExpr VisitExpr_(const RegionTALoadNode* load) override {
      // std::cout << "[TE]    Lowering load " << GetRef<PrimExpr>(load) << std::endl;
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
          // args.push_back(ExprFunctor::VisitExpr(index));
          args.push_back(this->VisitExpr(index));
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
          // std::cout << "[TE]    Arg dims1 " << pl_op->self_index_dimensions[0] << std::endl;
          arg_dims.push_back(pl_op->self_index_dimensions[0]);
        } else {
          // std::cout << "[TE]    Arg dims2 " << pl_op->self_index_dimensions << " "
          // << load->indices.size() << std::endl;
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
        // std::cout << "[TE]    Arg dims3 " << orig_call->argument_dimensions << std::endl;
        arg_dims.push_back_all(orig_call->argument_dimensions);
      }

      if (tensor->shape.size() != args.size()) {
        // std::cout << "[TE]    Mismaytchd shaped " << tensor << " " << args << std::endl;
        CHECK(tensor->shape[tensor->shape.size() - 1].as<IntImmNode>()->value == 1) << tensor;
        args.push_back(IntImm(DataType::Int(32), 0));
        CHECK_EQ(args.size(), arg_dims.size()) << args << " " << arg_dims;
      }

      VarReplacer var_replacer(rmap);
      PrimExpr ret =
          var_replacer(CallNode::make(tensor->dtype, tensor->op->name, args, CallNode::Halide,
                                      arg_dims, tensor->op, tensor->value_index));
      // std::cout << "[TE]    Call in argument3 " << GetRef<PrimExpr>(load) << " " << ret << " "
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
                Map<te::Operation, te::Operation>* p_new_op_mapping_, Dimension new_loop_dim_,
                Var orig_loop_var_, Var new_loop_var_)
      : input_vars(input_vars_),
        input_arguments(input_arguments_),
        declarations(declarations_),
        var_var_mapping(var_var_mapping_),
        var_tensor_mapping(var_tensor_mapping_),
        tas_tensorize_only_one_dim(tas_tensorize_only_one_dim_),
        new_op_mapping(*p_new_op_mapping_),
        new_loop_dim(new_loop_dim_),
        new_loop_var(new_loop_var_),
        orig_loop_var(orig_loop_var_),
        input_lowerer(this, declarations, var_var_mapping, var_tensor_mapping,
                      tas_tensorize_only_one_dim, orig_loop_var, new_loop_var) {
    // std::cout << "[TE]   Body lowerer created " << orig_loop_var << " " << new_loop_var
    // << std::endl;
    for (auto it : input_vars) {
      rmap_[it.first.as<VarNode>()] = input_lowerer.lower_input_argument(it.second, nullptr);
      // std::cout << "[TE]    rmap " << it.first << " " << rmap_[it.first.as<VarNode>()] <<
      // std::endl;
    }
  }

  PrimExpr lower_body(PrimExpr body) {
    // std::cout << "[TE]   Lowering body " << body << std::endl;
    PrimExpr expr = this->VisitExpr(body);
    VarReplacer replacer(rmap_);
    return replacer(expr);
  }

  Range lower_range(Range range) {
    return Range::make_by_min_extent(lower_body(range->min), lower_body(range->extent));
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
    // std::cout << "[TE]  Call  " << GetRef<PrimExpr>(call) << " " << call->func << " "
    //           << call->call_type << " " << CallNode::UninterpFunCall << std::endl;

    if (call->call_type == CallNode::Halide && call->func.defined() &&
        call->func.as<OperationNode>()) {
      auto op = Downcast<Operation>(call->func);
      if (auto pl_op = op.as<PlaceholderOpNode>()) {
        if (input_arguments.count(op)) {
          PrimExpr argument = input_arguments.at(op);
          auto ret = input_lowerer.lower_input_argument(argument, call);
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
    } else if (call->func.defined() && call->func.as<UninterpFunNode>()) {
      auto ufun = call->func.as<UninterpFunNode>();

      auto new_body = lower_body(ufun->body);
      // std::cout << "[TE]     Ufun body " << ufun->body << " " << new_body << std::endl;

      // If the new body contains the new_loop_iv, make that a
      // paramter of the UF
      Array<Var> parameters = ufun->parameters;
      Array<Dimension> dimensions = ufun->dimensions;
      {
        if (VarFinder::ContainsVariable(new_body, new_loop_var)) {
          Var param_var = new_loop_var.copy_with_suffix("_p2");
          new_body = VarReplacer({{new_loop_var.get(), param_var}})(new_body);
          parameters.push_back(param_var);
          dimensions.push_back(new_loop_dim);
          // std::cout << "[TE]      Replacing new loop var " << new_body << std::endl;
        }
      }

      auto new_ufun = UninterpFunNode::make(ufun->fname + "_se", lower_range(ufun->range),
                                            dimensions, parameters, new_body);

      Array<PrimExpr> args;
      Array<Dimension> arg_dims = call->argument_dimensions;
      for (auto arg : call->args) {
        args.push_back(this->VisitExpr(arg));
      }
      if (parameters.size() > ufun->parameters.size()) {
        args.push_back(new_loop_var);
        arg_dims.push_back(new_loop_dim);
      }
      return CallNode::make(call->dtype, call->name + "_se", args, CallNode::UninterpFunCall,
                            arg_dims, new_ufun, call->value_index);
    }
    return ExprMutator::VisitExpr_(call);
  }

  PrimExpr VisitExpr_(const VarNode* var) override {
    bool print = (var->name_hint == "batch_idx_p");
    PrimExpr ret = GetRef<Var>(var);
    if (var_var_mapping.count(GetRef<Var>(var))) {
      ret = var_var_mapping.at(GetRef<Var>(var));
    }
    // if (print) std::cout << "[TE]   Var argument " << var->name_hint << " " << ret << std::endl;
    return ret;
  }

  PrimExpr lower_argument(PrimExpr argument) {
    return input_lowerer.lower_input_argument(argument, nullptr);
  }

  Map<Var, PrimExpr> input_vars;
  Map<te::Operation, PrimExpr> input_arguments;
  TADeclarations declarations;
  Map<Var, Var> var_var_mapping;
  Map<Var, te::Tensor> var_tensor_mapping;
  std::unordered_set<const Object*> tas_tensorize_only_one_dim;
  Map<te::Operation, te::Operation> new_op_mapping;
  Dimension new_loop_dim;
  Var new_loop_var;
  Var orig_loop_var;
  InputArgumentLowerer input_lowerer;
  std::unordered_map<const VarNode*, PrimExpr> rmap_;
};

class UniqueNamer {
 public:
  std::string get_unique_name(std::string prefix) {
    // if (!names.count(prefix)) return prefix;
    std::string ret = prefix;
    int i = 0;
    while (names.count(ret = prefix + std::to_string(i))) {
      i++;
    }
    names.insert(ret);
    return ret;
  }
  std::unordered_set<std::string> names;
};

te::Tensor MakeTensor(std::string name, Array<PrimExpr> orig_shape, Array<PrimExpr> tensor_shape,
                      DataType dtype, te::Tensor original, Array<PrimExpr> index_exprs,
                      Var orig_loop_var, Range loop_range, Dimension new_loop_dim,
                      UniqueNamer* namer, bool only_one_dim = false) {
  Array<PrimExpr> shape;
  Array<Dimension> self_index_dimensions;
  Array<Dimension> dimensions;
  Array<tir::IterVar> itervars;
  Array<tir::UninterpFun> uninterpfuns;
  // std::cout << "[MT] Making tensor  " << name << " " << index_exprs << " " << only_one_dim << " "
  // << original << std::endl;
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

  auto unique_name = namer->get_unique_name(name);
  // std::cout << "[MT]   Making tensor  " << unique_name << std::endl;
  Operation op = PlaceholderOpNode::make(unique_name, shape, dtype, self_index_dimensions,
                                         dimensions, itervars, uninterpfuns);
  return op.output(original.defined() ? original->value_index : 0);
}

te::Tensor MakeTensor(const RegionTensorArrayNode* rta, te::Tensor original,
                      Array<PrimExpr> index_exprs, Var orig_loop_var, Range loop_range,
                      Dimension new_loop_dim, UniqueNamer* namer, bool only_one_dim = false) {
  std::string name = rta->name;
  if (original.defined()) {
    name = original->op->name + "_nt";
  }
  auto ret = MakeTensor(name, rta->shape, rta->tensor_shape, rta->dtype, original, index_exprs,
                        orig_loop_var, loop_range, new_loop_dim, namer, only_one_dim);
  // std::cout << "[TE] Made tensor " << ret << " " << GetRef<TensorArray>(rta) << std::endl;
  return ret;
}

te::Tensor MakeTensor(Buffer buf, te::Tensor original, PrimExpr index_expr, Var orig_loop_var,
                      Range loop_range, Dimension new_loop_dim, UniqueNamer* namer,
                      bool only_one_dim = false) {
  return MakeTensor(buf->name, buf->shape, {}, buf->dtype, original, {index_expr}, orig_loop_var,
                    loop_range, new_loop_dim, namer, only_one_dim);
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
                       std::unordered_map<te::Tensor, te::Tensor>* p_old_tensor_new_tensor_mapping_,
                       UniqueNamer* tensor_namer_)
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
        old_tensor_new_tensor_mapping(*p_old_tensor_new_tensor_mapping_),
        tensor_namer(tensor_namer_) {}

  TensorArray get_reshaped_tensor_array(TensorArray orig) {
    if (new_reshaped_tensor_arrays.count(orig)) {
      return new_reshaped_tensor_arrays.at(orig);
    } else {
      TensorArray reshaped;
      auto rta = orig.as<RegionTensorArrayNode>();
      Array<PrimExpr> new_tensor_shape;
      Array<PrimExpr> new_tensor_array_shape;
      if (tas_tensorize_only_one_dim.count(orig.get())) {
        for (size_t i = 0; i < rta->shape.size() - 1; ++i) {
          new_tensor_array_shape.push_back(rta->shape[i]);
        }
        new_tensor_shape.push_back(rta->shape[rta->shape.size() - 1]);
      } else {
        new_tensor_shape.push_back_all(rta->shape);
      }
      new_tensor_shape.push_back_all(rta->tensor_shape);

      reshaped = RegionTensorArrayNode::make(rta->ta_var.copy_with_suffix("_reshaped"), rta->dtype,
                                             new_tensor_array_shape, new_tensor_shape,
                                             rta->name + "_reshaped", rta->base_region_ta);
      new_reshaped_tensor_arrays.Set(orig, reshaped);
      declarations.add_tensor_array(reshaped);

      return reshaped;
    }
  }

  void process_dependent_argument(PrimExpr input, te::Tensor original_tensor) {
    if (auto load = input.as<RegionTALoadNode>()) {
      // std::cout << "[TE]  RegionTALoad " << input << " " << original_tensor << std::endl;
      auto ta = declarations.get_tensor_array(load->region_ta);
      te::Tensor tensor = NullValue<te::Tensor>();
      if (var_tensor_mapping.count(load->region_ta)) {
        tensor = var_tensor_mapping.at(load->region_ta);
      } else {
        tensor = MakeTensor(ta.as<RegionTensorArrayNode>(), original_tensor, load->indices,
                            loop->loop_var, Range::make_by_min_extent(loop->min, loop->extent),
                            new_loop_dim, tensor_namer, tas_tensorize_only_one_dim.count(ta.get()));
        // std::cout << "[TE]   Made tensor for inputs1 " << load->region_ta << " "
        // << load->region_ta.get() << " " << ta << " " << tensor << std::endl;
        var_tensor_mapping.Set(load->region_ta, tensor);
      }
      old_ta_new_tensor_mapping.Set(ta, tensor);
      if (original_tensor.defined()) {
        old_tensor_new_tensor_mapping[original_tensor] = tensor;
      }

      // Create a new reshaped tensor array
      TensorArray reshaped = get_reshaped_tensor_array(ta);

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
        // this->VisitExpr_(varnode);
        // std::cout << "[TE]  VarArg " << input << std::endl;
      }
    } else if (auto load = input.as<LoadNode>()) {
      // std::cout << "[TE]  BufferLoad " << input << std::endl;
      auto buf = declarations.get_buffer(load->buffer_var);
      te::Tensor tensor = NullValue<te::Tensor>();
      if (var_tensor_mapping.count(load->buffer_var)) {
        tensor = var_tensor_mapping.at(load->buffer_var);
      } else {
        tensor = MakeTensor(buf, {}, load->index, loop->loop_var,
                            Range::make_by_min_extent(loop->min, loop->extent), new_loop_dim,
                            tensor_namer);
        // std::cout << "[TE]   Made tensor for inputs2 " << input << " " << tensor << std::endl;
        var_tensor_mapping.Set(load->buffer_var, tensor);
      }
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

  void process_argument(PrimExpr input) {
    // std::cout << "[TE] Input Var " << input << " " << input_var << std::endl;
    process_dependent_argument(input, {});
  }

  void VisitExpr_(const RegionTALoadNode* load) override {
    // std::cout << "[TE]  RegionTALoad " << GetRef<PrimExpr>(load) << std::endl;
    auto ta = declarations.get_tensor_array(load->region_ta);
    te::Tensor tensor = NullValue<te::Tensor>();
    if (var_tensor_mapping.count(load->region_ta)) {
      tensor = var_tensor_mapping.at(load->region_ta);
    } else {
      tensor = MakeTensor(ta.as<RegionTensorArrayNode>(), {}, load->indices, loop->loop_var,
                          Range::make_by_min_extent(loop->min, loop->extent), new_loop_dim,
                          tensor_namer, tas_tensorize_only_one_dim.count(ta.get()));
      // std::cout << "[TE]   Made tensor for inputs3 " << GetRef<PrimExpr>(load) << " " << tensor
      // << " " << load->region_ta.get() << std::endl;
      var_tensor_mapping.Set(load->region_ta, tensor);
    }
    old_ta_new_tensor_mapping.Set(ta, tensor);

    // Create a new reshaped tensor array
    TensorArray reshaped = get_reshaped_tensor_array(ta);

    Array<PrimExpr> new_indices;
    if (tas_tensorize_only_one_dim.count(ta.get())) {
      for (size_t i = 0; i < load->indices.size() - 1; ++i) {
        new_indices.push_back(load->indices[i]);
      }
    }

    new_input_tensor_mapping.Set(RegionTALoadNode::make(reshaped->ta_var, new_indices, load->dtype),
                                 tensor);

    if (tas_tensorize_only_one_dim.count(ta.get())) {
      this->VisitExpr(load->indices[load->indices.size() - 1]);
    } else {
      for (auto index : load->indices) {
        this->VisitExpr(index);
      }
    }
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
    te::Tensor tensor = NullValue<te::Tensor>();
    if (var_tensor_mapping.count(load->buffer_var)) {
      tensor = var_tensor_mapping.at(load->buffer_var);
    } else {
      tensor = MakeTensor(buf, {}, load->index, loop->loop_var,
                          Range::make_by_min_extent(loop->min, loop->extent), new_loop_dim,
                          tensor_namer);
      // std::cout << "[TE]   Made tensor for inputs4 " << GetRef<PrimExpr>(load) << " " << tensor
      // << std::endl;
      var_tensor_mapping.Set(load->buffer_var, tensor);
    }
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
  UniqueNamer* tensor_namer;
};

struct ScanLiftingInfo {
  bool init_separate;
  Stmt init_stmt;
  Stmt update_stmt;
  PrimExpr init_limit;
  Map<TensorArray, te::Tensor> update_io_mapping;
};

class OneStoreLifter : UniqueNamer {
 public:
  OneStoreLifter(std::string creator) {}

  void Lift(TADeclarations declarations, const RegionTAStoreNode* store, const ForNode* loop,
            Array<TensorArray> tas_tensorize_only_one_dim, Dimension new_loop_dim, Range loop_range,
            PrimExpr loop_extent_upper_bound,
            Map<TensorArray, TensorArray>* p_new_reshaped_tensor_arrays,
            Array<Array<PrimExpr>>* p_region_ta_indices, Array<PrimExpr>* p_op_inputs,
            Map<TensorArray, te::Tensor>* p_old_ta_new_tensor_mapping) {
    auto loop_var = loop->loop_var;
    auto capsule = TECapsule::capsules.at(store->te_graph_name);

    Map<tir::Var, tir::Var> var_var_mapping;
    Map<tir::Var, te::Tensor> var_tensor_mapping;
    Map<PrimExpr, te::Tensor> new_input_tensor_mapping;
    Map<PrimExpr, Var> new_input_var_mapping;
    std::unordered_set<const Object*> tas_tensorize_only_one_dim_set;
    std::unordered_map<te::Tensor, te::Tensor> old_tensor_new_tensor_mapping;
    Map<TensorArray, te::Tensor>& old_ta_new_tensor_mapping = *p_old_ta_new_tensor_mapping;
    Map<TensorArray, TensorArray>& new_reshaped_tensor_arrays = *p_new_reshaped_tensor_arrays;
    for (auto ta : tas_tensorize_only_one_dim) {
      // std::cout << "[TE] ScanIO " << ta << std::endl;
      tas_tensorize_only_one_dim_set.insert(ta.get());
    }
    ProcessInputArgument arg_processor(
        declarations, loop, new_loop_dim, tas_tensorize_only_one_dim_set, &var_var_mapping,
        &var_tensor_mapping, &new_input_var_mapping, &new_input_tensor_mapping,
        &new_reshaped_tensor_arrays, &old_ta_new_tensor_mapping, &old_tensor_new_tensor_mapping,
        this);
    CHECK_EQ(capsule->inputs.size() + capsule->input_vars.size(), store->inputs.size());
    for (size_t i = 0; i < capsule->input_vars.size(); ++i) {
      arg_processor.process_argument(store->inputs[i]);
    }
    for (size_t i = 0; i < capsule->inputs.size(); ++i) {
      arg_processor.process_argument(store->inputs[i + capsule->input_vars.size()],
                                     capsule->inputs[i]);
    }
    arg_processor.process_argument(loop_range->min);
    arg_processor.process_argument(loop_range->extent);

    Array<te::Tensor> all_inputs(capsule->inputs);
    all_inputs.push_back_all(capsule->non_external_inputs);
    // std::cout << "[TE] Lifting ops between " << capsule->outputs << " " << capsule->inputs << " "
    // << capsule->non_external_inputs << std::endl;
    Array<te::Operation> operations = GetSubGraphOrAllGraph(capsule->outputs, all_inputs, true);
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
      // std::cout << "[TE]  Lifting op " << op << std::endl;

      Operation new_op = NullValue<Operation>();
      if (auto pl_op = op.as<te::PlaceholderOpNode>()) {
        auto new_name = get_unique_name(pl_op->name);
        if (new_name == pl_op->name) continue;
        auto n = make_object<PlaceholderOpNode>();
        n->name = get_unique_name(pl_op->name);
        n->tag = pl_op->tag;
        n->attrs = pl_op->attrs;
        n->shape = pl_op->shape;
        n->dtype = pl_op->dtype;
        n->self_index_dimensions = pl_op->self_index_dimensions;
        n->all_dimensions = pl_op->all_dimensions;
        n->loop_dimensions = pl_op->loop_dimensions;
        n->axis = pl_op->axis;
        n->index_dimensions = pl_op->index_dimensions;
        n->index_expressions = pl_op->index_expressions;
        new_op = Operation(n);
      } else if (auto c_op = op.as<te::ComputeOpNode>()) {
        Var new_loop_var = loop_var.copy_with_suffix("_liv_" + c_op->name);
        // std::cout << "[TE] New loop iv for  " << op << " " << new_loop_var << " " << loop_range
        // << std::endl;
        Map<Var, Var> updated_var_var_mapping(var_var_mapping);
        updated_var_var_mapping.Set(loop_var, new_loop_var);
        OpBodyLowerer body_lowerer(input_var_mapping, input_argument_mapping, declarations,
                                   updated_var_var_mapping, var_tensor_mapping,
                                   tas_tensorize_only_one_dim_set, &new_op_mapping, new_loop_dim,
                                   loop_var, new_loop_var);
        Array<PrimExpr> new_body_exprs;

        for (auto body_expr : c_op->body) {
          // std::cout << "[TE] Old body " << new_loop_iv->var << " " << body_expr << std::endl;
          PrimExpr new_body = body_lowerer.lower_body(body_expr);
          new_body_exprs.push_back(new_body);
          // std::cout << "[TE] New body " << new_body << std::endl;
        }
        Array<PrimExpr> new_pred_exprs;
        for (auto pred_expr : c_op->pred) {
          PrimExpr new_pred = body_lowerer.lower_body(pred_expr);
          new_pred_exprs.push_back(new_pred);
        }

        IterVar new_loop_iv = IterVarNode::make(
            Range::make_by_min_extent(body_lowerer.lower_argument(loop_range->min),
                                      body_lowerer.lower_argument(loop_range->extent)),
            new_loop_var, kDataPar, "");
        // std::cout << "[TE]   New dim range cop " << new_loop_iv->dom << std::endl;

        Array<Dimension> root_index_dimensions;
        Array<PrimExpr> output_shape_storage;
        Array<IterVar> itervars;
        Array<Dimension> dimensions;
        Array<UninterpFun> uninterpfuns;
        {
          root_index_dimensions.push_back(new_loop_dim);
          root_index_dimensions.push_back_all(c_op->GetRootIndexDimensions(0));

          PrimExpr dim_extent = loop_range->extent;
          if (loop_extent_upper_bound.defined()) {
            dim_extent = loop_extent_upper_bound;
          }

          output_shape_storage.push_back(dim_extent);
          output_shape_storage.push_back_all(c_op->output_shape_storage);

          // std::cout << "[TE]  Shape for new dim  " << loop_range->extent << " "
          // << loop_extent_upper_bound << " " << dim_extent << std::endl;
          dimensions.push_back(new_loop_dim);
          itervars.push_back(new_loop_iv);
          uninterpfuns.push_back(NullValue<UninterpFun>());

          for (auto dim_info : c_op->all_dimensions) {
            if (dim_info->dim->isRangeDim()) {
              dimensions.push_back(dim_info->dim);
              auto iv = dim_info->iv;
              Range new_dom = body_lowerer.lower_range(iv->dom);
              // std::cout << "[TE]   Old Dom  " << iv->dom << " " << new_dom << std::endl;

              itervars.push_back(
                  IterVarNode::make(new_dom, iv->var, iv->iter_type, iv->thread_tag));

              auto ufun = dim_info->ufun;
              if (ufun.defined()) {
                uninterpfuns.push_back(UninterpFunNode::make(
                    ufun->fname, body_lowerer.lower_range(ufun->range), ufun->dimensions,
                    ufun->parameters, body_lowerer.lower_body(ufun->body)));
              } else {
                uninterpfuns.push_back(ufun);
              }
            }
          }
        }

        new_op =
            ComputeOpNode::make(get_unique_name(c_op->name + "_te"), c_op->tag, c_op->attrs,
                                itervars, root_index_dimensions, output_shape_storage, itervars,
                                dimensions, uninterpfuns, new_body_exprs, new_pred_exprs);
      } else if (auto s_op = op.as<te::ScanOpNode>()) {
        Array<te::Tensor> state_placeholder;
        Array<te::Tensor> init;
        Array<te::Tensor> update;

        auto remap_tensor = [&](const te::Tensor& tensor) {
          if (new_op_mapping.count(tensor->op)) {
            return new_op_mapping.at(tensor->op).output(tensor->value_index);
          } else {
            // std::cout << "[TE]   No mapping found for " << tensor << std::endl;
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
        auto new_init = remap_tensor(s_op->init[0]);
        state_placeholder.push_back(new_state);
        init.push_back(new_init);
        non_external_input_tensors.push_back(new_state);
        update.push_back(new_update);

        // std::cout << "[TE]   Init " << s_op->init[0] << " " << new_init << std::endl;
        // std::cout << "[TE]   State " << s_op->state_placeholder[0] << " " << new_state <<
        // std::endl;

        for (auto t : capsule->non_external_inputs) {
          if (old_tensor_new_tensor_mapping.count(t)) {
            non_external_input_tensors.push_back(old_tensor_new_tensor_mapping.at(t));
          } else {
            auto remaped = remap_tensor(t);
            if (remaped != t) {
              non_external_input_tensors.push_back(t);
            }
          }
        }
        non_external_input_tensors.push_back(new_init);

        Array<Dimension> explicit_loops;
        Array<UninterpFun> explicit_idx_ufs;
        Array<IterVar> explicit_ivs;
        {
          Var new_loop_var = loop_var.copy_with_suffix("_liv_" + s_op->name);
          Map<Var, Var> updated_var_var_mapping(var_var_mapping);
          updated_var_var_mapping.Set(loop_var, new_loop_var);
          OpBodyLowerer body_lowerer(input_var_mapping, input_argument_mapping, declarations,
                                     updated_var_var_mapping, var_tensor_mapping,
                                     tas_tensorize_only_one_dim_set, &new_op_mapping, new_loop_dim,
                                     loop_var, new_loop_var);

          explicit_loops.push_back(new_loop_dim);
          explicit_idx_ufs.push_back(NullValue<UninterpFun>());
          IterVar new_loop_iv = IterVarNode::make(
              Range::make_by_min_extent(body_lowerer.lower_argument(loop_range->min),
                                        body_lowerer.lower_argument(loop_range->extent)),
              new_loop_var, kDataPar, "");
          explicit_ivs.push_back(new_loop_iv);
          for (size_t i = 0; i < s_op->explicit_dims.size(); ++i) {
            auto dim = s_op->explicit_dims[i];
            auto iv = s_op->explicit_loop_ivs[i];

            Range new_dom = body_lowerer.lower_range(iv->dom);

            explicit_loops.push_back(dim);
            explicit_ivs.push_back(
                IterVarNode::make(new_dom, iv->var, iv->iter_type, iv->thread_tag));
            explicit_idx_ufs.push_back(NullValue<UninterpFun>());
          }
        }

        PrimExpr scan_min = NullValue<PrimExpr>();
        PrimExpr scan_max = NullValue<PrimExpr>();
        if (auto bvd_op = update[0]->op.as<BaseVarDimOpNode>()) {
          auto dom = bvd_op->GetIterVarFromDim(update[0]->value_index, s_op->scan_dim)->dom;
          scan_min = dom->min;
          scan_max = scan_min + dom->extent;
        }

        new_op = ScanOpNode::make(
            get_unique_name(s_op->name), "", {}, UninterpFunNode::from_constant("min", scan_min),
            UninterpFunNode::from_constant("max", scan_max), s_op->scan_dim, false, init, update,
            state_placeholder, s_op->inputs, explicit_loops, explicit_idx_ufs, explicit_ivs);
      } else {
        CHECK(false) << "Lifting " << op << " not yet supported";
      }
      // std::cout << "[TE]   New op " << op << " " << new_op << std::endl;
      new_op_mapping.Set(op, new_op);
    }

    // New TECapsule fields
    Array<te::Tensor> output_tensors;
    for (auto tensor : capsule->outputs) {
      CHECK(new_op_mapping.count(tensor->op)) << tensor->op;
      output_tensors.push_back(new_op_mapping.at(tensor->op).output(tensor->value_index));
    }

    Array<te::Tensor> input_tensors;
    Array<Var> input_vars;
    // New RegionTAStore fields
    Array<PrimExpr>& op_inputs = *p_op_inputs;
    Array<Array<PrimExpr>>& region_ta_indices = *p_region_ta_indices;
    {
      for (auto it : new_input_var_mapping) {
        // std::cout << "[LIFT] Capsule Input Var " << it.second << " " << it.second << std::endl;
        input_vars.push_back(it.second);
        op_inputs.push_back(it.first);
      }

      for (auto it : new_input_tensor_mapping) {
        input_tensors.push_back(it.second);
        // std::cout << "[LIFT] Capsule Input Tensor " << it.second << " " << it.second->op
        // << std::endl;
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
  }
};

Stmt LiftLoopToComputeOp(TADeclarations declarations, Array<TensorArray>* p_scan_tensor_arrays,
                         const ForNode* loop) {
  Array<Array<PrimExpr>> region_ta_indices;
  Map<TensorArray, TensorArray> new_reshaped_tensor_arrays;
  Array<PrimExpr> op_inputs;
  Map<TensorArray, te::Tensor> old_ta_new_tensor_mapping;
  auto store = loop->body.as<RegionTAStoreNode>();
  auto capsule = TECapsule::capsules.at(store->te_graph_name);
  Dimension new_loop_dim =
      DimensionNode::make(loop->loop_var->name_hint + "_dim", DimensionNode::kRangeDim);

  OneStoreLifter("compute").Lift(declarations, store, loop, *p_scan_tensor_arrays, new_loop_dim,
                                 Range::make_by_min_extent(loop->min, loop->extent),
                                 loop->extent_upper_bound, &new_reshaped_tensor_arrays,
                                 &region_ta_indices, &op_inputs, &old_ta_new_tensor_mapping);
  Array<Var> store_tas;
  {
    for (size_t i = 0; i < store->region_tas.size(); ++i) {
      auto var = store->region_tas[i];
      auto old_ta = declarations.get_tensor_array(var);
      bool can_reuse_reshaped = true;
      {
        if (!new_reshaped_tensor_arrays.count(old_ta)) {
          can_reuse_reshaped = false;
        } else {
          auto reshaped = new_reshaped_tensor_arrays.at(old_ta).as<RegionTensorArrayNode>();
          if (reshaped->shape.size() != region_ta_indices[i].size()) {
            can_reuse_reshaped = false;
          }
        }
      }

      if (can_reuse_reshaped) {
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

  // std::cout << "[TE] " << std::endl;

  // Create reshape statements now
  Array<Stmt> stmts;
  for (auto it : new_reshaped_tensor_arrays) {
    stmts.push_back(ReshapeTANode::make(it.second->ta_var, it.first->ta_var));
  }
  stmts.push_back(new_store);
  return SeqStmt(stmts);
}

Stmt LiftLoopToScanOp(TADeclarations declarations, Array<TensorArray>* p_scan_tensor_arrays,
                      const ForNode* loop, const ScanLiftingInfo* scan_info) {
  Array<TensorArray>& scan_tensor_arrays = *p_scan_tensor_arrays;
  auto init_store = scan_info->init_stmt.as<RegionTAStoreNode>();
  const TECapsuleNode* init_capsule = nullptr;
  if (scan_info->init_separate) {
    init_capsule = TECapsule::capsules.at(init_store->te_graph_name);
  }
  Array<Array<PrimExpr>> init_region_ta_indices;
  Map<TensorArray, TensorArray> init_new_reshaped_tensor_arrays;
  Array<PrimExpr> init_op_inputs;
  Map<TensorArray, te::Tensor> init_old_ta_new_tensor_mapping;

  auto update_store = scan_info->update_stmt.as<RegionTAStoreNode>();
  auto update_capsule = TECapsule::capsules.at(update_store->te_graph_name);
  Array<Array<PrimExpr>> update_region_ta_indices;
  Map<TensorArray, TensorArray> update_new_reshaped_tensor_arrays;
  Array<PrimExpr> update_op_inputs;
  Map<TensorArray, te::Tensor> update_old_ta_new_tensor_mapping;

  Dimension new_loop_dim =
      DimensionNode::make(loop->loop_var->name_hint + "_dim", DimensionNode::kRangeDim);

  OneStoreLifter lifter("scan");

  if (scan_info->init_separate) {
    std::cout << "[TE] New scan init lift " << std::endl;
    lifter.Lift(declarations, init_store, loop, {}, new_loop_dim,
                Range::make_by_min_extent(loop->min, scan_info->init_limit), NullValue<PrimExpr>(),
                &init_new_reshaped_tensor_arrays, &init_region_ta_indices, &init_op_inputs,
                &init_old_ta_new_tensor_mapping);
  }

  Array<TensorArray> tas_tensorize_only_one_dim(scan_tensor_arrays);
  for (auto it : scan_info->update_io_mapping) {
    tas_tensorize_only_one_dim.push_back(it.first);
  }
  std::cout << "[TE] New scan update lift " << std::endl;
  lifter.Lift(declarations, update_store, loop, tas_tensorize_only_one_dim, new_loop_dim,
              Range::make_by_min_extent(loop->min, loop->extent), loop->extent_upper_bound,
              &update_new_reshaped_tensor_arrays, &update_region_ta_indices, &update_op_inputs,
              &update_old_ta_new_tensor_mapping);

  // Scan tensors
  Array<te::Tensor> state_placeholder;
  Array<te::Tensor> init;
  Array<te::Tensor> update;

  // TE capsule fields
  Array<te::Tensor> non_external_inputs;
  Array<tir::Var> input_vars;
  Array<te::Tensor> inputs;
  Array<te::Tensor> outputs;
  PrimExpr loop_extent = NullValue<PrimExpr>();
  PrimExpr loop_min = NullValue<PrimExpr>();
  {
    CHECK_EQ(update_capsule->outputs.size(), 1);
    CHECK_EQ(!scan_info->init_separate || init_capsule->outputs.size(), 1);

    // State placeholder
    CHECK_EQ(scan_info->update_io_mapping.size(), 1);
    auto old_ta = (*scan_info->update_io_mapping.begin()).first;
    auto old_tensor = (*scan_info->update_io_mapping.begin()).second;
    auto state_it = update_old_ta_new_tensor_mapping.find(old_ta);
    CHECK(state_it != update_old_ta_new_tensor_mapping.end()) << old_ta;
    auto new_state = (*state_it).second->op.output(old_tensor->value_index);
    auto state_pl_op = new_state->op.as<PlaceholderOpNode>();
    state_placeholder.push_back(new_state);

    // Init
    auto new_init_tensor = NullValue<Tensor>();
    if (scan_info->init_separate) {
      new_init_tensor = init_capsule->outputs[0];
      if (init_capsule->inputs.size() == 0) {
        non_external_inputs.push_back(new_init_tensor);
      }
      std::cout << "[SCAN] New init " << new_init_tensor->op << std::endl;
    } else {
      new_init_tensor =
          PlaceholderOpNode::make("init", new_state->shape, new_state->dtype,
                                  state_pl_op->self_index_dimensions, state_pl_op->all_dimensions)
              .output(0);
      non_external_inputs.push_back(new_init_tensor);
    }
    init.push_back(new_init_tensor);

    // Update
    auto update_tensor = update_capsule->outputs[0];
    update.push_back(update_tensor);
    if (auto bvd_op = update_capsule->outputs[0]->op.as<BaseVarDimOpNode>()) {
      auto dom =
          bvd_op->GetIterVarFromDim(update_capsule->outputs[0]->value_index, new_loop_dim)->dom;
      loop_min = dom->min;
      loop_extent = dom->extent;
    }
  }

  CHECK(loop_min.defined() && loop_extent.defined());

  Operation scan = ScanOpNode::make(
      loop->loop_var->name_hint + "_scan", "", {}, UninterpFunNode::from_constant("min", loop_min),
      UninterpFunNode::from_constant("max", loop_min + loop_extent), new_loop_dim, false, init,
      update, state_placeholder, {}, {}, {}, Array<IterVar>());

  Array<Var> store_tas;
  {
    for (auto var : update_store->region_tas) {
      auto old_ta = declarations.get_tensor_array(var);
      TensorArray new_ta = NullValue<TensorArray>();
      if (update_new_reshaped_tensor_arrays.count(old_ta)) {
        new_ta = update_new_reshaped_tensor_arrays.at(old_ta);
      } else {
        auto old_rta = old_ta.as<RegionTensorArrayNode>();
        Array<PrimExpr> new_ta_shape;
        for (size_t i = 0; i < old_ta->shape.size() - 1; ++i) {
          new_ta_shape.push_back(old_ta->shape[i]);
        }
        Array<PrimExpr> new_tensor_shape;
        new_tensor_shape.push_back(old_ta->shape[old_rta->shape.size() - 1]);
        new_tensor_shape.push_back_all(old_rta->tensor_shape);

        new_ta = RegionTensorArrayNode::make(old_rta->ta_var.copy_with_suffix("_reshaped"),
                                             old_rta->dtype, new_ta_shape, new_tensor_shape,
                                             old_rta->name + "_reshaped", old_rta->base_region_ta);
        declarations.add_tensor_array(new_ta);
        update_new_reshaped_tensor_arrays.Set(old_ta, new_ta);
      }
      store_tas.push_back(new_ta->ta_var);
      scan_tensor_arrays.push_back(new_ta);
    }
  }

  {
    if (scan_info->init_separate) {
      input_vars.push_back_all(init_capsule->input_vars);
      inputs.push_back_all(init_capsule->inputs);
    }
    input_vars.push_back_all(update_capsule->input_vars);
    inputs.push_back_all(update_capsule->inputs);
    outputs.push_back(scan.output(0));
  }

  Array<PrimExpr> op_inputs;
  {
    if (scan_info->init_separate) {
      for (size_t i = 0; i < init_capsule->input_vars.size(); ++i) {
        op_inputs.push_back(init_op_inputs[i]);
      }
      for (size_t i = 0; i < update_capsule->input_vars.size(); ++i) {
        op_inputs.push_back(update_op_inputs[i]);
      }

      for (size_t i = 0; i < init_capsule->inputs.size(); ++i) {
        op_inputs.push_back(init_op_inputs[i + init_capsule->input_vars.size()]);
      }
      for (size_t i = 0; i < update_capsule->inputs.size(); ++i) {
        op_inputs.push_back(update_op_inputs[i + update_capsule->input_vars.size()]);
      }
    } else {
      op_inputs = update_op_inputs;
    }
  }

  // Remove duplicate inputs
  {
    CHECK_EQ(op_inputs.size(), inputs.size() + input_vars.size());
    Array<PrimExpr> deduped_op_inputs;
    Array<te::Tensor> deduped_inputs;
    Array<Var> deduped_input_vars;
    for (size_t i = 0; i < input_vars.size(); ++i) {
      if (!deduped_op_inputs.Contains(op_inputs[i])) {
        deduped_op_inputs.push_back(op_inputs[i]);
        deduped_input_vars.push_back(input_vars[i]);
      }
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!deduped_op_inputs.Contains(op_inputs[i])) {
        deduped_op_inputs.push_back(op_inputs[i]);
        deduped_inputs.push_back(inputs[i]);
      }
    }
  }

  // Update TECapsule
  {
    TECapsuleNode* mut_capsule = const_cast<TECapsuleNode*>(update_capsule);
    mut_capsule->input_vars = input_vars;
    mut_capsule->inputs = inputs;
    mut_capsule->non_external_inputs = non_external_inputs;
    mut_capsule->outputs = outputs;
    mut_capsule->all_ops_ = {};
  }

  update_capsule->RefreshAllOps(true);

  // std::cout << "[TE] all ops " << update_capsule->all_ops_ << std::endl;

  Stmt new_store =
      RegionTAStoreNode::make(store_tas, update_region_ta_indices, update_capsule->name, op_inputs,
                              update_store->direct_inputs);

  // Create reshape statements now
  Array<Stmt> stmts;
  for (auto it : update_new_reshaped_tensor_arrays) {
    stmts.push_back(ReshapeTANode::make(it.second->ta_var, it.first->ta_var));
  }
  stmts.push_back(new_store);
  Stmt ret = SeqStmt(stmts);
  return ret;
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
  LoopFinderAndLowerer(TADeclarations declarations_, Array<TensorArray>* p_scan_tensor_arrays)
      : declarations(declarations_), scan_tensor_arrays(*p_scan_tensor_arrays) {}

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

  Map<TensorArray, te::Tensor> GetScanIOMapping(const RegionTAStoreNode* store) {
    bool print = false;
    if (print) std::cout << "[TE] ScanIO" << std::endl;
    std::unordered_set<const Object*> stored_tas;
    for (auto region_ta : store->region_tas) {
      stored_tas.insert(declarations.get_tensor_array(region_ta).get());
      if (print) std::cout << "[TE]  StoredTA " << region_ta << std::endl;
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

      if (print) std::cout << "[TE]  LoadedTA " << loaded_var << std::endl;
      auto read_ta = declarations.get_tensor_array(loaded_var);
      if (stored_tas.count(read_ta.get())) {
        if (print) std::cout << "[TE]  State " << read_ta << std::endl;
        ret.Set(read_ta, capsule->inputs[i]);
      }
    }
    return ret;
  }

  const ScanLiftingInfo* CanBeLiftedToScanOp(const ForNode* loop) {
    bool print = false;
    if (print) std::cout << "[TE]  Check " << loop->loop_var << std::endl;
    Var loop_var = loop->loop_var;
    auto body_check = [&](Stmt body, bool check_update) {
      if (auto store = body.as<RegionTAStoreNode>()) {
        if (print) std::cout << "[TE]  Body of loop " << loop->body->GetTypeKey() << std::endl;
        bool last_index_is_loop_var = true;
        for (auto indices : store->region_ta_indices) {
          if (!indices[indices.size() - 1].same_as(loop_var)) {
            last_index_is_loop_var = false;
          }
        }

        bool outputs_are_inputs = true;
        std::unordered_set<const Object*> read_tas;
        for (auto input : store->inputs) {
          if (print) std::cout << "[TE]   Input " << input << std::endl;
          Var loaded_var;
          if (auto load = input.as<RegionTALoadNode>()) {
            loaded_var = load->region_ta;
          } else if (auto load = input.as<PointerTALoadNode>()) {
            loaded_var = load->pointer_ta;
          } else {
            if (print) std::cout << "[TE]    Unmatched Input" << std::endl;
            continue;
          }
          read_tas.insert(declarations.get_tensor_array(loaded_var)->GetBaseTensorArray().get());
        }

        for (auto region_ta : store->region_tas) {
          if (!read_tas.count(
                  declarations.get_tensor_array(region_ta)->GetBaseTensorArray().get())) {
            if (print)
              std::cout << "[TE]    Not found output " << region_ta << " " << region_ta.get()
                        << std::endl;
            outputs_are_inputs = false;
          }
        }

        bool output_input_valid = check_update ? outputs_are_inputs : !outputs_are_inputs;

        if (print)
          std::cout << "[TE]   " << last_index_is_loop_var << (store->direct_inputs.size() == 0)
                    << (loop->for_type == ForType::Sequential) << output_input_valid << std::endl;

        return last_index_is_loop_var && (store->direct_inputs.size() == 0) &&
               (loop->for_type == ForType::Sequential) && output_input_valid &&
               store->region_tas.size() == 1;
      }
      return false;
    };

    ScanLiftingInfo* info = nullptr;
    if (auto ite = loop->body.as<IfThenElseNode>()) {
      Stmt then_case = ite->then_case;
      Stmt else_case = ite->else_case;
      PrimExpr condition = ite->condition;
      if (!else_case.defined()) return nullptr;

      if (print) std::cout << "[TE]   Else defined" << std::endl;

      bool condition_valid = true;
      PrimExpr init_limit;
      {
        PrimExpr left;
        PrimExpr right;
        if (auto lt = condition.as<LTNode>()) {
          left = lt->a;
          right = lt->b;
        } else if (auto le = condition.as<LENode>()) {
          left = le->a;
          right = le->b + 1;
        } else {
          if (print)
            std::cout << "[TE]   Cond invalid1 " << condition << " " << condition->GetTypeKey()
                      << std::endl;
          condition_valid = false;
        }

        if (condition_valid) {
          if (left.same_as(loop_var)) {
            if (print) std::cout << "[TE] Init limit " << right << std::endl;
            init_limit = right;
          } else {
            if (print) std::cout << "[TE]   Cond invalid2 " << left << std::endl;
            condition_valid = false;
          }
        }
      }

      bool then_case_valid = body_check(then_case, false);
      bool else_case_valid = body_check(else_case, true);
      if (!else_case_valid || !then_case_valid) {
        return nullptr;
      }
      bool together_valid = true;
      auto then_store = then_case.as<RegionTAStoreNode>();
      auto else_store = else_case.as<RegionTAStoreNode>();
      if (then_case_valid && else_case_valid) {
        if (then_store->region_tas.size() != else_store->region_tas.size()) {
          together_valid = false;
        } else {
          for (size_t i = 0; i < then_store->region_tas.size(); ++i) {
            if (!then_store->region_tas[i].same_as(else_store->region_tas[i])) {
              together_valid = false;
              break;
            }
          }
        }
      }

      if (print) {
        std::cout << "[TE]   ThenBody " << then_case_valid << std::endl;
        std::cout << "[TE]   ElseBody " << else_case_valid << std::endl;
      }

      info = new ScanLiftingInfo();
      info->init_separate = true;
      info->init_stmt = then_case;
      info->update_stmt = else_case;
      info->init_limit = init_limit;
      info->update_io_mapping = GetScanIOMapping(else_store);
    } else if (auto store = loop->body.as<RegionTAStoreNode>()) {
      if (body_check(loop->body, true)) {
        info = new ScanLiftingInfo();
        info->init_separate = false;
        info->update_stmt = loop->body;
        info->update_io_mapping = GetScanIOMapping(store);
      }
    }
    return info;
  }

  Stmt VisitStmt_(const ForNode* loop) override {
    // std::cout << "[TE] Check if liftable " << loop->loop_var << std::endl;
    if (CanBeLiftedToComputeOp(loop)) {
      std::cout << "[TE] Lifting to compute " << loop->loop_var << std::endl;
      return LiftLoopToComputeOp(declarations, &scan_tensor_arrays, loop);
    } else if (auto info = CanBeLiftedToScanOp(loop)) {
      std::cout << "[TE] Lifting to scan " << loop->loop_var << std::endl;
      return LiftLoopToScanOp(declarations, &scan_tensor_arrays, loop, info);
    } else {
      return StmtExprMutator::VisitStmt_(loop);
    }
  }

 private:
  TADeclarations declarations;
  Array<TensorArray>& scan_tensor_arrays;
};

tir::Stmt lift_to_ter(TADeclarations declarations, const tir::Stmt& input_program,
                      const int iters) {
  check_ta_uses(declarations, input_program);

  // std::cout << "[TE] Lifting" << std::endl;
  Stmt stmt = input_program;
  Array<TensorArray> scan_tensor_arrays;
  for (size_t i = 0; i < iters; ++i) {
    LoopFinderAndLowerer lowerer(declarations, &scan_tensor_arrays);
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
