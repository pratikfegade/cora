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
#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/ta_declarations.h>
#include <tvm/tir/te_capsule.h>

#include <algorithm>
#include <mutex>
#include <stack>

#include "../tir/ir/var_replacer.h"

namespace tvm {
using namespace tvm::tir;

PrimExpr GenerateIndex(Array<PrimExpr> shape, Array<PrimExpr> indices) {
  PrimExpr index = IntImm(DataType::Int(32), 0);
  CHECK_EQ(shape.size(), indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    index = indices[i] + index * shape[i];
  }
  return index;
}

class TensorArrayLowerer : public tir::StmtExprMutator {
 public:
  TensorArrayLowerer(Map<tir::TensorArray, tir::Buffer> ta_buffers_, TADeclarations declarations_,
                     const BuildConfig& build_config_)
      : ta_buffers(ta_buffers_), declarations(declarations_), build_config(build_config_) {}

 private:
  Stmt VisitStmt_(const PointerTAAllocateNode* alloc) override {
    auto pointer_ta = declarations.get_tensor_array(alloc->pointer_ta_var);
    CHECK(ta_buffers.count(pointer_ta));
    Buffer buf = ta_buffers.at(pointer_ta);

    Stmt ret = AllocateNode::make(buf->data, buf->dtype, buf->shape, IntImm(DataType::Bool(), 1),
                                  this->VisitStmt(alloc->body));
    ret = AttrStmtNode::make(buf->data, attr::storage_scope, StringImmNode::make(buf->scope), ret);
    return ret;
  }

  Stmt VisitStmt_(const RegionTAAllocateNode* alloc) override {
    auto region_ta = declarations.get_tensor_array(alloc->region_ta_var);
    CHECK(ta_buffers.count(region_ta));
    Buffer buf = ta_buffers.at(region_ta);

    Stmt ret = AllocateNode::make(buf->data, buf->dtype, buf->shape, IntImm(DataType::Bool(), 1),
                                  this->VisitStmt(alloc->body));
    ret = AttrStmtNode::make(buf->data, attr::storage_scope, StringImmNode::make(buf->scope), ret);
    return ret;
  }

  Stmt VisitStmt_(const AllocateNode* alloc) override {
    if (!buffer_attrs.count(alloc->buffer_var.get())) {
      return AttrStmtNode::make(
          alloc->buffer_var, attr::storage_scope,
          StringImmNode::make(declarations.get_buffer(alloc->buffer_var)->scope),
          StmtExprMutator::VisitStmt_(alloc));
    }
    return StmtExprMutator::VisitStmt_(alloc);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == attr::storage_scope) {
      const VarNode* buf = op->node.as<VarNode>();
      buffer_attrs.insert(buf);
      Stmt new_body = StmtExprMutator::VisitStmt(op->body);
      buffer_attrs.erase(buf);
      return AttrStmtNode::make(op->node, attr::storage_scope,
                                StmtExprMutator::VisitExpr(op->value), new_body);
    } else if (op->attr_key == attr::thread_extent) {
      CHECK(op->node.as<IterVarNode>());
      auto thread_iv = Downcast<IterVar>(op->node);
      thread_extent_stack.push_back(thread_iv);
      Stmt new_body = StmtExprMutator::VisitStmt(op->body);
      thread_extent_stack.resize(thread_extent_stack.size() - 1);
      return AttrStmtNode::make(op->node, attr::thread_extent,
                                StmtExprMutator::VisitExpr(op->value), new_body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const PointerTAStoreNode* store) override {
    auto pointer_ta = declarations.get_tensor_array(store->pointer_ta);
    auto pointer_tan = pointer_ta.as<PointerTensorArrayNode>();
    CHECK(pointer_tan) << store->pointer_ta << " " << pointer_ta;
    CHECK(ta_buffers.count(pointer_ta));
    Buffer buf = ta_buffers.at(declarations.get_tensor_array(store->pointer_ta));

    auto region_ta = pointer_tan->base_region_ta.as<RegionTensorArrayNode>();
    CHECK(region_ta);

    CHECK_EQ(buf->shape.size(), store->pointer_ta_indices.size() + 1);
    CHECK_EQ(region_ta->shape.size(), store->region_ta_indices.size());

    PrimExpr start_idx = GenerateIndex(pointer_ta->shape, store->pointer_ta_indices);
    Array<Stmt> stores;
    for (size_t i = 0; i < region_ta->shape.size(); ++i) {
      stores.push_back(StoreNode::make(buf->data, this->VisitExpr(store->region_ta_indices[i]),
                                       AddNode::make(start_idx, IntImm(DataType::Int(32), i)),
                                       IntImm(DataType::Bool(), 1), kAll));
    }

    return SeqStmt(stores);
  }

  PrimExpr VisitExpr_(const RegionTALoadNode* load) override {
    auto region_tan = declarations.get_tensor_array(load->region_ta).as<RegionTensorArrayNode>();
    CHECK(region_tan) << GetRef<PrimExpr>(load) << " " << load->region_ta;
    CHECK(region_tan->tensor_shape.size() == 0) << GetRef<PrimExpr>(load);
    auto region_ta = GetRef<TensorArray>(region_tan);
    CHECK(ta_buffers.count(region_ta));
    Buffer buf = ta_buffers.at(region_ta);

    return LoadNode::make(region_tan->dtype, buf->data,
                          GenerateIndex(region_ta->shape, load->indices),
                          IntImm(DataType::Bool(), 1), kAll);
  }

  Stmt VisitStmt_(const ReshapeTANode* store) override { return EvaluateNode::make(0); }

  void handle_input(ObjectRef parameter, PrimExpr input_expr,
                    Map<te::Tensor, Buffer>* p_buf_bindings,
                    Map<ObjectRef, Buffer>* p_partial_buf_bindings,
                    Map<ObjectRef, Array<PrimExpr>>* p_partial_index_bindings,
                    std::unordered_map<const VarNode*, PrimExpr>* p_input_var_arguments) {
    Map<te::Tensor, Buffer>& buf_bindings = *p_buf_bindings;
    Map<ObjectRef, Buffer>& partial_buf_bindings = *p_partial_buf_bindings;
    Map<ObjectRef, Array<PrimExpr>>& partial_index_bindings = *p_partial_index_bindings;
    std::unordered_map<const VarNode*, PrimExpr>& input_var_arguments = *p_input_var_arguments;

    // std::cout << "[LOW]  Input " << parameter << " " << input_expr << std::endl;
    if (input_expr.as<VarNode>()) {
      Var var = Downcast<Var>(input_expr);
      if (parameter.as<te::TensorNode>()) {
        buf_bindings.Set(Downcast<te::Tensor>(parameter), declarations.get_buffer(var));
      } else if (auto varnode = parameter.as<VarNode>()) {
        input_var_arguments[varnode] = input_expr;
      } else {
        CHECK(false) << parameter;
      }
    } else if (input_expr.as<IntImmNode>() || input_expr.as<FloatImmNode>()) {
      auto varnode = parameter.as<VarNode>();
      CHECK(varnode) << "Passing scalar arguments for tensors " << parameter << " " << input_expr;
      input_var_arguments[varnode] = input_expr;
    } else if (auto load = input_expr.as<RegionTALoadNode>()) {
      CHECK(ta_buffers.count(declarations.get_tensor_array(load->region_ta)));
      Buffer buf = ta_buffers.at(declarations.get_tensor_array(load->region_ta));
      Array<PrimExpr> load_indices;
      for (auto index : load->indices) {
        load_indices.push_back(this->VisitExpr(index));
      }
      partial_buf_bindings.Set(parameter, buf);
      partial_index_bindings.Set(parameter, load_indices);
    } else if (auto load = input_expr.as<PointerTALoadNode>()) {
      Buffer pta_buf = ta_buffers.at(declarations.get_tensor_array(load->pointer_ta));
      auto pta = declarations.get_tensor_array(load->pointer_ta);
      auto ptan = pta.as<PointerTensorArrayNode>();
      auto rta = ptan->base_region_ta;
      Buffer rta_buf = ta_buffers.at(rta);

      Array<PrimExpr> load_indices;
      {
        PrimExpr start_idx = IntImm(DataType::Int(32), 0);
        for (size_t i = 0; i < load->indices.size(); ++i) {
          start_idx = AddNode::make(
              start_idx, MulNode::make(this->VisitExpr(load->indices[i]), pta->shape[i]));
        }

        for (size_t i = 0; i < rta->shape.size(); ++i) {
          load_indices.push_back(
              LoadNode::make(DataType::Int(32), pta_buf->data,
                             AddNode::make(start_idx, IntImm(DataType::Int(32), i)),
                             IntImm(DataType::Bool(), 1), kAll));
        }
      }
      partial_buf_bindings.Set(parameter, rta_buf);
      partial_index_bindings.Set(parameter, load_indices);
    } else {
      CHECK(false) << "This case is not supported yet for store inputs " << input_expr;
    }
  }

  Stmt VisitStmt_(const RegionTAStoreNode* store) override {
    // std::cout << "[LOW] Lowering store " << GetRef<Stmt>(store);
    if (store->direct_inputs.defined() && store->direct_inputs.size() > 0) {
      // std::cout << "[LOW]   Direct inputs present" << std::endl;
      for (auto region_ta : store->region_tas) {
        auto region_tan = declarations.get_tensor_array(region_ta).as<RegionTensorArrayNode>();
        CHECK(region_tan->shape.size() == 1) << GetRef<Stmt>(store) << " " << region_ta;
      }

      Array<Stmt> stmts;
      CHECK_EQ(store->direct_inputs.size(), store->region_tas.size());
      for (size_t i = 0; i < store->direct_inputs.size(); ++i) {
        auto ta = declarations.get_tensor_array(store->region_tas[i]);
        auto buf = ta_buffers.at(ta);
        Array<PrimExpr> indices;
        for (auto index : store->region_ta_indices[i]) {
          indices.push_back(this->VisitExpr(index));
        }

        PrimExpr global_index = GenerateIndex(ta->shape, indices);
        PrimExpr local_index = global_index;
        PrimExpr condition = IntImm(DataType::Bool(), 1);
        if (declarations->ta_layouts.count(ta)) {
          auto layout = declarations->ta_layouts.at(ta);
          CHECK_EQ(layout->layout.size(), 1);
          Range r = layout->layout[0];
          condition = (global_index >= r->min) && (global_index >= (r->min + r->extent));
          local_index = indexmod(global_index, r->extent);
        }

        stmts.push_back(
            StoreNode::make(buf->data, store->direct_inputs[i], local_index, condition, kAll));
      }
      return SeqStmt(stmts);
    } else {
      for (auto region_ta : store->region_tas) {
        auto region_tan = declarations.get_tensor_array(region_ta).as<RegionTensorArrayNode>();
        CHECK(region_tan) << GetRef<Stmt>(store) << " " << region_ta;
      }

      CHECK(TECapsule::capsules.count(store->te_graph_name));
      TECapsule te_capsule = GetRef<TECapsule>(TECapsule::capsules.at(store->te_graph_name));
      Array<PrimExpr> inputs = store->inputs;

      CHECK_EQ(inputs.size(), te_capsule->input_vars.size() + te_capsule->inputs.size())
          << inputs.size() << " " << te_capsule->input_vars.size() << " "
          << te_capsule->inputs.size();

      Map<te::Tensor, Buffer> buf_bindings;
      Map<ObjectRef, Buffer> partial_buf_bindings;
      Map<ObjectRef, Array<PrimExpr>> partial_index_bindings;
      std::unordered_map<const VarNode*, PrimExpr> input_var_arguments;
      for (size_t i = 0; i < te_capsule->input_vars.size(); ++i) {
        handle_input(te_capsule->input_vars[i], store->inputs[i], &buf_bindings,
                     &partial_buf_bindings, &partial_index_bindings, &input_var_arguments);
      }

      for (size_t i = 0; i < te_capsule->inputs.size(); ++i) {
        handle_input(te_capsule->inputs[i], store->inputs[i + te_capsule->input_vars.size()],
                     &buf_bindings, &partial_buf_bindings, &partial_index_bindings,
                     &input_var_arguments);
      }

      // Scheduling can change TECapsule outputs due to calls to
      // single_kernel. So we perform it before we generate mappings for
      // outputs below.
      if (thread_extent_stack.size() > 0) {
        te_capsule = te_capsule->ScheduleToTIR(thread_extent_stack);
      }

      {
        CHECK_EQ(te_capsule->outputs.size(), store->region_tas.size());
        for (size_t i = 0; i < store->region_tas.size(); ++i) {
          Array<PrimExpr> indices;
          for (auto index : store->region_ta_indices[i]) {
            indices.push_back(this->VisitExpr(index));
          }

          auto region_ta = store->region_tas[i];
          te::Tensor output_tensor = te_capsule->outputs[i];
          // std::cout << "[LOW]  Output " << output_tensor << " " << region_ta << std::endl;
          Buffer buf = ta_buffers.at(declarations.get_tensor_array(region_ta));
          partial_buf_bindings.Set(output_tensor, buf);
          partial_index_bindings.Set(output_tensor, indices);
        }
      }

      Map<te::Tensor, Array<Range>> interface_bounds;
      {
        for (size_t i = 0; i < te_capsule->inputs.size(); ++i) {
          auto input_tensor = te_capsule->inputs[i];
          auto input = store->inputs[te_capsule->input_vars.size() + i];
          if (auto load = input.as<RegionTALoadNode>()) {
            auto loaded_ta = declarations.get_tensor_array(load->region_ta);
            if (declarations->ta_layouts.count(loaded_ta)) {
              auto ta_layout = declarations->ta_layouts.at(loaded_ta);
              Array<Range> tensor_bounds;
              for (size_t j = 0; j < input_tensor.ndim(); ++j) {
                tensor_bounds.push_back(ta_layout->layout[j + load->indices.size()]);
              }
              interface_bounds.Set(input_tensor, tensor_bounds);
            }
          }
        }

        for (size_t i = 0; i < te_capsule->outputs.size(); ++i) {
          auto output_tensor = te_capsule->outputs[i];
          auto stored_ta = declarations.get_tensor_array(store->region_tas[i]);
          if (declarations->ta_layouts.count(stored_ta)) {
            auto ta_layout = declarations->ta_layouts.at(stored_ta);
            Array<Range> tensor_bounds;
            for (size_t j = 0; j < output_tensor.ndim(); ++j) {
              tensor_bounds.push_back(ta_layout->layout[j + store->region_ta_indices[i].size()]);
            }
            interface_bounds.Set(output_tensor, tensor_bounds);
          }
        }
      }

      tir::Stmt lowered_op =
          te_capsule->LowerToTIR(build_config, buf_bindings, partial_buf_bindings,
                                 partial_index_bindings, interface_bounds);

      VarReplacer replacer(input_var_arguments);
      return replacer(lowered_op);
    }
  }

  Map<tir::TensorArray, tir::Buffer> ta_buffers;
  TADeclarations declarations;
  const BuildConfig& build_config;
  std::unordered_set<const Object*> buffer_attrs;
  Array<tir::IterVar> thread_extent_stack;
};

bool isCudaThread(const std::string& name) {
  return name == "blockIdx.x" || name == "blockIdx.y" || name == "blockIdx.z" ||
         name == "threadIdx.x" || name == "threadIdx.y" || name == "threadIdx.z";
}

bool isCudaThread(const IterVar& iv) { return isCudaThread(iv->var->name_hint); }

bool isCPUEnvThread(const std::string& name) {
  return name.find("cpu_par_thread") != std::string::npos;
}

bool isCPUEnvThread(const IterVar& iv) { return isCPUEnvThread(iv->var->name_hint); }

bool equalCudaThreads(const IterVar& iv1, const IterVar& iv2) {
  return iv1->var->name_hint == iv2->var->name_hint && isCudaThread(iv1->var->name_hint);
}

class EnvThreadReplacer : public StmtExprMutator {
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      // delete duplicated thread extent attr
      IterVar thread = Downcast<IterVar>(op->node);
      std::string name = thread->var->name_hint;
      if (isCudaThread(thread) || isCPUEnvThread(thread)) {
        if (!env_thread_map.count(name)) {
          env_thread_map[name] = thread->var;
          Stmt body = StmtExprMutator::VisitStmt(op->body);
          env_thread_map.erase(name);
          return AttrStmtNode::make(op->node, op->attr_key, op->value, body);
        } else {
          return StmtExprMutator::VisitStmt(op->body);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) {
    if (env_thread_map.count(op->name_hint)) {
      return env_thread_map.at(op->name_hint);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  std::unordered_map<std::string, Var> env_thread_map;
};

class LoweringChecker : public StmtExprVisitor {
 public:
  bool check(Stmt stmt) {
    this->VisitStmt(stmt);
    return !error;
  }

 private:
  void VisitStmt_(const RegionTAStoreNode* op) override {
    std::cout << "[Lowering] Unlowered TAStore " << GetRef<Stmt>(op) << std::endl;
    error = true;
  }

  void VisitStmt_(const PointerTAStoreNode* op) override {
    std::cout << "[Lowering] Unlowered TAStore " << GetRef<Stmt>(op) << std::endl;
    error = true;
  }

  void VisitStmt_(const RegionTAAllocateNode* op) override {
    std::cout << "[Lowering] Unlowered TAAllocate " << GetRef<Stmt>(op) << std::endl;
    error = true;
  }

  void VisitStmt_(const PointerTAAllocateNode* op) override {
    std::cout << "[Lowering] Unlowered TAAllocate " << GetRef<Stmt>(op) << std::endl;
    error = true;
  }

  void VisitStmt_(const ReshapeTANode* op) override {
    std::cout << "[Lowering] Unlowered TAReshape " << GetRef<Stmt>(op) << std::endl;
    error = true;
  }

  void VisitExpr_(const RegionTALoadNode* op) override {
    std::cout << "[Lowering] Unlowered TALoad " << GetRef<PrimExpr>(op) << std::endl;
    error = true;
  }

  void VisitExpr_(const PointerTALoadNode* op) override {
    std::cout << "[Lowering] Unlowered TALoad " << GetRef<PrimExpr>(op) << std::endl;
    error = true;
  }

  bool error = false;
};

class GlobalAllocationHoister : public StmtMutator {
 public:
  Stmt hoist_global_allocates(Stmt stmt) {
    Stmt body = this->VisitStmt(stmt);

    for (auto alloc : to_hoist_allocates) {
      body = AttrStmtNode::make(
          alloc->buffer_var, attr::storage_scope, StringImmNode::make("global"),
          AllocateNode::make(alloc->buffer_var, alloc->dtype, alloc->extents, alloc->condition,
                             body, alloc->new_expr, alloc->free_function));
    }
    return body;
  }

 private:
  Stmt VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == attr::thread_extent) {
      num_thread_extents++;
      Stmt ret = StmtMutator::VisitStmt_(op);
      num_thread_extents--;
      return ret;
    } else if (op->attr_key == attr::storage_scope) {
      if (op->value.as<StringImmNode>()->value == "global" && num_thread_extents > 0) {
        // We have a global allocation in thread scope that we now
        // need to hoist
        to_hoist_buffers.insert(op->node.as<VarNode>());
        return StmtMutator::VisitStmt(op->body);
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateNode* op) override {
    if (to_hoist_buffers.count(op->buffer_var.as<VarNode>())) {
      to_hoist_allocates.insert(op);
      return StmtMutator::VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }

  std::unordered_set<const VarNode*> to_hoist_buffers;
  std::unordered_set<const AllocateNode*> to_hoist_allocates;
  int num_thread_extents = 0;
};

Array<LoweredFunc> lower_tensor_arrays(const TADeclarations& declarations,
                                       const Array<ObjectRef>& input_arguments,
                                       const tir::Stmt& input_program, const Target& target_host,
                                       const BuildConfig& config, bool print_body) {
  bool loop_partition = true;

  // Contruct buffers for all tensor arrays
  Map<tir::TensorArray, tir::Buffer> ta_buffers;
  Map<tir::Var, tir::TensorArray> var_ta_mapping;
  for (auto ta : declarations.get_all_tensor_arrays()) {
    Array<PrimExpr> buffer_shape;
    DataType buffer_dtype;
    std::string buffer_name = ta->name + "_buf";
    std::string storage_scope = "global";

    if (declarations->ta_layouts.count(ta)) {
      TALayout layout = declarations->ta_layouts.at(ta);
      for (auto r : layout->layout) {
        buffer_shape.push_back(r->extent);
      }
      storage_scope = layout->storage_scope;
    } else {
      if (auto rta = ta.as<RegionTensorArrayNode>()) {
        for (auto dim : rta->shape) {
          buffer_shape.push_back(dim);
        }
        for (auto dim : rta->tensor_shape) {
          buffer_shape.push_back(dim);
        }
        buffer_dtype = rta->dtype;
      } else if (auto pta = ta.as<PointerTensorArrayNode>()) {
        auto rta = pta->base_region_ta;
        for (auto dim : pta->shape) {
          buffer_shape.push_back(dim);
        }
        buffer_shape.push_back(IntImm(DataType::Int(32), rta->shape.size()));
        buffer_dtype = DataType::Int(32);
      }
    }

    if (auto rta = ta.as<RegionTensorArrayNode>()) {
      buffer_dtype = rta->dtype;
    } else if (auto pta = ta.as<PointerTensorArrayNode>()) {
      buffer_dtype = DataType::Int(32);
    }

    // for (auto it : buffer_shape) {
    // std::cout << "[LOW] TABufferShape " << it << std::endl;
    // }

    Buffer buffer = BufferNode::make(Var(buffer_name, DataType::Handle()), buffer_dtype,
                                     buffer_shape, {}, Array<PrimExpr>(), PrimExpr(), buffer_name,
                                     storage_scope, 0, 0, kDefault, kAll);
    // std::cout << "[LOW] TABuffers " << ta << " " << buffer << std::endl;
    ta_buffers.Set(ta, buffer);
  }

  std::unordered_map<const tir::VarNode*, PrimExpr> reshaped_base_mapping;
  for (auto ta : declarations.get_all_tensor_arrays()) {
    Buffer base = ta_buffers.at(ta->GetBaseTensorArray());
    Buffer buf = ta_buffers.at(ta);
    reshaped_base_mapping[buf->data.as<VarNode>()] = base->data;
    // std::cout << "[LOW] Base Mapping " << buf->data << " " << buf->data.get() << " " <<
    // base->data
    // << std::endl;
  }

  // lower to TIR
  TensorArrayLowerer lowerer(ta_buffers, declarations, config);
  Stmt stmt = lowerer(input_program);

  // Replace reshaped tensor buffers by their base buffers

  VarReplacer replacer(reshaped_base_mapping);
  stmt = replacer(stmt);

  // std::cout << "[LOW] Aggregated TIR\n" << stmt << std::endl;

  // Replace inner thread vars by outer ones
  EnvThreadReplacer env_replace;
  stmt = env_replace(stmt);

  // Perform further lowering
  {
    if (loop_partition) {
      stmt = tir::LoopPartition(stmt, config->partition_const_loop);
    }
    if (config->disable_vectorize) {
      stmt = tir::SkipVectorize(stmt);
    } else {
      stmt = tir::VectorizeLoop(stmt);
    }
    stmt = tir::InjectVirtualThread(stmt);
    stmt = tir::InjectDoubleBuffer(stmt, config->double_buffer_split_loop);
    stmt = tir::StorageRewrite(stmt);
    stmt = tir::UnrollLoop(stmt, config->auto_unroll_max_step, config->auto_unroll_max_depth,
                           config->auto_unroll_max_extent, config->unroll_explicit);

    // Phase 2
    stmt = tir::Simplify(stmt);
    stmt = tir::RemoveNoOp(stmt);

    if (!(config->disable_select_rewriting)) stmt = tir::RewriteUnsafeSelect(stmt);

    if (config->instrument_bound_checkers) stmt = tir::InstrumentBoundCheckers(stmt);
  }

  Array<ObjectRef> out_arg_list;
  for (auto inp : input_arguments) {
    if (inp.as<VarNode>()) {
      out_arg_list.push_back(inp);
    } else if (inp.as<BufferNode>()) {
      out_arg_list.push_back(inp);
    } else if (inp.as<TensorArrayNode>()) {
      out_arg_list.push_back(ta_buffers.at(Downcast<TensorArray>(inp)));
    } else {
      CHECK(false) << "Only TensorArrays, Buffers or Vars allowed as inputs. But instead we found "
                   << inp;
    }
  }

  // We need to hoist all all global intermediate tensor allocates
  // above any thread extent begins as we cannot allocate global
  // memory in CUDA and the codegen will choke.
  stmt = GlobalAllocationHoister().hoist_global_allocates(stmt);

  if (print_body) {
    std::cout << "[TE] Making func of " << stmt << std::endl;
  }

  // Ensure that we don't have any high level tensor array ops in the IR at this point
  LoweringChecker lowering_checker;
  CHECK(lowering_checker.check(stmt));

  auto funcs = Array<LoweredFunc>({tir::MakeAPI(stmt, "func0", out_arg_list, 0, true)});

  return funcs;
}

TVM_REGISTER_GLOBAL("tir.lower_tensor_arrays").set_body_typed(lower_tensor_arrays);

TVM_DLL runtime::Module build_tensor_arrays(const Array<tir::LoweredFunc>& funcs,
                                            const Target& target, const Target& target_host,
                                            const BuildConfig& config) {
  return build(funcs, target, target_host, config);
}

TVM_REGISTER_GLOBAL("tir.build_tensor_arrays").set_body_typed(build_tensor_arrays);

}  // namespace tvm
