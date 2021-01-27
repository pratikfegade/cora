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
  TensorArrayLowerer(Map<tir::TensorArray, tir::Buffer> ta_buffers_,
                     Map<tir::Var, tir::TensorArray> var_ta_mapping_,
                     Map<tir::Var, tir::Buffer> var_buf_mapping_, const BuildConfig& build_config_)
      : ta_buffers(ta_buffers_),
        var_ta_mapping(var_ta_mapping_),
        var_buf_mapping(var_buf_mapping_),
        build_config(build_config_) {}

  Stmt VisitStmt_(const PointerTAAllocateNode* alloc) override {
    CHECK(var_ta_mapping.count(alloc->pointer_ta_var));
    auto pointer_ta = var_ta_mapping.at(alloc->pointer_ta_var);
    CHECK(ta_buffers.count(pointer_ta));
    Buffer buf = ta_buffers.at(pointer_ta);

    Stmt ret = AllocateNode::make(buf->data, buf->dtype, buf->shape, IntImm(DataType::Bool(), 1),
                                  this->VisitStmt(alloc->body));
    ret = AttrStmtNode::make(buf->data, attr::storage_scope, StringImmNode::make(buf->scope), ret);
    return ret;
  }

  Stmt VisitStmt_(const RegionTAAllocateNode* alloc) override {
    CHECK(var_ta_mapping.count(alloc->region_ta_var));
    auto region_ta = var_ta_mapping.at(alloc->region_ta_var);
    CHECK(ta_buffers.count(region_ta));
    Buffer buf = ta_buffers.at(region_ta);

    Stmt ret = AllocateNode::make(buf->data, buf->dtype, buf->shape, IntImm(DataType::Bool(), 1),
                                  this->VisitStmt(alloc->body));
    ret = AttrStmtNode::make(buf->data, attr::storage_scope, StringImmNode::make(buf->scope), ret);
    return ret;
  }

  Stmt VisitStmt_(const AllocateNode* alloc) override {
    if (!buffer_attrs.count(alloc->buffer_var.get())) {
      CHECK(var_buf_mapping.count(alloc->buffer_var))
          << alloc->buffer_var << " " << alloc->buffer_var.get();
      return AttrStmtNode::make(alloc->buffer_var, attr::storage_scope,
                                StringImmNode::make(var_buf_mapping.at(alloc->buffer_var)->scope),
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
    CHECK(var_ta_mapping.count(store->pointer_ta));
    auto pointer_ta = var_ta_mapping.at(store->pointer_ta);
    auto pointer_tan = pointer_ta.as<PointerTensorArrayNode>();
    CHECK(pointer_tan) << store->pointer_ta << " " << pointer_ta;
    CHECK(ta_buffers.count(pointer_ta));
    Buffer buf = ta_buffers.at(var_ta_mapping.at(store->pointer_ta));

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
    auto region_tan = var_ta_mapping.at(load->region_ta).as<RegionTensorArrayNode>();
    CHECK(region_tan) << GetRef<PrimExpr>(load) << " " << load->region_ta;
    CHECK(region_tan->tensor_shape.size() == 0);
    auto region_ta = GetRef<TensorArray>(region_tan);
    CHECK(ta_buffers.count(region_ta));
    Buffer buf = ta_buffers.at(region_ta);

    return LoadNode::make(region_tan->dtype, buf->data,
                          GenerateIndex(region_ta->shape, load->indices),
                          IntImm(DataType::Bool(), 1), kAll);
  }

  Stmt VisitStmt_(const ReshapeTANode* store) override { return EvaluateNode::make(0); }

  Stmt VisitStmt_(const RegionTAStoreNode* store) override {
    std::cout << "[LOW] Lowering store " << GetRef<Stmt>(store) << std::endl;
    if (store->direct_inputs.defined() && store->direct_inputs.size() > 0) {
      std::cout << "[LOW]   Direct inputs present" << std::endl;
      for (auto region_ta : store->region_tas) {
        auto region_tan = var_ta_mapping.at(region_ta).as<RegionTensorArrayNode>();
        CHECK(region_tan->shape.size() == 1) << GetRef<Stmt>(store) << " " << region_ta;
      }

      Array<Stmt> stmts;
      CHECK_EQ(store->direct_inputs.size(), store->region_tas.size());
      for (size_t i = 0; i < store->direct_inputs.size(); ++i) {
        auto ta = var_ta_mapping.at(store->region_tas[i]);
        auto buf = ta_buffers.at(ta);
        stmts.push_back(StoreNode::make(buf->data, store->direct_inputs[i],
                                        GenerateIndex(ta->shape, store->region_ta_indices[i]),
                                        IntImm(DataType::Bool(), 1), kAll));
      }
      return SeqStmt(stmts);
    } else {
      for (auto region_ta : store->region_tas) {
        auto region_tan = var_ta_mapping.at(region_ta).as<RegionTensorArrayNode>();
        CHECK(region_tan) << GetRef<Stmt>(store) << " " << region_ta;
      }

      CHECK(TECapsule::capsules.count(store->te_graph_name));
      TECapsule te_capsule = GetRef<TECapsule>(TECapsule::capsules.at(store->te_graph_name));
      Array<PrimExpr> inputs = store->inputs;

      CHECK_EQ(inputs.size(), te_capsule->input_vars.size() + te_capsule->inputs.size())
          << inputs.size() << " " << te_capsule->input_vars.size() << " "
          << te_capsule->inputs.size();

      for (size_t i = 0; i < te_capsule->input_vars.size(); ++i) {
      }

      Map<te::Tensor, Buffer> buf_bindings;
      Map<te::Tensor, Buffer> partial_buf_bindings;
      Map<te::Tensor, Array<PrimExpr>> partial_index_bindings;
      for (size_t i = 0; i < te_capsule->inputs.size(); ++i) {
        te::Tensor input_tensor = te_capsule->inputs[i];
        PrimExpr input = store->inputs[i + te_capsule->input_vars.size()];
        if (input.as<VarNode>()) {
          Var var = Downcast<Var>(input);
          CHECK(var_buf_mapping.count(var)) << var << " " << var.get();
          buf_bindings.Set(input_tensor, var_buf_mapping.at(var));
        } else if (auto load = input.as<RegionTALoadNode>()) {
          CHECK(var_ta_mapping.count(load->region_ta));
          CHECK(ta_buffers.count(var_ta_mapping.at(load->region_ta)));
          Buffer buf = ta_buffers.at(var_ta_mapping.at(load->region_ta));
          Array<PrimExpr> load_indices;
          for (auto index : load->indices) {
            load_indices.push_back(this->VisitExpr(index));
          }
          partial_buf_bindings.Set(input_tensor, buf);
          partial_index_bindings.Set(input_tensor, load_indices);
        } else if (auto load = input.as<PointerTALoadNode>()) {
          Buffer pta_buf = ta_buffers.at(var_ta_mapping.at(load->pointer_ta));
          auto pta = var_ta_mapping.at(load->pointer_ta);
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
          partial_buf_bindings.Set(input_tensor, rta_buf);
          partial_index_bindings.Set(input_tensor, load_indices);
        } else {
          CHECK(false) << "This case is not supported yet for store inputs " << input;
        }
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
          Buffer buf = ta_buffers.at(var_ta_mapping.at(region_ta));
          partial_buf_bindings.Set(output_tensor, buf);
          partial_index_bindings.Set(output_tensor, indices);
        }
      }
      tir::Stmt lowered_op = te_capsule->LowerToTIR(build_config, buf_bindings,
                                                    partial_buf_bindings, partial_index_bindings);

      {
        TECapsule te_capsule = GetRef<TECapsule>(TECapsule::capsules.at(store->te_graph_name));

        // std::cout << "[LOW] Generated TIR\n"
        //           << lowered_op << " " << te_capsule->input_vars.size() << " "
        //           << te_capsule->inputs.size() << " " << te_capsule << " "
        //           << te_capsule->schedule.defined() << std::endl;
      }
      // return GetRef<Stmt>(store);
      return lowered_op;
    }
  }

 private:
  Map<tir::TensorArray, tir::Buffer> ta_buffers;
  Map<tir::Var, tir::TensorArray> var_ta_mapping;
  Map<tir::Var, tir::Buffer> var_buf_mapping;
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

tir::Stmt lower_tensor_arrays(const Array<tir::TensorArray> tensor_arrays,
                              const Array<tir::Buffer> buffers, const tir::Stmt& input_program,
                              const Target& target_host, const BuildConfig& config) {
  bool loop_partition = true;

  // Contruct buffers for all tensor arrays
  Map<tir::TensorArray, tir::Buffer> ta_buffers;
  Map<tir::Var, tir::TensorArray> var_ta_mapping;
  for (auto ta : tensor_arrays) {
    Array<PrimExpr> buffer_shape;
    DataType buffer_dtype;
    std::string buffer_name = ta->name + "_buf";
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

    ta_buffers.Set(ta, tir::decl_buffer(buffer_shape, buffer_dtype, buffer_name));
    var_ta_mapping.Set(ta->ta_var, ta);
  }

  Map<tir::Var, tir::Buffer> var_buf_mapping;
  for (auto buf : buffers) {
    var_buf_mapping.Set(buf->data, buf);
    std::cout << "[LOW] Var-Buffer Mapping " << buf->data << " " << buf->data.get() << std::endl;
  }

  std::unordered_map<const tir::VarNode*, PrimExpr> reshaped_base_mapping;
  for (auto ta : tensor_arrays) {
    Buffer base = ta_buffers.at(ta->GetBaseTensorArray());
    Buffer buf = ta_buffers.at(ta);
    reshaped_base_mapping[buf->data.as<VarNode>()] = base->data;
    std::cout << "[LOW] Base Mapping " << buf->data << " " << buf->data.get() << " " << base->data
              << std::endl;
  }

  // lower to TIR
  TensorArrayLowerer lowerer(ta_buffers, var_ta_mapping, var_buf_mapping, config);
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
  return stmt;
}

TVM_REGISTER_GLOBAL("tir.lower_tensor_arrays").set_body_typed(lower_tensor_arrays);

}  // namespace tvm
