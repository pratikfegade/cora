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
 * \file storage_flatten.cc
 */
// Flattens storage from multi-dimensional array to 1D
// buffer access as in Halide pipeline.
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/target/target_info.h>
#include <tvm/te/operation.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>

#include "../../arith/compute_expr.h"
#include "../../arith/ir_visitor_with_analyzer.h"
#include "../../runtime/thread_storage_scope.h"
#include "arg_binder.h"
#include "ir_util.h"

namespace tvm {
namespace tir {

using intrinsic::tvm_address_of;
using runtime::StorageRank;
using runtime::StorageScope;
using runtime::ThreadScope;

class StorageFlattener : public StmtExprMutator {
 public:
  explicit StorageFlattener(Map<te::Tensor, Buffer> extern_buffer, int cache_line_size,
                            bool create_bound_attributes, IRVisitorWithAnalyzer* bounded_analyzer)
      : bounded_analyzer_(bounded_analyzer), create_bound_attributes_(create_bound_attributes) {
    for (auto kv : extern_buffer) {
      BufferEntry e;
      e.buffer = kv.second;
      e.external = true;
      buf_map_[TensorKey{kv.first->op, kv.first->value_index}] = e;
    }
    cache_line_size_ = cache_line_size;
  }

  SyncType getSyncType(FunctionRef func) {
    if (func.as<te::OperationNode>()->attrs.count("no_sync")) {
      // std::cout << "[NONE] " << func << std::endl;
      return kNone;
    } else if (func.as<te::OperationNode>()->attrs.count("no_war_sync")) {
      // std::cout << "[NONE] " << func << std::endl;
      return kNoWar;
    } else if (func.as<te::ScanOpNode>()) {
      // std::cout << "[NOWAR] " << func << std::endl;
      return kNoWar;
    } else if (auto sk_op = func.as<te::SingleKernelEnvelopeOpNode>()) {
      for (auto t : sk_op->inputs) {
        if (t->op.as<te::ScanOpNode>()) {
          // std::cout << "[NOWAR] " << func << std::endl;
          return kNoWar;
        }
      }
    }
    // std::cout << "[ALL] " << func << std::endl;
    return kAll;
  }

  SyncType getSyncType(FunctionRef func, Buffer buf) {
    SyncType op_sync = getSyncType(func);
    SyncType buf_sync = buf->sync_type;
    if (buf_sync == kNoWar) {
      // std::cout << "[NOWAR] " << buf << std::endl;
    }
    if (op_sync > buf_sync)
      return op_sync;
    else
      return buf_sync;
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() && !it->second.same_as(op->buffer_var)) {
      CHECK(it->second.as<VarNode>());
      Var buf_var = Downcast<Var>(it->second);
      return StoreNode::make(buf_var, op->value, op->index, op->predicate, op->sync_type);
    } else {
      return stmt;
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::realize_scope) {
      storage_scope_[op->node.get()] = op->value.as<StringImmNode>()->value;
      return this->VisitStmt(op->body);
    } else if (op->attr_key == attr::double_buffer_scope &&
               op->node->IsInstance<te::OperationNode>()) {
      auto func = Downcast<te::Operation>(op->node);
      Stmt body = this->VisitStmt(op->body);
      for (int i = 0; i < func->num_outputs(); ++i) {
        TensorKey key{func, i};
        auto it = buf_map_.find(key);
        CHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << key.f;
        body = AttrStmtNode::make(it->second.buffer->data, op->attr_key, op->value, body);
      }
      return body;
    } else if (op->attr_key == attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ThreadScope ts = ThreadScope::make(iv->thread_tag);
      curr_thread_scope_.push_back(ts);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      curr_thread_scope_.pop_back();
      return stmt;
    } else if (op->attr_key == attr::buffer_bind_scope) {
      return HandleBufferBindScope(op);
    } else if (op->attr_key == attr::buffer_dim_align) {
      auto tensor = Downcast<te::Tensor>(op->node);
      const CallNode* tuple = op->value.as<CallNode>();
      CHECK(tuple && tuple->is_intrinsic(intrinsic::tvm_tuple));
      TensorKey key{tensor->op, tensor->value_index};
      auto& vinfo = dim_align_[key];
      int dim = tuple->args[0].as<IntImmNode>()->value;
      if (static_cast<size_t>(dim) >= vinfo.size()) {
        vinfo.resize(dim + 1);
      }
      vinfo[dim].align_factor = tuple->args[1].as<IntImmNode>()->value;
      vinfo[dim].align_offset = tuple->args[2].as<IntImmNode>()->value;
      return this->VisitStmt(op->body);
    } else if (op->attr_key == attr::opengl_stage_scope) {
      is_opengl_ = true;
    } else if (op->attr_key == attr::aux_data_structure) {
      if (auto ufn = op->node.as<UninterpFunNode>()) {
        UninterpFun new_uf =
            UninterpFunNode::make(ufn->fname, ufn->range, ufn->dimensions, ufn->parameters,
                                  this->VisitExpr(ufn->body), ufn->type);
        return AttrStmtNode::make(new_uf, op->attr_key, this->VisitExpr(op->value),
                                  this->VisitStmt(op->body));
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ProvideNode* op) final {
    // std::cout << "[PROVIDE] " << op->func << std::endl;
    if (create_bound_attributes_) shape_collector_.clear();
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<ProvideNode>();
    TensorKey key{op->func, op->value_index};
    auto it = buf_map_.find(key);
    CHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << key.f;
    const BufferEntry& e = it->second;
    CHECK(!e.released) << "Read a buffer that is already out of scope";
    if (is_opengl_) {
      return EvaluateNode::make(CallNode::make(DataType(), CallNode::glsl_texture_store,
                                               {e.buffer->data, op->value}, CallNode::Intrinsic));
    } else {
      if (op->custom_realize_bounds.size() == op->args.size()) {
        // std::cout << "[SF] Custom realize bounds for provide " << op->func << std::endl;
        // for (auto it : op->custom_realize_bounds) {
        //   std::cout << "[SF]  Bound " << it << std::endl;
        // }
      }

      Stmt body = e.buffer.vstore(e.RelIndex(this, op->args, op->custom_realize_bounds), op->value,
                                  getSyncType(op->func, e.buffer));
      body = this->VisitStmt(body);
      if (create_bound_attributes_ && ShapeIsValid(e.buffer->shape->get_dense_shape())) {
        shape_collector_.push_back(
            std::make_pair(e.buffer->data, e.buffer->shape->get_dense_shape()));
      }
      // To create bound attribute collector should has at least one item.
      if (create_bound_attributes_ && shape_collector_.size()) {
        for (size_t i = 0; i < shape_collector_.size(); ++i) {
          body = AttrStmtNode::make(shape_collector_[i].first, tir::attr::buffer_bound,
                                    MakeBound(e.buffer->dtype, shape_collector_[i].second), body);
        }
      }
      return body;
    }
  }

  Stmt VisitStmt_(const RealizeNode* op) final {
    // std::cout << "[REALIZE] " << op->func << std::endl;
    TensorKey key{op->func, op->value_index};
    if (buf_map_.count(key)) {
      CHECK(buf_map_.at(key).external);
      return this->VisitStmt(op->body);
    } else {
      // create a buffer entry
      BufferEntry e;
      e.bounds = op->bounds;
      Array<PrimExpr> shape;
      for (auto r : e.bounds) {
        shape.push_back(r->extent);
      }
      // deduce current storage scope.
      auto it = storage_scope_.find(op->func.get());
      CHECK(it != storage_scope_.end())
          << "Cannot find storage scope of " << op->func << " value_index=" << op->value_index;
      StorageScope skey;
      const std::string& strkey = it->second;
      if (strkey.length() == 0) {
        if (curr_thread_scope_.size() != 0) {
          skey.rank = runtime::DefaultStorageRank(curr_thread_scope_.back().rank);
        }
      } else {
        skey = StorageScope::make(strkey);
      }

      Modes layout = NullValue<Modes>();
      if (op->layout.defined()) {
        layout = Downcast<Modes>(op->layout);
      }

      // use small alignment for small arrays
      int32_t const_size = AllocateNode::constant_allocation_size(shape, layout);
      int align = GetTempAllocaAlignment(op->dtype, const_size);
      if (skey.tag.length() != 0) {
        MemoryInfo info = GetMemoryInfo(skey.to_string());
        if (info.defined()) {
          align = (info->max_simd_bits + op->dtype.bits() - 1) / op->dtype.bits();
          CHECK_LE(const_size * op->dtype.bits(), info->max_num_bits)
              << "Allocation exceed bound of memory tag " << skey.to_string();
        }
      }
      Array<PrimExpr> strides;
      if (dim_align_.count(key) != 0 && shape.size() != 0) {
        // std::cout << "[SF] Found align for " << key.f << std::endl;
        std::vector<PrimExpr> rstrides;
        const std::vector<DimAlignInfo>& avec = dim_align_[key];
        int first_dim = 0;
        PrimExpr stride = make_const(shape[first_dim].dtype(), 1);
        for (size_t i = shape.size(); i != 0; --i) {
          size_t dim = i - 1;
          if (dim < avec.size() && avec[dim].align_factor != 0) {
            PrimExpr factor = make_const(stride.dtype(), avec[dim].align_factor);
            PrimExpr offset = make_const(stride.dtype(), avec[dim].align_offset);
            stride = stride + indexmod(factor + offset - indexmod(stride, factor), factor);
            // stride = stride + offset;
            stride = tir::Simplify(stride);
            shape.Set(dim, shape[dim] + offset);
            std::cout << "[SF]     F, O, S " << factor << " " << offset << " " << stride
                      << std::endl;
          }
          std::cout << "[SF]   Stride " << stride << std::endl;
          rstrides.push_back(stride);
          stride = stride * shape[dim];
        }
        strides = Array<PrimExpr>(rstrides.rbegin(), rstrides.rend());

        // for (auto it: strides) {
        //   std::cout << "[SF]   Stride " << it << std::endl;
        // }
      }

      // if (layout.defined()) {
      if (false) {
        // std::cout << "[SF] Dimensions for " << op->func << std::endl;
        e.buffer = BufferNode::make(Var(key.GetName(), DataType::Handle()), op->dtype, layout,
                                    strides, PrimExpr(), key.GetName(), skey.to_string(), align, 0,
                                    kDefault, getSyncType(op->func));

      } else {
        e.buffer = BufferNode::make(Var(key.GetName(), DataType::Handle()), op->dtype, shape,
                                    strides, PrimExpr(), key.GetName(), skey.to_string(), align, 0,
                                    kDefault, getSyncType(op->func));
      }

      // std::cout << "[SF] Realize node for " << key.f << std::endl;
      buf_map_[key] = e;
      Stmt body = this->VisitStmt(op->body);
      buf_map_[key].released = true;
      // std::cout << "[SF] Realize node released for " << key.f << std::endl;
      Stmt ret;

      DataType storage_type = e.buffer->dtype;
      // specially handle bool, lower its storage
      // type to beDataType::Int(8)(byte)
      if (storage_type == DataType::Bool()) {
        storage_type = DataType::Int(8);
      }
      if (strides.size() != 0) {
        CHECK(!layout.defined()) << "Not sure how to handle storage layouts in the presence of "
                                    "strides. This happens for "
                                 << op->func;
        int first_dim = 0;
        ret = AllocateNode::make(
            e.buffer->data, storage_type,
            {e.buffer->strides[first_dim] * e.buffer->shape->get_dense_shape()[first_dim]},
            make_const(DataType::Bool(e.buffer->dtype.lanes()), true), body);
      } else {
        shape = e.buffer->shape->get_dense_shape();
        if (shape.size() == 0) {
          shape.push_back(make_const(DataType::Int(32), 1));
        }
	Array<PrimExpr> inlined_shape;

	for (auto e: shape) {
	  inlined_shape.push_back(this->VisitExpr(e));
	}
        ret = AllocateNode::make(e.buffer->data, storage_type, inlined_shape, layout,
                                 make_const(DataType::Bool(e.buffer->dtype.lanes()), true), body);
      }
      ret = AttrStmtNode::make(e.buffer->data, attr::storage_scope,
                               StringImmNode::make(e.buffer->scope), ret);

      if (create_bound_attributes_ && ShapeIsValid(e.buffer->shape->get_dense_shape())) {
        ret =
            AttrStmtNode::make(e.buffer->data, tir::attr::buffer_bound,
                               MakeBound(e.buffer->dtype, e.buffer->shape->get_dense_shape()), ret);
      }
      return ret;
    }
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    // std::cout << "[LOAD] " << op->buffer_var << std::endl;
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<LoadNode>();
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() && !it->second.same_as(op->buffer_var)) {
      CHECK(it->second.as<VarNode>());
      Var buf_var = Downcast<Var>(it->second);
      return LoadNode::make(op->dtype, buf_var, op->index, op->predicate, op->sync_type);
    } else {
      return expr;
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_remap_.find(op);
    if (it != var_remap_.end()) {
      return it->second;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    if (op != nullptr && op->call_type == CallNode::Halide) {
      bool print = false;//op->func.as<te::OperationNode>()->name == "QKV";
      if (print) std::cout << "[CALL] " << op->func << std::endl;
      TensorKey key{op->func, op->value_index};
      auto it = buf_map_.find(key);
      CHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << key.f
                                  << GetRef<PrimExpr>(op);
      const BufferEntry& e = it->second;
      CHECK(!e.released) << "Read a buffer that is already out of scope";

      auto buffer_dense_shape = e.buffer->shape->get_dense_shape();
      if (create_bound_attributes_ && ShapeIsValid(buffer_dense_shape)) {
        shape_collector_.push_back(std::make_pair(e.buffer->data, buffer_dense_shape));
      }
      Array<PrimExpr> args;
      for (auto arg: op->args) {
	args.push_back(this->VisitExpr(arg));
      }
      auto ret =
	this->VisitExpr(
	  UninterpFun::InlineUninterpFunCalls(e.buffer.vload(e.RelIndex(this, args, op->custom_realize_bounds),
							     e.buffer->dtype, getSyncType(op->func, e.buffer))));
      if (print) std::cout << "[SF] Ret for " << GetRef<PrimExpr>(op) << " " << ret << std::endl;
      return ret;
    } else {
      return expr;
    }
  }

  Stmt VisitStmt_(const PrefetchNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<PrefetchNode>();
    CHECK(op != nullptr);
    TensorKey key{op->func, op->value_index};
    auto it = buf_map_.find(key);
    CHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << key.f;
    const BufferEntry& e = it->second;

    CHECK(!e.released) << "Read a buffer that is already out of scope";

    auto buffer_dense_shape = e.buffer->shape->get_dense_shape();
    CHECK_EQ(buffer_dense_shape.size(), op->bounds.size())
        << "Prefetch dim should be the same as buffer dim";

    int block_size = 1, elem_cnt = cache_line_size_ / e.buffer->dtype.bytes(), shape = 0;

    int starts = op->bounds.size() - 1;
    while (starts > 0 && arith::GetConstInt(buffer_dense_shape[starts], &shape) &&
           elem_cnt >= block_size * shape) {
      block_size *= shape;
      starts--;
    }
    PrimExpr stride(elem_cnt / block_size);

    Array<PrimExpr> args;
    std::vector<Var> vars;

    for (int i = op->bounds.size() - 1; i > starts; --i) {
      args.push_back(op->bounds[i]->min);
    }
    auto& func_name = op->func->func_name();
    vars.push_back(Var("prefetch." + func_name + "." + std::to_string(starts), DataType::Int(32)));
    args.push_back(op->bounds[starts]->min + stride * vars.back());
    for (int i = starts - 1; i >= 0; --i) {
      vars.push_back(Var("prefetch." + func_name + "." + std::to_string(i), DataType::Int(32)));
      args.push_back(vars.back() + op->bounds[i]->min);
    }
    for (int i = starts; i >= 0; --i) {
      if (i < starts) {
        stmt = ForNode::make(vars[i], 0, op->bounds[i]->extent, ForType::Serial, DeviceAPI::None,
                             stmt);
      } else {
        PrimExpr load = e.buffer.vload(e.RelIndex(this, args), e.buffer->dtype,
                                       getSyncType(op->func, e.buffer));
        load = this->VisitExpr(load);
        PrimExpr address =
            CallNode::make(DataType::Handle(), tvm_address_of, {load}, CallNode::PureIntrinsic);
        PrimExpr prefetch =
            CallNode::make(op->dtype, CallNode::prefetch, {address, 0, 3, 1}, CallNode::Intrinsic);
        stmt = EvaluateNode::make(prefetch);
        PrimExpr extent = (op->bounds[i]->extent - 1) / stride + 1;
        stmt = ForNode::make(vars[i], 0, extent, ForType::Serial, DeviceAPI::None, stmt);
      }
    }
    return stmt;
  }

 private:
  // The specific tensor data layout is not determined before
  // StorageFlatten pass. We use buffer_bind_scope
  // to specify before hand we want to bind a subregion
  // of tensor to a symbolic buffer, which get used in extern.
  //
  // Example:
  //
  // realize A in range [i*4, extent=10) {
  //   bind Ab to A in [i*4+1, extent=4) {
  //     call_func(Ab.ptr, Ab.shape[0])
  //   }
  // }
  //
  // After StorageFlatten
  //
  // alloc A[10]
  //   call(A + 1,  4)
  //
  // Buffer is a protocol to declare specific
  // data layout and shape we expect.
  // So this function need to check:
  // - If the bind range is within the realize range
  // - If we can match the requirement of buffer
  // - Remap variables such as Ab.ptr to the actual value.
  //
  // Here are a few possible failure cases:
  // - Buffer is declared to have constant shape,
  //   but we try to bind it to a different one.
  // - Buffer is declared to be compact(no strides)
  //   but this binded region is a subregion of
  //   a matrix(tensor), which means it requires strides.
  //
  // We do support a few relaxed case, such as bindingx
  // region with shape [1, 1, n, m] to buffer with shape [n, m]
  Stmt HandleBufferBindScope(const AttrStmtNode* op) {
    Array<ObjectRef> arr = Downcast<Array<ObjectRef>>(op->node);
    CHECK_EQ(arr.size(), 2U);
    const BufferNode* buffer = arr[0].as<BufferNode>();
    const te::TensorNode* tensor = arr[1].as<te::TensorNode>();
    const CallNode* tuple = op->value.as<CallNode>();
    CHECK(buffer && tensor);
    CHECK(tuple && tuple->is_intrinsic(intrinsic::tvm_tuple));
    TensorKey key{tensor->op, tensor->value_index};
    CHECK(buf_map_.count(key)) << "Cannot find buffer of " << tensor->op
                               << " value=" << tensor->value_index;
    const BufferEntry& be = buf_map_.at(key);
    CHECK(!be.released);
    auto buffer_dense_shape = be.buffer->shape->get_dense_shape();
    if (tuple->args.size() != buffer_dense_shape.size() * 2) {
      for (auto s : buffer_dense_shape) {
        std::cout << "[SHAPE]   " << s << std::endl;
      }
    }
    CHECK_EQ(tuple->args.size(), buffer_dense_shape.size() * 2)
        << " " << GetRef<PrimExpr>(tuple) << " " << be.buffer;
    Array<PrimExpr> begins, extents;
    if (be.bounds.size() != 0) {
      CHECK_EQ(tuple->args.size(), be.bounds.size() * 2);
      for (size_t i = 0; i < buffer_dense_shape.size(); ++i) {
        begins.push_back(tuple->args[2 * i] - be.bounds[i]->min);
        extents.push_back(tuple->args[2 * i + 1]);
      }
    } else {
      for (size_t i = 0; i < tuple->args.size(); i += 2) {
        begins.push_back(tuple->args[i]);
        auto new_extent = bounded_analyzer_->Simplify(tuple->args[i + 1]);
        extents.push_back(new_extent);
      }
    }
    Buffer slice = be.buffer.MakeSlice(begins, extents);
    if (buffer->strides.size() == 0) {
      CHECK_EQ(slice->strides.size(), 0U)
          << "Trying to bind compact buffer to strided one strides=" << slice->strides;
    } else {
      slice = slice.MakeStrideView();
    }
    // start binding
    ArgBinder binder(&var_remap_);
    binder.BindBuffer(Downcast<Buffer>(arr[0]), slice, buffer->name, true);
    // Apply the remaps
    Stmt body = MergeNest(binder.asserts(), op->body);
    body = MergeNest(binder.init_nest(), body);
    body = this->VisitStmt(body);
    // remove the binds
    for (const Var& v : binder.defs()) {
      var_remap_.erase(v.get());
    }
    return body;
  }
  // The buffer entry in the flatten map
  struct DimAlignInfo {
    int align_factor{0};
    int align_offset{0};
  };
  // The buffer entry in the flatten map
  struct BufferEntry {
    // the buffer of storage
    Buffer buffer;
    // the bounds of realization, can be null, means everything
    Region bounds;
    // Whether the buffer is external
    bool external{false};
    // Whether we are out of allocation bounds and buffer get released.
    bool released{false};
    // relative index
    inline Array<PrimExpr> RelIndex(StorageFlattener* flattener, Array<PrimExpr> args,
                                    Array<Range> override_realize_bounds = {}) const {
      if (bounds.size() != 0) {
        Array<PrimExpr> index;
        if (override_realize_bounds.size() > 0) {
          CHECK_EQ(override_realize_bounds.size(), args.size()) << buffer;
          for (size_t i = 0; i < override_realize_bounds.size(); ++i) {
            PrimExpr rel_index = tir::Simplify(flattener->VisitExpr(
                UninterpFun::InlineUninterpFunCalls(args[i] - override_realize_bounds[i]->min)));
            index.push_back(rel_index);
          }
        } else {
          CHECK_EQ(bounds.size(), args.size()) << buffer;
          for (size_t i = 0; i < bounds.size(); ++i) {
            PrimExpr rel_index = tir::Simplify(flattener->VisitExpr(
                UninterpFun::InlineUninterpFunCalls(args[i] - bounds[i]->min)));
            index.push_back(rel_index);
          }
        }
        return index;
      } else {
        return args;
      }
    }
  };

  bool ShapeIsValid(const Array<PrimExpr>& shape) {
    // Zero-dimensional tensor does not need boundary check.
    if (!shape.size()) return false;

    for (size_t i = 0; i < shape.size(); ++i) {
      if (!shape[i].defined() || !shape[i].dtype().is_scalar() || is_negative_const(shape[i])) {
        return false;
      }
    }
    return true;
  }

  PrimExpr MakeBound(const DataType& type, const Array<PrimExpr>& shape) {
    // We have already checked the shape size to be greater then 0.
    PrimExpr bound = MulNode::make(make_const(shape[0].dtype(), type.lanes()), shape[0]);
    for (size_t i = 1; i < shape.size(); ++i) {
      bound =
          MulNode::make(bound, MulNode::make(make_const(bound.dtype(), type.lanes()), shape[i]));
    }
    return bound;
  }

  // The buffer assignment map
  // Variable remap
  std::unordered_map<const VarNode*, PrimExpr> var_remap_;
  // Buffer map
  std::unordered_map<TensorKey, BufferEntry> buf_map_;
  // Dimension alignment
  std::unordered_map<TensorKey, std::vector<DimAlignInfo>> dim_align_;
  // Storage scope
  std::unordered_map<const Object*, std::string> storage_scope_;
  // The current thread scope.
  std::vector<ThreadScope> curr_thread_scope_;
  // Collects shapes.
  std::vector<std::pair<Var, Array<PrimExpr>>> shape_collector_;
  // bounds populator. We really need the analyzer from it.
  // However
  IRVisitorWithAnalyzer* bounded_analyzer_;
  // The size of cacheline
  int cache_line_size_;
  // The current stage is an OpenGL shader.
  bool is_opengl_{false};
  // Whether to mark load/store with theirs bounds.
  bool create_bound_attributes_{false};
};

Stmt StorageFlatten(Stmt stmt, Map<te::Tensor, Buffer> extern_buffer, int cache_line_size,
                    bool create_bound_attributes) {
  // std::cout << "Yo flattening" << std::endl;
  IRVisitorWithAnalyzer bounded_analyzer;
  bounded_analyzer(stmt);
  stmt = StorageFlattener(extern_buffer, cache_line_size, create_bound_attributes,
                          &bounded_analyzer)(std::move(stmt));
  return stmt;
  // std::cout << "[SF] Inlining " << std::endl;
  // return UninterpFun::InlineUninterpFunCalls(stmt);
}

}  // namespace tir
}  // namespace tvm
