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
 *  Lower TVM related buildin intrinsics such as packed call.
 * \file lower_tvm_buildin.cc
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_set>

#include "../../arith/compute_expr.h"
#include "ir_util.h"

namespace tvm {
namespace tir {

inline PrimExpr ConstInt32(size_t index) {
  CHECK_LE(index, std::numeric_limits<int>::max());
  return make_const(DataType::Int(32), static_cast<int>(index));
}

inline PrimExpr StackAlloca(std::string type, size_t num) {
  Array<PrimExpr> args = {StringImmNode::make(type), ConstInt32(num)};
  return CallNode::make(DataType::Handle(), intrinsic::tvm_stack_alloca, args, CallNode::Intrinsic);
}

// Calculate the statistics of packed function.
// These information are needed during codegen.

class PushLetsBelowCPUThread : public StmtExprMutator {
 public:
  PushLetsBelowCPUThread(Array<Stmt> lets_) : lets(lets_) {}

  Stmt VisitStmt_(const ForNode* op) override {
    // if (op->loop_var->name_hint == "cpu_par_thread.x") {
    if (op->for_type == ForType::Parallel) {
      // CHECK(!pushed);
      Stmt body = op->body;
      for (auto s : lets) {
        auto let = s.as<LetStmtNode>();
        body = LetStmtNode::make(let->var, let->value, body);
      }
      pushed = true;
      return ForNode::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body,
                           op->hfuse_group_id);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Array<Stmt> lets;
  bool pushed = false;
};

class BuiltinLower : public StmtExprMutator {
 public:
  Stmt Build(Stmt stmt) {
    stack_shape_ = Var("stack_shape", DataType::Handle());
    stack_array_ = Var("stack_array", DataType::Handle());
    stack_value_ = Var("stack_value", DataType::Handle());
    stack_tcode_ = Var("stack_tcode", DataType::Handle());
    stmt = this->VisitStmt(stmt);
    Array<Stmt> all_stmts;
    all_stmts.push_back(stmt);
    all_stmts.push_back_all(prep_free_stmts_);
    stmt = SeqStmt(all_stmts);

    auto noop = EvaluateNode::make(0);
    Array<Stmt> lets;
    if (max_shape_stack_ != 0) {
      lets.push_back(LetStmtNode::make(stack_shape_, StackAlloca("shape", max_shape_stack_), noop));
    }
    if (max_array_stack_ != 0) {
      lets.push_back(LetStmtNode::make(stack_array_, StackAlloca("array", max_array_stack_), noop));
    }
    if (max_arg_stack_ != 0) {
      lets.push_back(
          LetStmtNode::make(stack_value_, StackAlloca("arg_value", max_arg_stack_), noop));
      lets.push_back(
          LetStmtNode::make(stack_tcode_, StackAlloca("arg_tcode", max_arg_stack_), noop));
    }

    auto let_pusher = PushLetsBelowCPUThread(lets);
    stmt = let_pusher(stmt);

    if (!let_pusher.pushed) {
      for (auto s : lets) {
        auto let = s.as<LetStmtNode>();
        stmt = LetStmtNode::make(let->var, let->value, stmt);
      }
    }

    return stmt;
  }

  Stmt VisitStmt(const Stmt& s) final {
    auto stmt = StmtExprMutator::VisitStmt(s);
    CHECK_EQ(run_shape_stack_, 0);
    CHECK_EQ(run_array_stack_, 0);

    if (prep_seq_.size() != 0) {
      Stmt ret = SeqStmt::Flatten(prep_seq_, stmt);
      prep_seq_.clear();
      return ret;
    } else {
      return stmt;
    }
  }

  Stmt VisitStmt_(const AllocateNode* op) {
    // Lower allocate to device allocate when needed.
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<AllocateNode>();
    if (op->new_expr.defined()) return stmt;
    // Get constant allocation bound.
    int64_t dev_type;
    int64_t nbytes = GetVectorBytes(op->dtype);
    if (device_type_.defined()) {
      if (arith::GetConst(device_type_, &dev_type)) {
        if (dev_type == kDLCPU) {
          int32_t constant_size = op->constant_allocation_size();
          if (constant_size > 0 && constant_size * nbytes < runtime::kMaxStackAlloca) {
            return stmt;
          }
        }
      }
    }
    PrimExpr total_bytes = make_const(op->extents[0].dtype(), nbytes);
    for (size_t i = 0; i < op->extents.size(); ++i) {
      total_bytes = total_bytes * op->extents[i];
    }
    CHECK(device_type_.defined()) << "Unknown device type in current IR";
    CHECK(device_id_.defined()) << "Unknown device id in current IR";
    Stmt throw_last_error = EvaluateNode::make(CallNode::make(
        DataType::Int(32), intrinsic::tvm_throw_last_error, {}, CallNode::Intrinsic));

    Stmt body = SeqStmt(
        {IfThenElseNode::make(CallNode::make(DataType::Bool(1), intrinsic::tvm_handle_is_null,
                                             {op->buffer_var}, CallNode::PureIntrinsic),
                              throw_last_error),
         op->body});

    PrimExpr allocate_device_type = in_prep_code_ ? kDLCPU : device_type_;
    PrimExpr allocate_device_id = in_prep_code_ ? 0 : device_id_;

    Stmt alloca =
        LetStmtNode::make(op->buffer_var,
                          CallNode::make(op->buffer_var.dtype(), "TVMBackendAllocWorkspace",
                                         {cast(DataType::Int(32), allocate_device_type),
                                          cast(DataType::Int(32), allocate_device_id),
                                          cast(DataType::UInt(64), total_bytes),
                                          IntImm(DataType::Int(32), op->dtype.code()),
                                          IntImm(DataType::Int(32), op->dtype.bits())},
                                         CallNode::Extern),
                          body);

    PrimExpr free_op = CallNode::make(DataType::Int(32), "TVMBackendFreeWorkspace",
                                      {cast(DataType::Int(32), allocate_device_type),
                                       cast(DataType::Int(32), allocate_device_id), op->buffer_var},
                                      CallNode::Extern);
    Stmt free_stmt =
        IfThenElseNode::make(free_op != make_zero(DataType::Int(32)), throw_last_error);
    // if (in_prep_code_) {
    // body = alloca;
    // prep_free_stmts_.push_back(free_stmt);
    // } else {
    body = SeqStmt({alloca, free_stmt});
    // }
    body = AttrStmtNode::make(op->buffer_var, attr::storage_alignment,
                              make_const(DataType::Int(32), runtime::kTempAllocaAlignment), body);
    return body;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::device_context_id) {
      CHECK(!device_id_.defined());
      device_id_ = op->value;
      return this->VisitStmt(op->body);
    } else if (op->attr_key == attr::device_context_type) {
      CHECK(!device_type_.defined());
      device_type_ = op->value;
      return this->VisitStmt(op->body);
    } else if (op->attr_key == attr::prep_code_scope) {
      in_prep_code_ = true;
      auto ret = StmtExprMutator::VisitStmt_(op);
      in_prep_code_ = false;
      return ret;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->is_intrinsic(intrinsic::tvm_call_packed)) {
      return MakeCallPacked(op);
    } else if (op->is_intrinsic(intrinsic::tvm_call_trace_packed)) {
      return MakeCallTracePacked(op);
    } else if (op->is_intrinsic(intrinsic::tvm_stack_make_shape)) {
      return MakeShape(op);
    } else if (op->is_intrinsic(intrinsic::tvm_stack_make_array)) {
      return MakeArray(op);
    } else if (op->is_intrinsic(intrinsic::tvm_context_id)) {
      return make_zero(op->dtype);
    } else if (op->is_intrinsic(intrinsic::tvm_memcopy_to_device)) {
      return MakeMemcpy(op);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }
  // call shape
  PrimExpr MakeShape(const CallNode* op) {
    size_t stack_begin = run_shape_stack_;
    run_shape_stack_ += op->args.size();
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    for (size_t i = 0; i < op->args.size(); ++i) {
      prep_seq_.emplace_back(StoreNode::make(stack_shape_, cast(DataType::Int(64), op->args[i]),
                                             ConstInt32(stack_begin + i), const_true(1),
                                             tir::kAll));
    }
    return AddressOffset(stack_shape_, DataType::Int(64), stack_begin);
  }
  // make array
  PrimExpr MakeArray(const CallNode* op) {
    size_t idx = run_array_stack_;
    run_array_stack_ += 1;
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, intrinsic::kArrData, op->args[0]));
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, intrinsic::kArrShape, op->args[1]));
    PrimExpr strides = op->args[2];
    if (!strides.defined() || is_zero(strides)) {
      strides = make_zero(DataType::Handle());
    }
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, intrinsic::kArrStrides, strides));
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, intrinsic::kArrNDim, op->args[3]));
    DataType dtype = op->args[4].dtype();
    prep_seq_.emplace_back(
        TVMStructSet(stack_array_, idx, intrinsic::kArrTypeCode,
                     make_const(DataType::UInt(8), static_cast<int>(dtype.code()))));
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, intrinsic::kArrTypeBits,
                                        make_const(DataType::UInt(8), dtype.bits())));
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, intrinsic::kArrTypeLanes,
                                        make_const(DataType::UInt(16), dtype.lanes())));
    // set byte offset
    int data_bytes = GetVectorBytes(dtype);
    PrimExpr byte_offset = op->args[5];
    if (!is_zero(byte_offset)) {
      byte_offset = byte_offset * make_const(byte_offset.dtype(), data_bytes);
    }
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, intrinsic::kArrByteOffset,
                                        cast(DataType::UInt(64), byte_offset)));
    CHECK(device_type_.defined()) << "Unknown device type in current IR";
    CHECK(device_id_.defined()) << "Unknown device id in current IR";
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, intrinsic::kArrDeviceId,
                                        cast(DataType::Int(32), device_id_)));
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, intrinsic::kArrDeviceType,
                                        cast(DataType::Int(32), device_type_)));
    return TVMStructGet(DataType::Handle(), stack_array_, idx, intrinsic::kArrAddr);
  }
  // call packed.
  PrimExpr MakeCallPacked(const CallNode* op) {
    size_t restore_shape_stack = run_shape_stack_;
    size_t restore_array_stack = run_array_stack_;
    size_t arg_stack_begin = run_arg_stack_;
    run_arg_stack_ += op->args.size();
    // Specially handle the buffer packed intrinsic
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    for (size_t i = 1; i < op->args.size(); ++i) {
      PrimExpr stack_index = ConstInt32(arg_stack_begin + i - 1);
      PrimExpr arg = op->args[i];
      DataType t = arg.dtype();
      DataType api_type = APIType(t);
      if (t != api_type) {
        arg = CastNode::make(api_type, arg);
      }
      prep_seq_.emplace_back(TVMStructSet(stack_value_, static_cast<int>(arg_stack_begin + i - 1),
                                          intrinsic::kTVMValueContent, arg));
      int arg_tcode = api_type.code();
      if (api_type.is_handle() && arg.as<StringImmNode>()) {
        arg_tcode = kTVMStr;
      }
      if (IsArrayHandle(arg)) arg_tcode = kTVMDLTensorHandle;
      prep_seq_.emplace_back(StoreNode::make(stack_tcode_, ConstInt32(arg_tcode), stack_index,
                                             const_true(1), tir::kAll));
    }
    // UPDATE stack value
    max_arg_stack_ = std::max(run_arg_stack_, max_arg_stack_);
    max_shape_stack_ = std::max(run_shape_stack_, max_shape_stack_);
    max_array_stack_ = std::max(run_array_stack_, max_array_stack_);
    run_shape_stack_ = restore_shape_stack;
    run_array_stack_ = restore_array_stack;
    run_arg_stack_ = arg_stack_begin;
    Array<PrimExpr> packed_args = {op->args[0], stack_value_, stack_tcode_,
                                   ConstInt32(arg_stack_begin),
                                   ConstInt32(arg_stack_begin + op->args.size() - 1)};
    return CallNode::make(DataType::Int(32), intrinsic::tvm_call_packed_lowered, packed_args,
                          CallNode::Intrinsic);
  }

  PrimExpr MakeCallTracePacked(const CallNode* op) {
    size_t restore_shape_stack = run_shape_stack_;
    size_t restore_array_stack = run_array_stack_;
    size_t arg_stack_begin = run_arg_stack_;
    run_arg_stack_ += op->args.size();
    size_t args_size = op->args.size();
    CHECK_GT(args_size, 0);
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    for (size_t i = 1; i < op->args.size(); ++i) {
      PrimExpr stack_index = ConstInt32(arg_stack_begin + i - 1);
      PrimExpr arg = op->args[i];
      DataType t = arg.dtype();
      DataType api_type = APIType(t);
      if (t != api_type) {
        arg = CastNode::make(api_type, arg);
      }
      prep_seq_.emplace_back(TVMStructSet(stack_value_, static_cast<int>(arg_stack_begin + i - 1),
                                          intrinsic::kTVMValueContent, arg));
      int arg_tcode = api_type.code();
      CHECK(!IsArrayHandle(arg)) << "Trace does not support Buffers";
      prep_seq_.emplace_back(StoreNode::make(stack_tcode_, ConstInt32(arg_tcode), stack_index,
                                             const_true(1), tir::kAll));
    }
    // UPDATE stack value
    max_arg_stack_ = std::max(run_arg_stack_, max_arg_stack_);
    max_shape_stack_ = std::max(run_shape_stack_, max_shape_stack_);
    max_array_stack_ = std::max(run_array_stack_, max_array_stack_);
    run_shape_stack_ = restore_shape_stack;
    run_array_stack_ = restore_array_stack;
    // Update the top of the stack, so we can use more than one
    // packed function's arguments with the one stack.
    run_arg_stack_ = arg_stack_begin + args_size - 1;
    Array<PrimExpr> packed_args = {op->args[0], stack_value_, stack_tcode_,
                                   ConstInt32(arg_stack_begin),
                                   ConstInt32(arg_stack_begin + op->args.size() - 1),
                                   // Pass traced value.
                                   op->args[args_size - 1]};
    return CallNode::make(op->dtype, intrinsic::tvm_call_trace_packed_lowered, packed_args,
                          CallNode::Intrinsic);
  }

  PrimExpr MakeMemcpy(const CallNode* op) {
    return CallNode::make(
        op->dtype, "TVMBackendCopyMemory",
        {op->args[0], cast(DataType::UInt(32), op->args[1]), op->args[2],
         cast(DataType::UInt(32), op->args[3]), cast(DataType::UInt(32), op->args[4]),
         cast(DataType::UInt(32), op->args[5]), cast(DataType::UInt(32), op->args[6]),
         cast(DataType::UInt(32), op->args[7]), cast(DataType::UInt(32), op->args[8]),
         cast(DataType::UInt(32), op->args[9]), cast(DataType::UInt(32), op->args[10])},
        CallNode::Extern);
  }

 private:
  bool IsArrayHandle(const PrimExpr& arg) {
    // specially set array handle.
    if (const CallNode* buf = arg.as<CallNode>()) {
      if (buf->is_intrinsic(intrinsic::tvm_struct_get) &&
          buf->args[2].as<IntImmNode>()->value == intrinsic::kArrAddr) {
        return true;
      }
    }
    return false;
  }

  // The prepration sequence to be emitted.
  std::vector<Stmt> prep_seq_;
  PrimExpr device_type_;
  PrimExpr device_id_;
  // Var handle for each stack.
  Var stack_shape_;
  Var stack_array_;
  Var stack_tcode_;
  Var stack_value_;
  // The running statistics
  uint64_t run_shape_stack_{0};
  uint64_t run_array_stack_{0};
  uint64_t run_arg_stack_{0};
  // statistics of stacks
  uint64_t max_shape_stack_{0};
  uint64_t max_array_stack_{0};
  uint64_t max_arg_stack_{0};
  // Where are we
  bool in_prep_code_{false};
  Array<Stmt> prep_free_stmts_;
};

LoweredFunc LowerTVMBuiltin(LoweredFunc f) {
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  n->body = BuiltinLower().Build(n->body);
  // exit(0);
  return LoweredFunc(n);
}

}  // namespace tir
}  // namespace tvm
