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
 * \file arg_binder.cc
 * \brief Helper utility to match and bind arguments.
 */
#include "arg_binder.h"

#include <tvm/runtime/device_api.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>

#include "../../arith/compute_expr.h"
#include "ir_util.h"

namespace tvm {
namespace tir {

void BinderAddAssert(PrimExpr cond, const std::string& arg_name, std::vector<Stmt>* asserts) {
  PrimExpr scond = Simplify(cond);
  if (is_zero(scond)) {
    LOG(FATAL) << "Bind have an unmet assertion: " << cond << ", "
               << " on argument " << arg_name;
  }
  if (!is_one(scond)) {
    std::ostringstream os;
    os << "Argument " << arg_name << " has an unsatisfied constraint";
    asserts->emplace_back(AssertStmtNode::make(scond, os.str(), EvaluateNode::make(0)));
  }
}

bool ArgBinder::Bind_(const PrimExpr& arg, const PrimExpr& value, const std::string& arg_name,
                      bool with_lets) {
  CHECK_EQ(arg.dtype(), value.dtype());
  if (const VarNode* v = arg.as<VarNode>()) {
    auto it = def_map_->find(v);
    if (it == def_map_->end()) {
      Var v_arg = Downcast<Var>(arg);
      defs_.emplace_back(v_arg);
      if (with_lets) {
        (*def_map_)[v] = arg;
        init_nest_.emplace_back(LetStmtNode::make(v_arg, value, EvaluateNode::make(0)));
      } else {
        (*def_map_)[v] = value;
      }
      return true;
    } else {
      if (!(it->second.dtype().is_handle() && value.dtype().is_handle())) {
        // std::cout << "[BIND] " << it->second << " " << value << " " << it->second.dtype() << " "
                  // << value.dtype() << std::endl;
        BinderAddAssert(it->second == value, arg_name, &asserts_);
      }
    }
  } else {
    // std::cout << "[BIND] " << arg << " " << value << " " << arg_name << std::endl;
    BinderAddAssert(arg == value, arg_name, &asserts_);
  }
  return false;
}

void ArgBinder::Bind(const PrimExpr& arg, const PrimExpr& value, const std::string& arg_name,
                     bool with_let) {
  Bind_(arg, value, arg_name, with_let);
}

void ArgBinder::BindArray(const Array<PrimExpr>& arg, const Array<PrimExpr>& value,
                          const std::string& arg_name) {
  CHECK_EQ(arg.size(), value.size()) << "Argument " << arg_name << " array size mismatch";
  for (size_t i = 0; i < arg.size(); ++i) {
    std::ostringstream os;
    os << arg_name << "[" << i << "]";
    this->Bind(arg[i], value[i], os.str());
  }
}

void ArgBinder::BindBuffer(const Buffer& arg, const Buffer& value, const std::string& arg_name,
                           bool fuzzy_match) {
  CHECK_EQ(arg->scope, value->scope) << "Argument " << arg_name << " Buffer bind scope mismatch";
  CHECK_EQ(arg->dtype, value->dtype)
      << "Argument " << arg_name << " Buffer bind data type mismatch";
  if (value->data_alignment % arg->data_alignment != 0) {
    LOG(WARNING) << "Trying to bind buffer to another one with lower alignment requirement "
                 << " required_alignment=" << arg->data_alignment
                 << ", provided_alignment=" << value->data_alignment;
  }
  // bind pointer and offset.
  if (is_zero(arg->elem_offset)) {
    CHECK(is_zero(value->elem_offset))
        << "Trying to bind a Buffer with offset into one without offset "
        << " required elem_offset=" << arg->elem_offset
        << ", provided elem_offset=" << value->elem_offset;
  }

  this->Bind(arg->data, value->data, arg_name + ".data");
  if (Bind_(arg->elem_offset, value->elem_offset, arg_name + ".elem_offset", false)) {
    if (arg->offset_factor > 1) {
      PrimExpr offset = value->elem_offset;
      PrimExpr factor = make_const(offset.dtype(), arg->offset_factor);
      PrimExpr zero = make_zero(offset.dtype());
      BinderAddAssert(truncmod(offset, factor) == zero, arg_name + ".elem_offset", &asserts_);
    }
  }

  auto value_dense_shape = value->shape->get_dense_shape();
  auto arg_dense_shape = arg->shape->get_dense_shape();

  if (arg->shape->ndim() < value->shape->ndim()) {
    CHECK(fuzzy_match) << "Argument " << arg_name << " size mismatch";
    size_t diff = value->shape->ndim() - arg->shape->ndim();
    for (size_t i = 0; i < diff; ++i) {
      CHECK(is_one(Simplify(value_dense_shape[i])))
          << "Argument " << arg_name << " shape mismatch" << arg->shape << " vs " << value->shape;
    }
    for (size_t i = 0; i < arg->shape->ndim(); ++i) {
      std::ostringstream os;
      os << arg_name << ".shape[" << i << "]";
      this->Bind(arg_dense_shape[i], value_dense_shape[i + diff], os.str());
    }
    if (value->strides.size() != 0) {
      CHECK_EQ(arg->strides.size(), arg->shape->ndim());
      CHECK_EQ(value->strides.size(), value->shape->ndim());
      for (size_t i = 0; i < arg->strides.size(); ++i) {
        std::ostringstream os;
        os << arg_name << ".strides[" << i << "]";
        this->Bind(arg->strides[i], value->strides[i + diff], os.str());
      }
    }
  } else {
    this->BindArray(arg_dense_shape, value_dense_shape, arg_name + ".shape");
    this->BindArray(arg->strides, value->strides, arg_name + ".strides");
  }
}

inline PrimExpr TVMArrayGet(DataType t, Var arr, intrinsic::TVMStructFieldKind kind) {
  return TVMStructGet(t, arr, 0, kind);
}

void ArgBinder::BindDLTensor(const Buffer& buffer, const PrimExpr& device_type,
                             const PrimExpr& device_id, const Var& handle,
                             const std::string& arg_name) {
  const DataType tvm_shape_type = DataType::ShapeIndex();
  const DataType tvm_ndim_type = DataType::Int(32);
  const Stmt nop = EvaluateNode::make(0);
  // dimension checks
  PrimExpr v_ndim = TVMArrayGet(tvm_ndim_type, handle, intrinsic::kArrNDim);
  PrimExpr a_ndim = make_const(tvm_ndim_type, static_cast<int64_t>(buffer->shape->ndim()));
  std::ostringstream ndim_err_msg;
  ndim_err_msg << arg_name << ".ndim is expected to equal " << buffer->shape->ndim();
  asserts_.emplace_back(AssertStmtNode::make(a_ndim == v_ndim, ndim_err_msg.str(), nop));
  // type checks
  DataType dtype = buffer->dtype;
  std::ostringstream type_err_msg;
  type_err_msg << arg_name << ".dtype is expected to be " << dtype;
  PrimExpr cond = (TVMArrayGet(DataType::UInt(8), handle, intrinsic::kArrTypeCode) ==
                       IntImm(DataType::UInt(8), dtype.code()) &&
                   TVMArrayGet(DataType::UInt(8), handle, intrinsic::kArrTypeBits) ==
                       IntImm(DataType::UInt(8), dtype.bits()) &&
                   TVMArrayGet(DataType::UInt(16), handle, intrinsic::kArrTypeLanes) ==
                       IntImm(DataType::UInt(16), dtype.lanes()));
  asserts_.emplace_back(AssertStmtNode::make(cond, type_err_msg.str(), nop));
  // data field
  if (Bind_(buffer->data, TVMArrayGet(DataType::Handle(), handle, intrinsic::kArrData),
            arg_name + ".data", true)) {
    Var vptr(buffer->data);
    def_handle_dtype_.Set(vptr, tir::TypeAnnotation(buffer->dtype));
    // mark alignment of external bufs
    init_nest_.emplace_back(AttrStmtNode::make(vptr, tir::attr::storage_alignment,
                                               IntImm(DataType::Int(32), buffer->data_alignment),
                                               nop));
  }

  auto buffer_dense_shape = buffer->shape->get_dense_shape();

  Var v_shape(arg_name + ".shape", DataType::Handle());
  def_handle_dtype_.Set(v_shape, make_const(tvm_shape_type, 0));
  init_nest_.emplace_back(LetStmtNode::make(
      v_shape, TVMArrayGet(DataType::Handle(), handle, intrinsic::kArrShape), nop));
  for (size_t k = 0; k < buffer->shape->ndim(); ++k) {
    // std::cout << "[AB] Buf size " << buffer->data << " " << buffer_dense_shape[k] << std::endl;
    std::ostringstream field_name;
    field_name << v_shape->name_hint << '[' << k << ']';
    // DEBUGDEBUGDEBUG
    // Bind_(buffer_dense_shape[k],
    //       cast(buffer_dense_shape[k].dtype(),
    //            LoadNode::make(tvm_shape_type, v_shape, IntImm(DataType::Int(32), k),
    //            const_true(1),
    //                           tir::kAll)),
    //       field_name.str(), true);
  }
  // strides field
  Var v_strides(arg_name + ".strides", DataType::Handle());
  def_handle_dtype_.Set(v_strides, tir::TypeAnnotation(tvm_shape_type));
  init_nest_.emplace_back(LetStmtNode::make(
      v_strides, TVMArrayGet(DataType::Handle(), handle, intrinsic::kArrStrides), nop));
  PrimExpr is_null = CallNode::make(DataType::Bool(1), intrinsic::tvm_handle_is_null, {v_strides},
                                    CallNode::PureIntrinsic);
  if (buffer->strides.size() == 0) {
    // Assert the buffer is compact
    DataType stype = buffer->DefaultIndexType();
    PrimExpr expect_stride = make_const(stype, 1);
    Array<PrimExpr> conds;
    for (size_t i = buffer->shape->ndim(); i != 0; --i) {
      size_t k = i - 1;
      PrimExpr svalue =
          cast(stype, LoadNode::make(tvm_shape_type, v_strides, IntImm(DataType::Int(32), k),
                                     const_true(1), tir::kAll));
      conds.push_back(expect_stride == svalue);
      expect_stride = expect_stride * buffer_dense_shape[k];
    }
    std::ostringstream stride_err_msg;
    stride_err_msg << arg_name << ".strides:"
                   << " expected to be compact array";
    if (conds.size() != 0) {
      Stmt check = AssertStmtNode::make(arith::ComputeReduce<tir::AndNode>(conds, PrimExpr()),
                                        stride_err_msg.str(), EvaluateNode::make(0));
      check = IfThenElseNode::make(NotNode::make(is_null), check, Stmt());
      // DEBUGDEBUGDEBUG
      // asserts_.emplace_back(SeqStmt({check, EvaluateNode::make(0)}));
    }
  } else if (buffer->buffer_type == kAutoBroadcast) {
    DataType stype = buffer->DefaultIndexType();
    PrimExpr stride = make_const(stype, 1);
    for (size_t i = buffer->shape->ndim(); i != 0; --i) {
      size_t k = i - 1;
      std::ostringstream field_name;
      field_name << v_strides->name_hint << '[' << k << ']';
      PrimExpr value = cast(buffer_dense_shape[k].dtype(),
                            LoadNode::make(tvm_shape_type, v_strides, IntImm(DataType::Int(32), k),
                                           const_true(1), tir::kAll));
      value = tvm::if_then_else(is_null, stride, value);
      value = tvm::if_then_else(buffer_dense_shape[k] == 1, 0, value);
      Bind_(buffer->strides[k], value, field_name.str(), true);
      stride = Simplify(stride * buffer_dense_shape[k]);
    }
  } else {
    std::ostringstream stride_null_err_msg;
    stride_null_err_msg << arg_name << ".strides: expected non-null strides.";
    asserts_.emplace_back(
        AssertStmtNode::make(NotNode::make(is_null), stride_null_err_msg.str(), nop));

    for (size_t k = 0; k < buffer->strides.size(); ++k) {
      std::ostringstream field_name;
      field_name << v_strides->name_hint << '[' << k << ']';
      Bind_(buffer->strides[k],
            cast(buffer_dense_shape[k].dtype(),
                 LoadNode::make(tvm_shape_type, v_strides, IntImm(DataType::Int(32), k),
                                const_true(1), tir::kAll)),
            field_name.str(), true);
    }
  }
  // Byte_offset field.
  int data_bytes = GetVectorBytes(buffer->dtype);
  int64_t const_offset;
  if (arith::GetConst(buffer->elem_offset, &const_offset)) {
    Bind_(make_const(DataType::UInt(64), const_offset * data_bytes),
          TVMArrayGet(DataType::UInt(64), handle, intrinsic::kArrByteOffset),
          arg_name + ".byte_offset", true);
  } else {
    if (Bind_(buffer->elem_offset,
              cast(buffer->elem_offset.dtype(),
                   (TVMArrayGet(DataType::UInt(64), handle, intrinsic::kArrByteOffset) /
                    make_const(DataType::UInt(64), data_bytes))),
              arg_name + ".elem_offset", true)) {
      if (buffer->offset_factor > 1) {
        PrimExpr offset = buffer->elem_offset;
        PrimExpr factor = make_const(offset.dtype(), buffer->offset_factor);
        PrimExpr zero = make_zero(offset.dtype());
        BinderAddAssert(truncmod(offset, factor) == zero, arg_name + ".elem_offset", &asserts_);
      }
    }
  }
  // device info.
  Bind_(device_type, TVMArrayGet(DataType::Int(32), handle, intrinsic::kArrDeviceType),
        arg_name + ".device_type", true);
  Bind_(device_id, TVMArrayGet(DataType::Int(32), handle, intrinsic::kArrDeviceId),
        arg_name + ".device_id", true);
}

}  // namespace tir
}  // namespace tvm
