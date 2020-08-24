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
 * \file intrin_rule_llvm.cc
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/tir/op.h>
#include "intrin_rule_llvm.h"

namespace tvm {
namespace codegen {
namespace llvm {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.prefetch")
.set_body(DispatchLLVMIntrin<::llvm::Intrinsic::prefetch, 0>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.exp")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::exp, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.fma")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.log")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::log, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.sqrt")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::sqrt, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.floor")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::floor, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.ceil")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::ceil, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.trunc")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::trunc, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.fabs")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::fabs, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.round")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::round, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.nearbyint")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.tanh")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  using tir::make_const;
  using tir::make_zero;
  PrimExpr e = targs[0];
  const tir::CallNode* call = e.as<tir::CallNode>();
  CHECK(call != nullptr);
  const PrimExpr& x = call->args[0];
  PrimExpr one = make_const(x.dtype(), 1);
  PrimExpr two = make_const(x.dtype(), 2);
  PrimExpr neg_two = make_const(x.dtype(), -2);

  PrimExpr exp_neg2x = tir::CallNode::make(
      x.dtype(), "exp", {neg_two * x}, tir::CallNode::PureIntrinsic);
  PrimExpr exp_pos2x = tir::CallNode::make(
      x.dtype(), "exp", {two * x}, tir::CallNode::PureIntrinsic);

  PrimExpr tanh_pos = (one - exp_neg2x) / (one + exp_neg2x);
  PrimExpr tanh_neg = (exp_pos2x - one) / (exp_pos2x + one);
  *rv = tir::SelectNode::make(
      x >= make_zero(x.dtype()), tanh_pos, tanh_neg);
});

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.fast_tanh")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  using tir::make_const;
  using tir::make_zero;
  PrimExpr e = targs[0];
  const tir::CallNode* call = e.as<tir::CallNode>();
  CHECK(call != nullptr);
  const PrimExpr& in = call->args[0];

  // Clamp the inputs to the range [-9, 9] since anything outside
  // this range is +/-1.0f in single-precision.
  auto x = tir::MaxNode::make(tir::MinNode::make(in, make_const(call->dtype, 9.0)), make_const(call->dtype, -9.0));

  // The monomial coefficients of the numerator polynomial (odd).
  auto alpha_1 = make_const(call->dtype, 4.89352455891786e-03);
  auto alpha_3 = make_const(call->dtype, 6.37261928875436e-04);
  auto alpha_5 = make_const(call->dtype, 1.48572235717979e-05);
  auto alpha_7 = make_const(call->dtype, 5.12229709037114e-08);
  auto alpha_9 = make_const(call->dtype, -8.60467152213735e-11);
  auto alpha_11 = make_const(call->dtype, 2.00018790482477e-13);
  auto alpha_13 = make_const(call->dtype, -2.76076847742355e-16);

  // The monomial coefficients of the denominator polynomial (even).
  auto beta_0 = make_const(call->dtype, 4.89352518554385e-03);
  auto beta_2 = make_const(call->dtype, 2.26843463243900e-03);
  auto beta_4 = make_const(call->dtype, 1.18534705686654e-04);
  auto beta_6 = make_const(call->dtype, 1.19825839466702e-06);

  auto x2 = x * x;
  auto p = x2 * alpha_13 + alpha_11;
  p = x2 * p + alpha_9;
  p = x2 * p + alpha_7;
  p = x2 * p + alpha_5;
  p = x2 * p + alpha_3;
  p = x2 * p + alpha_1;
  p = x * p;

  auto q = x2 * beta_6 + beta_4;
  q = x2 * q + beta_2;
  q = x2 * q + beta_0;
  PrimExpr ret = (p / q);
  *rv = ret;
});

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.fast_exp")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  using tir::make_const;
  using tir::make_zero;
  PrimExpr e = targs[0];
  const tir::CallNode* call = e.as<tir::CallNode>();
  CHECK(call != nullptr);
  const PrimExpr& in = call->args[0];

  auto x_hi = make_const(DataType::Float(32), 88.3762626647950f);
  auto x_lo = make_const(DataType::Float(32), -88.3762626647949f);
  auto log2e = make_const(DataType::Float(32), 1.44269504088896341f);
  auto ln2 = make_const(DataType::Float(32), 0.6931471805599453f);
  PrimExpr p[6] = {make_const(DataType::Float(32), 1.9875691500E-4f),
                   make_const(DataType::Float(32), 1.3981999507E-3f),
                   make_const(DataType::Float(32), 8.3334519073E-3f),
                   make_const(DataType::Float(32), 4.1665795894E-2f),
                   make_const(DataType::Float(32), 1.6666665459E-1f),
                   make_const(DataType::Float(32), 5.0000001201E-1f)};
  auto one = make_const(DataType::Float(32), 1.0f);
  auto one_half = make_const(DataType::Float(32), 0.5f);
  auto b = make_const(DataType::Float(32), 127.0f);

  // clamp x
  auto x = tir::MaxNode::make(tir::MinNode::make(in, x_hi), x_lo);
  // integer part
  auto n = tir::CallNode::make(in->dtype, "floor", {x * log2e + one_half}, tir::CallNode::PureIntrinsic);
  // fractional part
  auto f = x - n * ln2;
  auto y =
    (((((p[0] * f + p[1]) * f + p[2]) * f + p[3]) * f + p[4]) * f + p[5]) * f * f + f + one;
  // Return 2^m * exp(r).
  auto ef =
    tvm::reinterpret(DataType::Float(32), ::tvm::cast(DataType::Int(32), n + b) << 23);
  auto ret = tir::MaxNode::make(ef * y, in);  // NOLINT(*)
  *rv = ret;
});

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.fast_sigmoid")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  using tir::make_const;
  using tir::make_zero;
  PrimExpr e = targs[0];
  const tir::CallNode* call = e.as<tir::CallNode>();
  CHECK(call != nullptr);
  const PrimExpr& in = call->args[0];

  auto one = make_const(in->dtype, 1.0f);
  auto exp = tir::CallNode::make(in->dtype, "fast_exp", {-in}, tir::CallNode::PureIntrinsic);
  auto ret = one / (one - exp);
  *rv = ret;
});

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.pow")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::pow, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.popcount")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::ctpop, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.cos")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::cos, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.sin")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::sin, 1>);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
