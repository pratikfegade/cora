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
 * \file tvm/arith/const_int_bound.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr_functor.h>

#include <algorithm>

#include "int_operator.h"
#include "pattern_match.h"

namespace tvm {
namespace arith {

using namespace tir;

TVM_REGISTER_NODE_TYPE(ConstIntBoundNode);

ConstIntBound::ConstIntBound(int64_t min_value, int64_t max_value) {
  auto node = make_object<ConstIntBoundNode>();
  node->min_value = min_value;
  node->max_value = max_value;
  data_ = std::move(node);
}

ConstIntBound MakeConstIntBound(int64_t min_value, int64_t max_value) {
  return ConstIntBound(min_value, max_value);
}

TVM_REGISTER_GLOBAL("arith.ConstIntBound").set_body_typed(MakeConstIntBound);

inline void PrintBoundValue(std::ostream& os, int64_t val) {
  if (val == ConstIntBound::kPosInf) {
    os << "pos_inf";
  } else if (val == ConstIntBound::kNegInf) {
    os << "neg_inf";
  } else {
    os << val;
  }
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ConstIntBoundNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ConstIntBoundNode*>(node.get());
      p->stream << "ConstIntBound[";
      PrintBoundValue(p->stream, op->min_value);
      p->stream << ',';
      PrintBoundValue(p->stream, op->max_value);
      p->stream << ']';
    });

// internal entry for const int bound
struct ConstIntBoundAnalyzer::Entry {
  int64_t min_value;
  int64_t max_value;

  bool is_const(int64_t value) const { return min_value == max_value && min_value == value; }

  bool operator==(const Entry& other) const {
    return min_value == other.min_value && max_value == other.max_value;
  }
};

class ConstIntBoundAnalyzer::Impl
    : public ExprFunctor<ConstIntBoundAnalyzer::Entry(const PrimExpr&)> {
 public:
  /*! \brief additional bound info about expr \in bound */
  struct BoundInfo {
    /*! \brief The expr */
    PrimExpr expr;
    /*! \brief The additional bound */
    Entry bound;

    BoundInfo() {}
    BoundInfo(PrimExpr expr, Entry bound) : expr(expr), bound(bound) {}
  };

  void Bind(const Var& var, const Range& range) {
    Entry a = VisitExpr(range->min);
    Entry b = VisitExpr(range->extent);
    Entry ret;
    ret.min_value = a.min_value;
    ret.max_value = InfAwareAdd(a.max_value, InfAwareAdd(b.max_value, -1));
    Update(var, ret, false);
  }

  void Update(const Var& var, const Entry& info, bool override) {
    if (!override) {
      auto it = var_map_.find(var);
      if (it != var_map_.end()) {
        CHECK(it->second == info) << "Trying to update var \'" << var << "\'"
                                  << " with a different const bound: "
                                  << "original="
                                  << ConstIntBound(it->second.min_value, it->second.max_value)
                                  << ", new=" << ConstIntBound(info.min_value, info.max_value);
      }
    }
    var_map_[var] = info;
  }

  void Update(const Var& var, const ConstIntBound& info, bool override) {
    Update(var, MakeBound(info->min_value, info->max_value), override);
  }

  // Override visitor behaviors
  Entry VisitExprDefault_(const Object* op) final {
    return Everything(static_cast<const PrimExprNode*>(op)->dtype);
  }

  Entry VisitExpr(const PrimExpr& expr) final {
    // std::cout << "[CIB]   Expr " << expr << std::endl;
    Entry res = ExprFunctor::VisitExpr(expr);
    // a linear search over additional info
    // assume we won't have a lot of conditions
    for (const BoundInfo& info : additional_info_) {
      if (tir::Equal(expr, info.expr)) {
        res = Intersect(res, info.bound);
      }
    }
    return res;
  }

  Entry VisitExpr_(const CastNode* op) final {
    Entry a = VisitExpr(op->value);
    Entry b = Everything(op->dtype);
    return Intersect(a, b);
  }

  Entry VisitExpr_(const IntImmNode* op) final { return MakeBound(op->value, op->value); }

  Entry VisitExpr_(const AddNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    Entry ret;
    ret.min_value = InfAwareAdd(a.min_value, b.min_value);
    ret.max_value = InfAwareAdd(a.max_value, b.max_value);

    // std::cout << "    CIB Add A " << a.min_value << " " << a.max_value << std::endl;
    // std::cout << "    CIB Add B " << b.min_value << " " << b.max_value << std::endl;
    // std::cout << "    CIB Add Ret " << ret.min_value << " " << ret.max_value << std::endl;

    return ret;
  }

  Entry VisitExpr_(const SubNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    Entry ret;
    ret.min_value = InfAwareAdd(a.min_value, -b.max_value);
    ret.max_value = InfAwareAdd(a.max_value, -b.min_value);

    // std::cout << "    CIB Sub A " << a.min_value << " " << a.max_value << std::endl;
    // std::cout << "    CIB Sub B " << b.min_value << " " << b.max_value << std::endl;
    // std::cout << "    CIB Sub Ret " << ret.min_value << " " << ret.max_value << std::endl;

    return ret;
  }

  Entry VisitExpr_(const MulNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    return BinaryOpBoundry(a, b, InfAwareMul);
  }

  Entry VisitExpr_(const DivNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    CHECK(!b.is_const(0)) << "divide by zero";
    // assume no division by 0
    if (b.min_value == 0) b.min_value = 1;
    if (b.max_value == 0) b.max_value = -1;
    Entry ret = BinaryOpBoundry(a, b, InfAwareDiv);

    if (b.min_value < 0 && b.max_value > 0) {
      int64_t v1 = InfAwareDiv(a.min_value, 1);
      int64_t v2 = InfAwareDiv(a.max_value, 1);
      int64_t v3 = InfAwareDiv(a.min_value, -1);
      int64_t v4 = InfAwareDiv(a.max_value, -1);
      ret.min_value = std::min(ret.min_value, std::min(std::min(std::min(v1, v2), v3), v4));
      ret.max_value = std::max(ret.max_value, std::max(std::max(std::max(v1, v2), v3), v4));
    }
    return ret;
  }

  Entry VisitExpr_(const ModNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    if (b.min_value > 0) {
      int64_t b_max_cap = InfAwareAdd(b.max_value, -1);
      if (a.min_value >= 0) {
        // 0 <= [a_min, a_max] < b_min
        if (a.max_value < b.min_value) return a;
        // other case, we can get close to 0
        return MakeBound(0, std::min(a.max_value, b_max_cap));
      } else {
        return MakeBound(std::max(a.min_value, -b_max_cap),
                         std::min(std::max(a.max_value, (int64_t)0), b_max_cap));
      }
    } else {
      CHECK(!b.is_const(0)) << "mod by zero";
      // mod by negative value is rare,
      // and we just use the simpliest rule.
      return Everything(op->dtype);
    }
  }

  Entry VisitExpr_(const FloorDivNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);

    // std::cout << " CIB FDiv InpA " << op->a << std::endl;
    // std::cout << " CIB FDiv InpB " << op->b << std::endl;
    // std::cout << "    CIB FDiv A " << a.min_value << " " << a.max_value << std::endl;
    // std::cout << "    CIB FDiv B " << b.min_value << " " << b.max_value << std::endl;

    CHECK(!b.is_const(0)) << "floordiv by zero";
    // assume no division by 0
    if (b.min_value == 0) b.min_value = 1;
    if (b.max_value == 0) b.max_value = -1;
    Entry ret = BinaryOpBoundry(a, b, InfAwareFloorDiv);

    if (b.min_value < 0 && b.max_value > 0) {
      int64_t v1 = InfAwareFloorDiv(a.min_value, 1);
      int64_t v2 = InfAwareFloorDiv(a.max_value, 1);
      int64_t v3 = InfAwareFloorDiv(a.min_value, -1);
      int64_t v4 = InfAwareFloorDiv(a.max_value, -1);
      ret.min_value = std::min(ret.min_value, std::min(std::min(std::min(v1, v2), v3), v4));
      ret.max_value = std::max(ret.max_value, std::max(std::max(std::max(v1, v2), v3), v4));
    }

    // std::cout << "    CIB FDiv Ret " << ret.min_value << " " << ret.max_value << std::endl;
    return ret;
  }

  Entry VisitExpr_(const FloorModNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);

    // std::cout << " CIB FMod Inp " << GetRef<PrimExpr>(op) << std::endl;
    // std::cout << "    CIB FMod A " << a.min_value << " " << a.max_value << std::endl;
    // std::cout << "    CIB FMod B " << b.min_value << " " << b.max_value << std::endl;

    if (b.min_value > 0) {
      int64_t b_max_cap = InfAwareAdd(b.max_value, -1);
      if (a.min_value >= 0) {
        // 0 <= [a_min, a_max] < b_min
        if (a.max_value < b.min_value) return a;
        // other case, we can get close to 0
        // std::cout << "     Ret1 " << std::min(a.max_value, b_max_cap) << std::endl;
        return MakeBound(0, std::min(a.max_value, b_max_cap));
      } else {
        // std::cout << "     Ret2 " << b_max_cap << std::endl;
        return MakeBound(0, b_max_cap);
      }
    } else {
      CHECK(!b.is_const(0)) << "floormod by zero";
      // mod by negative value is rare,
      // and we just use the simpliest rule.
      return Everything(op->dtype);
    }
  }

  Entry VisitExpr_(const MinNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    Entry ret;
    ret.min_value = std::min(a.min_value, b.min_value);
    ret.max_value = std::min(a.max_value, b.max_value);
    return ret;
  }

  Entry VisitExpr_(const MaxNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    Entry ret;
    ret.min_value = std::max(a.min_value, b.min_value);
    ret.max_value = std::max(a.max_value, b.max_value);
    return ret;
  }

  Entry VisitExpr_(const SelectNode* op) final {
    Entry a = VisitExpr(op->true_value);
    Entry b = VisitExpr(op->false_value);
    return Union(a, b);
  }

  Entry VisitExpr_(const CallNode* op) final {
    // only special handle >> and & which can be
    // used for index calculation.
    if (op->is_intrinsic(CallNode::shift_right)) {
      return VisitRightShift(op);
    } else if (op->is_intrinsic(CallNode::bitwise_and)) {
      return VisitBitwiseAnd(op);
    } else if (auto ufun = op->func.as<UninterpFunNode>()) {
      // std::cout << " CIB UFUN Range " << op->func << " " << ufun->range << std::endl;
      Entry ret;
      Entry a = this->VisitExpr(ufun->range->min);
      Entry b = this->VisitExpr(ufun->range->max_inclusive());
      ret.min_value = a.min_value;
      ret.max_value = b.max_value;

      // std::cout << "    CIB UFUN Range " << ufun->range << std::endl;
      // std::cout << "    CIB UFUN Min Range " << a.min_value << " " << a.max_value << std::endl;
      // std::cout << "    CIB UFUN Max Range " << b.min_value << " " << b.max_value << std::endl;

      return ret;
    } else {
      return Everything(op->dtype);
    }
  }

  Entry VisitExpr_(const VarNode* op) final {
    Var v = GetRef<Var>(op);
    auto it = var_map_.find(v);
    if (it != var_map_.end()) {
      return it->second;
    } else {
      return Everything(op->dtype);
    }
  }

  Entry VisitExpr_(const SizeVarNode* op) final {
    SizeVar v = GetRef<SizeVar>(op);
    auto it = var_map_.find(v);
    if (it != var_map_.end()) {
      return it->second;
    } else {
      return MakeBound(0, kPosInf);
    }
  }

  Entry VisitRightShift(const CallNode* op) {
    Entry a = VisitExpr(op->args[0]);
    Entry b = VisitExpr(op->args[1]);
    return BinaryOpBoundry(a, b, InfAwareRightShift);
  }

  Entry VisitBitwiseAnd(const CallNode* op) {
    Entry a = VisitExpr(op->args[0]);
    Entry b = VisitExpr(op->args[1]);
    // handle positive index case.
    if (a.min_value >= 0 && b.min_value >= 0) {
      return MakeBound(0, std::min(a.max_value, b.max_value));
    } else {
      if (b.min_value >= 0) {
        return MakeBound(0, b.max_value);
      }
      if (a.min_value >= 0) {
        return MakeBound(0, a.max_value);
      }
      return Everything(op->dtype);
    }
  }

  std::function<void()> EnterConstraint(const PrimExpr& constraint) {
    std::vector<BoundInfo> info = DetectBoundInfo(constraint);
    if (info.size() == 0) return nullptr;
    size_t old_size = additional_info_.size();
    additional_info_.insert(additional_info_.end(), info.begin(), info.end());
    size_t new_size = old_size + info.size();
    auto frecover = [old_size, new_size, this]() {
      CHECK_EQ(additional_info_.size(), new_size);
      additional_info_.resize(old_size);
    };
    return frecover;
  }

 private:
  // internal variable map
  std::unordered_map<Var, Entry, ObjectHash, ObjectEqual> var_map_;
  // additional bound info
  std::vector<BoundInfo> additional_info_;
  // constants: the limit value means umlimited
  // NOTE: kNegInf/kPosInf are used to represent infinity.
  static const constexpr int64_t kNegInf = ConstIntBound::kNegInf;
  static const constexpr int64_t kPosInf = ConstIntBound::kPosInf;
  static_assert(-kNegInf == kPosInf, "invariant of inf");
  // internal helper functions
  /*!
   * \brief Get boundary of binary op who are monotonic wrt to one argument.
   * \param param a The entry of the left operand.
   * \param param a The entry of the right operand.
   * \param op The operator.
   * \tparam F the operator function type.
   * \return The result.
   */
  template <typename F>
  static Entry BinaryOpBoundry(Entry a, Entry b, const F& op) {
    Entry ret;
    // The boundary point must be shihft of the original boundary.
    int64_t v1 = op(a.min_value, b.min_value);
    int64_t v2 = op(a.max_value, b.max_value);
    int64_t v3 = op(a.min_value, b.max_value);
    int64_t v4 = op(a.max_value, b.min_value);
    ret.min_value = std::min(std::min(std::min(v1, v2), v3), v4);
    ret.max_value = std::max(std::max(std::max(v1, v2), v3), v4);
    return ret;
  }
  /*!
   * \brief Compute x + y, aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareAdd(int64_t x, int64_t y) {
    // std::cout << "       InfAdd A " << x << " " << y << " " << kNegInf << " " << kPosInf
    // << std::endl;

    if (x == kPosInf) {
      CHECK(y != kNegInf);
      // std::cout << "       InfAdd Ret1" << std::endl;
      return kPosInf;
    }
    if (x == kNegInf) {
      CHECK(y != kPosInf);
      // std::cout << "       InfAdd Ret2" << std::endl;
      return kNegInf;
    }
    if (y == kPosInf || y == kNegInf) {
      // std::cout << "       InfAdd Ret3" << std::endl;
      return y;
    }
    if (WillOverflow<AddNode>(x, y, kNegInf, kPosInf)) {
      if (x > 0) {
        // std::cout << "       InfAdd Ret4" << std::endl;
        return kPosInf;
      }
      // std::cout << "       InfAdd Ret5" << std::endl;
      return kNegInf;
    }
    // std::cout << "       InfAdd Ret6" << std::endl;
    return x + y;
  }
  /*!
   * \brief Compute x * y, aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareMul(int64_t x, int64_t y) {
    if (!WillOverflow<MulNode>(x, y, kNegInf, kPosInf)) return x * y;
    if ((x > 0 && y > 0) || (x < 0 && y < 0)) return kPosInf;
    return kNegInf;
  }
  /*!
   * \brief Compute x / y, aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareDiv(int64_t x, int64_t y) {
    CHECK_NE(y, 0);
    if (x == kPosInf || x == kNegInf) {
      if (y > 0) return x;
      return -x;
    }
    return x / y;
  }
  /*!
   * \brief Compute floodiv(x, y), aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareFloorDiv(int64_t x, int64_t y) {
    CHECK_NE(y, 0);
    if (x == kPosInf || x == kNegInf) {
      if (y > 0) return x;
      return -x;
    }
    return floordiv(x, y);
  }
  /*!
   * \brief Compute x / y, aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareRightShift(int64_t x, int64_t y) {
    if (x == kPosInf || x == kNegInf) return x;
    return x >> y;
  }
  /*!
   * \brief Make a new bound entry.
   */
  static Entry MakeBound(int64_t min_value, int64_t max_value) {
    Entry e;
    e.min_value = min_value;
    e.max_value = max_value;
    return e;
  }
  /*!
   * \brief Create union of two sets.
   * \param a The left operand.
   * \param b the right operand.
   */
  static Entry Union(Entry a, Entry b) {
    Entry ret;
    ret.min_value = std::min(a.min_value, b.min_value);
    ret.max_value = std::max(a.max_value, b.max_value);
    return ret;
  }
  /*!
   * \brief Create intersect of two sets.
   * \param a The left operand.
   * \param b the right operand.
   */
  static Entry Intersect(Entry a, Entry b) {
    Entry ret;
    ret.min_value = std::max(a.min_value, b.min_value);
    ret.max_value = std::min(a.max_value, b.max_value);
    return ret;
  }
  /*!
   * \brief return everything dtype can represent.
   * \param dtype The data type.
   * \return Bound that represent everything dtype can represent.
   */
  static Entry Everything(DataType dtype) {
    if (!dtype.is_int() && !dtype.is_uint()) {
      return MakeBound(kNegInf, kPosInf);
    }
    Entry ret;
    int64_t vbits = dtype.bits() - static_cast<int>(dtype.is_int());
    if (dtype.is_uint()) {
      ret.min_value = 0;
    } else {
      if (vbits >= 63) {
        ret.min_value = kNegInf;
      } else {
        ret.min_value = -(static_cast<int64_t>(1) << vbits);
      }
    }
    if (vbits >= 63) {
      ret.max_value = kPosInf;
    } else {
      ret.max_value = (static_cast<int64_t>(1) << vbits) - 1;
    }
    return ret;
  }

  /*!
   * \brief Detect additional constant bound from cond, if any
   * \param cond The constraint condition.
   * \return List of detected bounds.
   */
  static std::vector<BoundInfo> DetectBoundInfo(const PrimExpr& cond) {
    PVar<PrimExpr> x, y;
    PVar<IntImm> c;
    // NOTE: canonical form always use <= or <
    if ((c <= x).Match(cond)) {
      return {BoundInfo(x.Eval(), MakeBound(c.Eval()->value, kPosInf))};
    }
    if ((c < x).Match(cond)) {
      return {BoundInfo(x.Eval(), MakeBound(c.Eval()->value + 1, kPosInf))};
    }
    if ((x <= c).Match(cond)) {
      return {BoundInfo(x.Eval(), MakeBound(kNegInf, c.Eval()->value))};
    }
    if ((x < c).Match(cond)) {
      return {BoundInfo(x.Eval(), MakeBound(kNegInf, c.Eval()->value - 1))};
    }
    if ((x && y).Match(cond)) {
      auto ret1 = DetectBoundInfo(x.Eval());
      auto ret2 = DetectBoundInfo(y.Eval());
      ret1.insert(ret1.end(), ret2.begin(), ret2.end());
      return ret1;
    }
    return {};
  }
};

ConstIntBound ConstIntBoundAnalyzer::operator()(const PrimExpr& expr) {
  // std::cout << "CIB Operator " << expr << std::endl;
  Entry ret = impl_->VisitExpr(expr);
  // std::cout << "    CIB Ret " << ret.min_value << " " << ret.max_value << std::endl;
  return ConstIntBound(ret.min_value, ret.max_value);
}

void ConstIntBoundAnalyzer::Update(const Var& var, const ConstIntBound& info, bool override) {
  impl_->Update(var, info, override);
}

void ConstIntBoundAnalyzer::Bind(const Var& var, const Range& range) { impl_->Bind(var, range); }

std::function<void()> ConstIntBoundAnalyzer::EnterConstraint(const PrimExpr& constraint) {
  return impl_->EnterConstraint(constraint);
}

ConstIntBoundAnalyzer::ConstIntBoundAnalyzer(Analyzer* parent) : impl_(new Impl()) {}

ConstIntBoundAnalyzer::~ConstIntBoundAnalyzer() { delete impl_; }

}  // namespace arith
}  // namespace tvm
