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
 * \file int_set.cc
 * \brief The integer set functions
 */
#include <tvm/arith/int_set.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/uf_equality.h>
#include <tvm/tir/uninterp_fun.h>

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "../tir/ir/var_replacer.h"
#include "interval_set.h"
#include "pattern_match.h"
#include "projection_set.h"

namespace tvm {
namespace arith {

using tir::is_one;
using tir::is_zero;
using tir::make_const;
using tir::make_zero;

PrimExpr SymbolicLimits::pos_inf_ = Var("pos_inf", DataType::Handle());
PrimExpr SymbolicLimits::neg_inf_ = Var("neg_inf", DataType::Handle());

ProjectionSet::ProjectionSet(UninterpFun ufun, Map<te::Dimension, IntSet> arguments) {
  for (auto dim : ufun->dimensions) {
    CHECK(arguments.count(dim)) << dim->name;
  }

  auto node = make_object<ProjectionSetNode>();
  node->ufun = std::move(ufun);
  node->arguments = std::move(arguments);
  data_ = std::move(node);
}

IntervalSet::IntervalSet(PrimExpr min_value, PrimExpr max_value) {
  auto node = make_object<IntervalSetNode>();
  node->min_value = std::move(min_value);
  node->max_value = std::move(max_value);
  data_ = std::move(node);
}

IntervalSet MakeIntervalSet(PrimExpr min_value, PrimExpr max_value) {
  return IntervalSet(min_value, max_value);
}

TVM_REGISTER_GLOBAL("arith.IntervalSet").set_body_typed(MakeIntervalSet);

IntervalSet Intersect(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  PrimExpr max_value = min(a->max_value, b->max_value);
  PrimExpr min_value = max(a->min_value, b->min_value);
  if ((max_value.dtype().is_int() || max_value.dtype().is_uint()) &&
      (min_value.dtype().is_int() || min_value.dtype().is_uint()) &&
      analyzer->CanProveGreaterEqual(min_value - max_value, 1)) {
    return IntervalSet::Empty();
  } else {
    return IntervalSet(min_value, max_value);
  }
}

IntSet Union(Analyzer* analyzer, IntSet a, IntSet b) {
  if (a->IsInstance<IntervalSetNode>() && b->IsInstance<IntervalSetNode>()) {
    return Union(analyzer, Downcast<IntervalSet>(a), Downcast<IntervalSet>(b));
  } else if (a->IsInstance<ProjectionSetNode>() && b->IsInstance<ProjectionSetNode>()) {
    return Union(analyzer, Downcast<ProjectionSet>(a), Downcast<ProjectionSet>(b));
  } else {
    PrimExpr max_value = max(a.max(), b.max());
    PrimExpr min_value = min(a.min(), b.min());
    return IntervalSet(min_value, max_value);
  }
}

IntervalSet Union(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  PrimExpr max_value = max(a->max_value, b->max_value);
  PrimExpr min_value = min(a->min_value, b->min_value);
  return IntervalSet(min_value, max_value);
}

IntSet Union(Analyzer* analyzer, ProjectionSet a, ProjectionSet b) {
  auto mapping_and_equals = CheckUninterpFunEquality(a->ufun, b->ufun);
  if (mapping_and_equals.equals) {
    Map<Dimension, IntSet> arg_unions;
    for (const auto& p : a->arguments) {
      const auto& dim = p.first;
      const auto& a_arg = p.second;
      const auto& b_arg = b->arguments.at(dim);
      arg_unions.Set(dim, Union(analyzer, a_arg, b_arg));
    }
    return ProjectionSet(a->ufun, arg_unions);
  } else
    return IntervalSet::Everything();
}

IntSet ReplaceIntSet(IntSet set, std::unordered_map<const VarNode*, PrimExpr> vsub) {
  VarReplacer replacer(vsub);
  if (auto iset = set.as<IntervalSetNode>()) {
    return IntervalSet(replacer(iset->min_value), replacer(iset->max_value));
  } else if (auto pset = set.as<ProjectionSetNode>()) {
    Map<Dimension, IntSet> arguments;
    for (const auto& it : pset->arguments) {
      arguments.Set(it.first, ReplaceIntSet(it.second, vsub));
    }
    return ProjectionSet(pset->ufun, arguments);
  } else {
    CHECK(false) << "No such Intset " << set;
    return {};
  }
}

// type traits
template <typename OP>
struct is_logical_op {
  static const bool value = false;
};

#define TVM_DECLARE_LOGICAL_OP(OP)  \
  template <>                       \
  struct is_logical_op<tir::OP> {   \
    static const bool value = true; \
  };

TVM_DECLARE_LOGICAL_OP(AndNode);
TVM_DECLARE_LOGICAL_OP(OrNode);
TVM_DECLARE_LOGICAL_OP(EQNode);
TVM_DECLARE_LOGICAL_OP(NENode);
TVM_DECLARE_LOGICAL_OP(GENode);
TVM_DECLARE_LOGICAL_OP(GTNode);
TVM_DECLARE_LOGICAL_OP(LENode);
TVM_DECLARE_LOGICAL_OP(LTNode);
TVM_DECLARE_LOGICAL_OP(NotNode);

/*!
 * \brief Combine two interval set under arithmetic operations.
 * \note this can possibly relax the set.
 */
template <typename Op>
inline IntervalSet Combine(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    PrimExpr res = TryConstFold<Op>(a->min_value, b->min_value);
    if (!res.defined()) res = Op::make(a->min_value, b->min_value);
    return IntervalSet::SinglePoint(res);
  }
  if (is_logical_op<Op>::value) {
    return IntervalSet(make_const(a->min_value.dtype(), 0), make_const(a->min_value.dtype(), 1));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  if (a->IsEverything()) return a;
  if (b->IsEverything()) return b;
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::AddNode>(Analyzer* analyer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value + b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  PrimExpr min_value =
      a->HasLowerBound() && b->HasLowerBound() ? a->min_value + b->min_value : neg_inf();
  PrimExpr max_value =
      a->HasUpperBound() && b->HasUpperBound() ? a->max_value + b->max_value : pos_inf();
  return IntervalSet(min_value, max_value);
}

template <>
inline IntervalSet Combine<tir::SubNode>(Analyzer* analyer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value - b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  PrimExpr min_value =
      a->HasLowerBound() && b->HasUpperBound() ? a->min_value - b->max_value : neg_inf();
  PrimExpr max_value =
      a->HasUpperBound() && b->HasLowerBound() ? a->max_value - b->min_value : pos_inf();
  return IntervalSet(min_value, max_value);
}

template <>
inline IntervalSet Combine<tir::MulNode>(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value * b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  if (a->IsSinglePoint()) {
    std::swap(a, b);
  }
  if (b->IsSinglePoint()) {
    if (is_zero(b->min_value)) return b;
    if (is_one(b->min_value)) return a;
    if (analyzer->CanProveGreaterEqual(b->min_value, 0)) {
      PrimExpr min_value = a->HasLowerBound() ? a->min_value * b->min_value : neg_inf();
      PrimExpr max_value = a->HasUpperBound() ? a->max_value * b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (analyzer->CanProveGreaterEqual(-b->min_value, 1)) {
      PrimExpr min_value = a->HasUpperBound() ? a->max_value * b->min_value : neg_inf();
      PrimExpr max_value = a->HasLowerBound() ? a->min_value * b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (a->HasUpperBound() && a->HasLowerBound()) {
      using tir::SelectNode;
      PrimExpr sign = b->min_value >= make_zero(b->min_value.dtype().element_of());
      PrimExpr e1 = a->min_value * b->min_value;
      PrimExpr e2 = a->max_value * b->min_value;
      return IntervalSet(SelectNode::make(sign, e1, e2), SelectNode::make(sign, e2, e1));
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Mul";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::DivNode>(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value / b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  if (b->IsSinglePoint()) {
    if (is_zero(b->min_value)) {
      LOG(FATAL) << "Divide by zero in CombineInterval Div";
    }
    if (is_one(b->min_value)) return a;
    // no relaxation is needed in here due to set is inclusive
    if (analyzer->CanProveGreaterEqual(b->min_value, 0)) {
      PrimExpr min_value = a->HasLowerBound() ? a->min_value / b->min_value : neg_inf();
      PrimExpr max_value = a->HasUpperBound() ? a->max_value / b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (analyzer->CanProveGreaterEqual(-b->min_value, 1)) {
      PrimExpr min_value = a->HasUpperBound() ? a->max_value / b->min_value : neg_inf();
      PrimExpr max_value = a->HasLowerBound() ? a->min_value / b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (a->HasUpperBound() && a->HasLowerBound()) {
      using tir::SelectNode;
      PrimExpr sign = b->min_value >= make_zero(b->min_value.dtype().element_of());
      PrimExpr e1 = a->min_value / b->min_value;
      PrimExpr e2 = a->max_value / b->min_value;
      return IntervalSet(SelectNode::make(sign, e1, e2), SelectNode::make(sign, e2, e1));
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Div";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::ModNode>(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(truncmod(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;

  if (b->IsSinglePoint()) {
    const PrimExpr& divisor = b->min_value;
    if (is_zero(divisor)) {
      LOG(FATAL) << "Modular by zero in CombineInterval Mod";
    }
    // We need to add more bound constraints throughout the code.
    // The logic below assumes a is non-negative, which usually
    // is the case of our application.
    // TODO(tqchen): add bound constraints for a.
    if (analyzer->CanProveGreaterEqual(divisor, 0)) {
      return IntervalSet(make_zero(divisor.dtype()), divisor - 1);
    } else {
      PrimExpr bound = abs(divisor) - 1;
      return IntervalSet(-bound, bound);
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Mod";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::FloorDivNode>(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(floordiv(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  if (b->IsSinglePoint()) {
    if (is_zero(b->min_value)) {
      LOG(FATAL) << "Divide by zero in CombineInterval Div";
    }
    if (is_one(b->min_value)) return a;
    // no relaxation is needed in here due to set is inclusive
    if (analyzer->CanProveGreaterEqual(b->min_value, 0)) {
      PrimExpr min_value = a->HasLowerBound() ? floordiv(a->min_value, b->min_value) : neg_inf();
      PrimExpr max_value = a->HasUpperBound() ? floordiv(a->max_value, b->min_value) : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (analyzer->CanProveGreaterEqual(-b->min_value, 1)) {
      PrimExpr min_value = a->HasUpperBound() ? floordiv(a->max_value, b->min_value) : neg_inf();
      PrimExpr max_value = a->HasLowerBound() ? floordiv(a->min_value, b->min_value) : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (a->HasUpperBound() && a->HasLowerBound()) {
      using tir::SelectNode;
      PrimExpr sign = b->min_value >= make_zero(b->min_value.dtype().element_of());
      PrimExpr e1 = floordiv(a->min_value, b->min_value);
      PrimExpr e2 = floordiv(a->max_value, b->min_value);
      return IntervalSet(SelectNode::make(sign, e1, e2), SelectNode::make(sign, e2, e1));
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Div";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::FloorModNode>(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(floormod(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;

  if (b->IsSinglePoint()) {
    const PrimExpr& divisor = b->min_value;
    if (is_zero(divisor)) {
      LOG(FATAL) << "Modular by zero in CombineInterval Mod";
    }
    if (analyzer->CanProveGreaterEqual(divisor, 0)) {
      return IntervalSet(make_zero(divisor.dtype()), divisor - 1);
    } else {
      PrimExpr bound = abs(divisor) - 1;
      return IntervalSet(-bound, bound);
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Mod";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::MaxNode>(Analyzer* analzyer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(max(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  return IntervalSet(max(a->min_value, b->min_value), max(a->max_value, b->max_value));
}

template <>
inline IntervalSet Combine<tir::MinNode>(Analyzer* analzyer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(min(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  return IntervalSet(min(a->min_value, b->min_value), min(a->max_value, b->max_value));
}

template <typename Op>
inline IntSet CombineIntSets(Analyzer* analyzer, IntSet a, IntSet b) {
  if (a.as<IntervalSetNode>() && b.as<IntervalSetNode>()) {
    return Combine<Op>(analyzer, Downcast<IntervalSet, IntSet>(a),
                       Downcast<IntervalSet, IntSet>(b));
  } else if (a.is_single_point() && b.is_single_point()) {
    return IntervalSet::SinglePoint(tir::BinaryOpNode<Op>::make(a.point_value(), b.point_value()));
  }
  DLOG(WARNING) << "Cannot combine int sets that aren't projection sets" << std::endl;
  return IntervalSet::Everything();
}

// internal helper function to get an interval set
IntervalSet ToIntervalSet(IntSet set) {
  if (auto* node = set.as<IntervalSetNode>()) {
    return GetRef<IntervalSet>(node);
  }
  DLOG(INFO) << "cannot resolve int set " << set;
  return IntervalSet::Everything();
}

using namespace tir;

// Simplified version of int set evaluator that operates on IntervalSet
// We might use better set analysis in the future to replace the intervalset.
class IntSetEvaluator : public ExprFunctor<IntSet(const PrimExpr&)> {
 public:
  IntSetEvaluator(Analyzer* analyzer, const Map<Var, IntSet>& dom_map, bool eval_vec = false)
      : analyzer_(analyzer), dom_map_(dom_map), eval_vec_(eval_vec) {}

  IntSet Eval(const PrimExpr& val) { return this->VisitExpr(val); }
  // evaluate and relax the set
  IntSet Eval(IntSet val) {
    // avoid recursive indefinite recursive expansion.
    if (static_cast<size_t>(recur_depth_) >= dom_map_.size()) return val;
    if (auto set = val.as<IntervalSetNode>()) {
      ++recur_depth_;
      IntSet min_set = this->Eval(set->min_value);
      IntSet max_set = this->Eval(set->max_value);
      --recur_depth_;
      return IntervalSet(min_set.min(), max_set.max());
    } else if (auto set = val.as<ProjectionSetNode>()) {
      ++recur_depth_;
      Map<te::Dimension, IntSet> arguments;
      for (auto pair : set->arguments) {
        arguments.Set(pair.first, this->Eval(pair.second));
      }
      --recur_depth_;
      auto res = ProjectionSet(set->ufun, arguments);
      if (res.is_single_point()) {
        return IntervalSet::SinglePoint(res.point_value());
      }
      return res;
    } else {
      DLOG(WARNING) << "No such set type\n";
      return IntervalSet::Everything();
    }
  }

  IntSet VisitExpr_(const IntImmNode* op) final {
    return IntervalSet::SinglePoint(GetRef<PrimExpr>(op));
  }

  IntSet VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    auto it = dom_map_.find(var);
    if (it != dom_map_.end()) {
      IntervalSet res = ToIntervalSet((*it).second);
      if (res->min_value.same_as(var) && res->max_value.same_as(var)) {
        // std::cout << "[ISE]    Var val1 " << var << " " << res << std::endl;
        return res;
      }
      // recursively evaluate mapped result
      // in case the domain contains variables to be relaxed.
      auto set = Eval(res);
      // std::cout << "[ISE]    Var val2 " << var << " " << (*it).second << " " << set << std::endl;
      return set;
    } else {
      auto set = IntervalSet::SinglePoint(var);
      // std::cout << "[ISE]    Var val3 " << var << " " << set << std::endl;
      return set;
    }
  }

  IntSet VisitExpr_(const CallNode* op) final {
    auto func = op->func;
    if (auto func_node = func.as<UninterpFunNode>()) {
      return IntervalSet::SinglePoint(GetRef<PrimExpr>(op));
      // // if (func_node->is_complex()) {
      // if (true) {
      //   CHECK_EQ(op->argument_dimensions.size(), op->args.size());
      //   UninterpFun ufun = Downcast<UninterpFun, FunctionRef>(func);
      //   Map<te::Dimension, IntSet> arg_sets;
      //   for (size_t i = 0; i < op->args.size(); ++i) {
      //     if (ufun->dimensions.Contains(op->argument_dimensions[i])) {
      //       auto set = this->Eval(op->args[i]);
      //       std::cout << "[ISE]    Arg set " << op->args[i] << " " << op->argument_dimensions[i]
      //                 << " " << set << std::endl;
      //       arg_sets.Set(op->argument_dimensions[i], set);
      //     }
      //   }
      //   std::cout << "[ISE]     Evaling projset " << GetRef<PrimExpr>(op) << std::endl;
      //   return ProjectionSet(ufun, arg_sets);
      // } else {
      //   auto set = this->Eval(func_node->substitute(op->args, op->argument_dimensions));
      //   std::cout << "[ISE]     Evaling set " << GetRef<PrimExpr>(op) << " " << set << std::endl;
      //   return set;
      // }
    } else {
      DLOG(WARNING) << "cannot evaluate expression " << GetRef<PrimExpr>(op);
      std::cout << "[ISE]     Evaling everything " << GetRef<PrimExpr>(op) << std::endl;
      return IntervalSet::Everything();
      // return this->Eval(func_node->substitute(call->args));
    }
  }

  IntSet VisitExpr_(const AddNode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const SubNode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const MulNode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const DivNode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const ModNode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const FloorDivNode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const FloorModNode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const MinNode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const MaxNode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const EQNode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const NENode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const LTNode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const LENode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const GTNode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const GENode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const AndNode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const OrNode* op) final { return VisitBinaryExpr_(op); }

  IntSet VisitExpr_(const RampNode* op) final {
    CHECK(eval_vec_);
    IntSet base_int_set = Eval(op->base);
    if (base_int_set.as<IntervalSetNode>()) {
      IntervalSet base = Downcast<IntervalSet, IntSet>(base_int_set);
      PVar<IntImm> stride;
      if (stride.Match(op->stride)) {
        DataType t = op->base.dtype();
        int64_t vstride = stride.Eval()->value;
        if (vstride > 0) {
          return Combine<AddNode>(
              analyzer_, base, IntervalSet(make_zero(t), make_const(t, vstride * op->lanes - 1)));
        } else {
          return Combine<AddNode>(
              analyzer_, base, IntervalSet(make_const(t, vstride * op->lanes + 1), make_zero(t)));
        }
      }
    }
    DLOG(WARNING) << "cannot evaluate set on expression " << GetRef<PrimExpr>(op);
    return IntervalSet::Everything();
  }

  IntSet VisitExpr_(const BroadcastNode* op) final {
    CHECK(eval_vec_);
    return VisitExpr(op->value);
  }

  IntSet VisitExpr_(const SelectNode* op) final {
    IntSet true_set = this->Eval(op->true_value);
    IntSet false_set = this->Eval(op->false_value);
    return Union(analyzer_, false_set, true_set);
  }

  IntSet VisitExprDefault_(const Object* op) final {
    DLOG(WARNING) << "cannot evaluate set type " << op->GetTypeKey();
    return IntervalSet::Everything();
  }

 private:
  // whether set is exactly single point that equals value.
  bool MatchPoint(const IntSet& set, const PrimExpr& value) const {
    return set.min().same_as(value) && set.max().same_as(value);
  }

  template <typename T>
  inline IntSet VisitBinaryExpr_(const T* op) {
    IntSet a = this->Eval(op->a);
    IntSet b = this->Eval(op->b);
    if (MatchPoint(a, op->a) && MatchPoint(b, op->b)) {
      return IntervalSet::SinglePoint(GetRef<PrimExpr>(op));
    }
    return CombineIntSets<T>(analyzer_, a, b);
  }

  // recursive depth
  int recur_depth_{0};
  // analyzer
  Analyzer* analyzer_;
  const Map<Var, IntSet>& dom_map_;
  bool eval_vec_{false};
};

class IntSetAnalyzer::Impl {
 public:
  explicit Impl(Analyzer* analyzer) : analyzer_(analyzer) {}

  IntSet Eval(const PrimExpr& expr, const Map<Var, IntSet>& dom_map) const {
    return IntSetEvaluator(analyzer_, dom_map).Eval(expr);
  }

 private:
  Analyzer* analyzer_;
};

IntSetAnalyzer::IntSetAnalyzer(Analyzer* parent) : impl_(new Impl(parent)) {}

IntSetAnalyzer::~IntSetAnalyzer() { delete impl_; }

IntSet IntSetAnalyzer::operator()(const PrimExpr& expr, const Map<Var, IntSet>& dom_map) {
  return impl_->Eval(expr, dom_map);
}

// Quickly adapt to IntSet interface
// TODO(tqchen): revisit IntSet interface as well.
Range IntSet::cover_range(Range max_range) const {
  IntSet temp;
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();

  if (s_int) {
    CHECK(s_int != nullptr);
    if (s_int->HasUpperBound() && s_int->HasLowerBound()) {
      return Range::make_by_min_extent(s_int->min_value,
                                       Simplify(s_int->max_value + 1 - s_int->min_value));
    }
  }
  return max_range;
}

PrimExpr IntSet::min() const {
  if (const IntervalSetNode* s_int = (*this).as<IntervalSetNode>()) {
    CHECK(s_int);
    return s_int->min_value;
  } else if (auto s_proj = (*this).as<ProjectionSetNode>()) {
    if (this->is_single_point()) {
      return this->point_value();
    } else {
      return s_proj->ufun->range->min;
    }
  } else {
    return SymbolicLimits::neg_inf_;
  }
}

PrimExpr IntSet::max() const {
  if (const IntervalSetNode* s_int = (*this).as<IntervalSetNode>()) {
    CHECK(s_int);
    return s_int->max_value;
  } else if (auto s_proj = (*this).as<ProjectionSetNode>()) {
    if (this->is_single_point()) {
      return this->point_value();
    } else {
      return s_proj->ufun->range->min + s_proj->ufun->range->extent - 1;
    }
  } else {
    return SymbolicLimits::pos_inf_;
  }
}

bool IntSet::is_nothing() const {
  if (const IntervalSetNode* s_int = (*this).as<IntervalSetNode>()) {
    return (s_int && s_int->IsEmpty());
  } else if (auto s_proj = (*this).as<ProjectionSetNode>()) {
    for (auto arg_set : s_proj->arguments) {
      if (!arg_set.second.is_nothing()) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

bool IntSet::is_everything() const {
  if (const IntervalSetNode* s_int = (*this).as<IntervalSetNode>()) {
    return (s_int && s_int->IsEverything());
  } else {
    return false;
  }
}

bool IntSet::is_single_point() const {
  if (const IntervalSetNode* s_int = (*this).as<IntervalSetNode>()) {
    return (s_int && s_int->IsSinglePoint());
  } else if (const ProjectionSetNode* s_proj = (*this).as<ProjectionSetNode>()) {
    for (auto s : s_proj->arguments) {
      if (!s.second.is_single_point()) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

bool IntSet::can_prove_positive() const {
  if (const IntervalSetNode* s_int = (*this).as<IntervalSetNode>()) {
    return (s_int && is_positive_const(tir::Simplify(s_int->min_value)));
  } else if (const ProjectionSetNode* s_proj = (*this).as<ProjectionSetNode>()) {
    return is_positive_const(tir::Simplify(s_proj->ufun->range->min)) &&
           is_positive_const(
               tir::Simplify(s_proj->ufun->range->min + s_proj->ufun->range->extent - 1));
  } else {
    return false;
  }
}

bool IntSet::can_prove_negative() const {
  if (const IntervalSetNode* s_int = (*this).as<IntervalSetNode>()) {
    return (s_int && is_negative_const(tir::Simplify(s_int->max_value)));
  } else if (const ProjectionSetNode* s_proj = (*this).as<ProjectionSetNode>()) {
    return is_negative_const(tir::Simplify(s_proj->ufun->range->min)) &&
           is_negative_const(
               tir::Simplify(s_proj->ufun->range->min + s_proj->ufun->range->extent - 1));
  } else {
    return false;
  }
}

bool IntSet::can_prove_non_positive() const {
  if (const auto* s_int = (*this).as<IntervalSetNode>()) {
    auto max = tir::Simplify(s_int->max_value);
    return is_zero(max) || is_negative_const(max);
  } else if (const ProjectionSetNode* s_proj = (*this).as<ProjectionSetNode>()) {
    auto start = tir::Simplify(s_proj->ufun->range->min);
    auto end = tir::Simplify(s_proj->ufun->range->min + s_proj->ufun->range->extent - 1);
    return (is_negative_const(start) || is_zero(start)) && (is_negative_const(end) || is_zero(end));
  } else {
    return false;
  }
}

bool IntSet::can_prove_non_negative() const {
  if (const IntervalSetNode* s_int = (*this).as<IntervalSetNode>()) {
    auto min = tir::Simplify(s_int->min_value);
    return is_zero(min) || is_positive_const(min);
  } else if (const ProjectionSetNode* s_proj = (*this).as<ProjectionSetNode>()) {
    auto start = tir::Simplify(s_proj->ufun->range->min);
    auto end = tir::Simplify(s_proj->ufun->range->min + s_proj->ufun->range->extent - 1);
    return (is_positive_const(start) || is_zero(start)) && (is_positive_const(end) || is_zero(end));
  } else {
    return false;
  }
}

SignType IntSet::sign_type() const {
  if (can_prove_positive()) {
    return kPositive;
  } else if (can_prove_negative()) {
    return kNegative;
  } else if (is_single_point() && is_zero(point_value())) {
    return kZero;
  } else {
    return kUnknown;
  }
}

PrimExpr IntSet::point_value() const {
  if (const IntervalSetNode* s_int = (*this).as<IntervalSetNode>()) {
    CHECK(s_int && s_int->IsSinglePoint());
    return s_int->min_value;
  } else if (const ProjectionSetNode* s_proj = (*this).as<ProjectionSetNode>()) {
    CHECK(this->is_single_point());
    Array<PrimExpr> args;
    Array<Dimension> arg_dims;
    for (auto it : s_proj->arguments) {
      args.push_back(it.second.point_value());
      arg_dims.push_back(it.first);
    }
    // return s_proj->ufun->substitute(args, arg_dims);
    return UninterpFun::MakeCallTo(s_proj->ufun, args, arg_dims);
  } else {
    return SymbolicLimits::neg_inf_;
  }
}

IntSet IntSet::nothing() { return IntervalSet::Empty(); }

IntSet IntSet::everything() { return IntervalSet::Everything(); }

IntSet IntSet::single_point(PrimExpr x) { return IntervalSet::SinglePoint(x); }

IntSet IntSet::interval(PrimExpr min, PrimExpr max) {
  if (min.same_as(max)) {
    return IntSet::single_point(min);
  }
  return IntervalSet(min, max);
}

// Range related code
inline bool ProveEqual(PrimExpr lhs, PrimExpr rhs) { return is_zero(tir::Simplify(lhs - rhs)); }

IntSet IntSet::range(Range r) {
  // must make sure it can be matched back by MatchRange.
  if (is_one(r->extent)) {
    return IntSet::single_point(r->min);
  }
  return IntervalSet(r->min, r->extent + r->min - 1);
}

bool IntSet::match_range(const Range& b) const {
  const IntSet& a = *this;
  const IntervalSetNode* a_int = a.as<IntervalSetNode>();
  if (!a_int) return false;
  return ProveEqual(a_int->min_value, b->min) &&
         ProveEqual(a_int->max_value, b->extent + b->min - 1);
}

IntSet Union(const Array<IntSet>& sets) {
  if (sets.size() == 0) return IntSet::nothing();
  if (sets.size() == 1) return sets[0];

  Analyzer ana;
  IntSet current = sets[0];
  for (size_t i = 1; i < sets.size(); ++i) {
    current = Union(&ana, current, sets[i]);
  }
  // IntervalSet x = ToIntervalSet(sets[0]);
  // for (size_t i = 1; i < sets.size(); ++i) {
  //   x = Union(&ana, x, ToIntervalSet(sets[i]));
  // }
  // return IntervalSet(tir::Simplify(x->min_value), tir::Simplify(x->max_value));
  return current;
}

IntSet Intersect(const Array<IntSet>& sets) {
  if (sets.size() == 0) return IntSet::nothing();
  if (sets.size() == 1) return sets[0];
  Analyzer ana;
  IntervalSet x = ToIntervalSet(sets[0]);
  for (size_t i = 1; i < sets.size(); ++i) {
    x = Intersect(&ana, x, ToIntervalSet(sets[i]));
  }
  return IntervalSet(tir::Simplify(x->min_value), tir::Simplify(x->max_value));
}

Map<Var, IntSet> ConvertDomMap(const Map<IterVar, IntSet>& dom_map) {
  Map<Var, IntSet> dmap;
  for (auto kv : dom_map) {
    dmap.Set(kv.first->var, kv.second);
  }
  return dmap;
}

Map<Var, IntSet> ConvertDomMap(const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  Map<Var, IntSet> dmap;
  for (auto kv : dom_map) {
    dmap.Set(GetRef<Var>(kv.first), kv.second);
  }
  return dmap;
}

IntSet EvalSet(PrimExpr e, const Map<Var, IntSet>& dom_map) {
  Analyzer ana;
  return IntSetEvaluator(&ana, dom_map, false).Eval(e);
}

IntSet IntSet::vector(PrimExpr x) {
  Analyzer ana;
  Map<Var, IntSet> dmap;
  return IntSetEvaluator(&ana, dmap, true).Eval(x);
}

IntSet EvalSet(PrimExpr e, const Map<IterVar, IntSet>& dom_map) {
  return EvalSet(e, ConvertDomMap(dom_map));
}

IntSet EvalSet(PrimExpr e, const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  return EvalSet(e, ConvertDomMap(dom_map));
}

IntSet EvalSet(Range r, const Map<Var, IntSet>& dom_map) {
  Analyzer ana;
  IntSetEvaluator m(&ana, dom_map);
  // Simplifying first can give tighter bounds if r->min and r->extent share variables
  PrimExpr sum = r->min + r->extent - 1;
  auto res = m.Eval(IntervalSet(r->min, Simplify(sum)));
  // return std::move(res);
  return res;
}

IntSet EvalSet(Range r, const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  return EvalSet(r, ConvertDomMap(dom_map));
}

IntSet EvalSet(IntSet s, const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  Analyzer ana;
  auto dmap = ConvertDomMap(dom_map);
  IntSetEvaluator m(&ana, dmap);
  const IntervalSetNode* s_int = s.as<IntervalSetNode>();
  PrimExpr vmax = s_int->HasUpperBound() ? m.Eval(s_int->max_value).max() : s_int->max_value;
  PrimExpr vmin = s_int->HasLowerBound() ? m.Eval(s_int->min_value).min() : s_int->min_value;
  return IntervalSet(vmin, vmax);
}

class SubExprIntSetEvaluator : public IntSetEvaluator {
 public:
  explicit SubExprIntSetEvaluator(Analyzer* analyzer, const Map<Var, IntSet>& dom_map)
      : IntSetEvaluator(analyzer, dom_map) {}

  IntSet VisitExpr(const PrimExpr& n) final {
    IntSet ret = IntSetEvaluator::VisitExpr(n);
    expr_map[n] = ret;
    return ret;
  }

  ExprIntSetMap expr_map;
};

ExprIntSetMap EvalSetForEachSubExpr(PrimExpr e,
                                    const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  Analyzer ana;
  auto dmap = ConvertDomMap(dom_map);
  SubExprIntSetEvaluator m(&ana, dmap);
  m.Eval(e);
  return m.expr_map;
}

IntSet EvalSet(Range r, const Map<IterVar, IntSet>& dom_map) {
  return EvalSet(r, ConvertDomMap(dom_map));
}

TVM_REGISTER_NODE_TYPE(IntervalSetNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IntervalSetNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IntervalSetNode*>(node.get());
      p->stream << "IntervalSet"
                << "[" << op->min_value << ", " << op->max_value << ']';
    });

TVM_REGISTER_NODE_TYPE(ProjectionSetNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ProjectionSetNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ProjectionSetNode*>(node.get());
      p->stream << "ProjectionSet"
                << "[" << op->ufun->body << "(";
      for (auto dim : op->ufun->dimensions) {
        p->stream << dim->name << ": " << op->arguments.at(dim);
      }
      p->stream << ")]";
    });

TVM_REGISTER_GLOBAL("arith.intset_single_point").set_body_typed(IntSet::single_point);

TVM_REGISTER_GLOBAL("arith.intset_vector").set_body_typed(IntSet::vector);

TVM_REGISTER_GLOBAL("arith.intset_interval").set_body_typed(IntSet::interval);

TVM_REGISTER_GLOBAL("arith.IntervalSetGetMin").set_body_method(&IntSet::min);

TVM_REGISTER_GLOBAL("arith.IntervalSetGetMax").set_body_method(&IntSet::max);

TVM_REGISTER_GLOBAL("arith.IntSetIsNothing").set_body_method(&IntSet::is_nothing);

TVM_REGISTER_GLOBAL("arith.IntSetIsEverything").set_body_method(&IntSet::is_everything);

}  // namespace arith
}  // namespace tvm
