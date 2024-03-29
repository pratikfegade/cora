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
 *  Lower intrinsic calls and ops to device specific ir when possible.
 * \file lower_intrin.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/op.h>

#include <unordered_set>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../arith/pattern_match.h"
#include "ir_util.h"

namespace tvm {
namespace tir {

class IntrinInjecter : public tvm::arith::IRMutatorWithAnalyzer {
 public:
  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt_;

  IntrinInjecter(arith::Analyzer* analyzer, std::string target) : IRMutatorWithAnalyzer(analyzer) {
    std::istringstream is(target);
    std::string starget;
    is >> starget;
    patterns_.push_back("tvm.intrin.rule." + starget + ".");
    patterns_.push_back("tvm.intrin.rule.default.");
    fma_ = runtime::Registry::Get(patterns_[0] + "fma");
    if (target == "stackvm") {
      support_bitwise_op_ = false;
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->call_type == CallNode::Intrinsic || op->call_type == CallNode::PureIntrinsic) {
      PrimExpr r = ApplyPattern(op->name, GetRef<PrimExpr>(op));
      if (r.defined()) return r;
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const AddNode* op) final {
    if (const MulNode* mb = op->b.as<MulNode>()) {
      return MakeFMA(mb->a, mb->b, op->a, op);
    } else if (const MulNode* ma = op->a.as<MulNode>()) {
      return MakeFMA(ma->a, ma->b, op->b, op);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  // We use floordiv for integer analysis,
  // but will need to lower them to native truncdiv instructions
  PrimExpr VisitExpr_(const FloorDivNode* op) final {
    auto e = GetRef<PrimExpr>(op);
    PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
    op = ret.as<FloorDivNode>();
    if (op == nullptr) return ret;
    int shift;
    const DataType& dtype = op->dtype;
    CHECK(dtype.is_int() || dtype.is_uint());

    if (support_bitwise_op_ && is_const_power_of_two_integer(op->b, &shift)) {
      // lower to right shift if possible.
      return op->a >> make_const(dtype, shift);
    }

    // std::cout << "[FLRD] Trying to prove non-negative " << op->b << std::endl;
    // std::cout << "[FLRD]                              " << op->a << std::endl;
    // std::cout << "[FLRD]                              " << e << std::endl;
    if (analyzer_->CanProveGreaterEqual(op->b, 0)) {
      // Common path, positive divisor
      if (analyzer_->CanProveGreaterEqual(op->a, 0) || analyzer_->CanProveGreaterEqual(e, 0)) {
        return truncdiv(op->a, op->b);
      } else {
        DLOG(INFO) << "LowerFloorDiv: Cannot decide the sign of divident";
        PrimExpr rdiv = truncdiv(op->a, op->b);
        PrimExpr rmod = truncmod(op->a, op->b);
        // condition on b >= 0.
        // truncmod(a, b) < 0 will implies ceildiv,
        // So we need to correct these cases.
        if ((dtype == DataType::Int(32) || dtype == DataType::Int(64)) && support_bitwise_op_) {
          // equivalent to rdiv + (rmod >= 0 ? 0: -1);
          return rdiv + (rmod >> make_const(dtype, dtype.bits() - 1));
        } else {
          return tir::SelectNode::make(rmod >= 0, rdiv, rdiv - make_const(dtype, 1));
        }
      }
    } else {
      // uncommon case
      DLOG(INFO) << "LowerFloorDiv: Cannot decide the sign of divisor";
      // b >= 0 => (rmod >=0 ? rdiv : rdiv - 1)
      // b < 0  => (rmod <= 0 ? rdiv : rdiv - 1)
      PrimExpr rdiv = truncdiv(op->a, op->b);
      PrimExpr rmod = truncmod(op->a, op->b);
      return tir::SelectNode::make((op->b >= 0 && rmod >= 0) || (op->b < 0 && rmod <= 0), rdiv,
                                   rdiv - make_const(dtype, 1));
    }
  }

  PrimExpr VisitExpr_(const FloorModNode* op) final {
    PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
    op = ret.as<FloorModNode>();
    if (op == nullptr) return ret;
    // std::cout << "[FLRD] Visiting floormod " << ret << std::endl;
    // Lower floordiv to native truncdiv.
    int shift;
    const DataType& dtype = op->dtype;
    CHECK(dtype.is_int() || dtype.is_uint());

    if (support_bitwise_op_ && is_const_power_of_two_integer(op->b, &shift)) {
      // lower to masking if possible.
      int64_t mask = (static_cast<int64_t>(1) << static_cast<int64_t>(shift)) - 1;
      // std::cout << "[FLRD]   Return1" << std::endl;
      return op->a & make_const(dtype, mask);
    }

    // std::cout << "[FLRD] Trying to prove non-negative " << op->b << std::endl;
    // std::cout << "[FLRD]                              " << op->a << std::endl;
    if (analyzer_->CanProveGreaterEqual(op->b, 0)) {
      // Common pass, positive divisor
      if (analyzer_->CanProveGreaterEqual(op->a, 0)) {
        // std::cout << "[FLRD]   Return2" << std::endl;
        return truncmod(op->a, op->b);
      } else {
        DLOG(INFO) << "LowerFloorMod: Cannot decide the sign of divident";
        // NOTE:condition on b >= 0.
        // mod(a, b) < 0 will imply we are doing ceildiv,
        // So we need to correct these cases.
        PrimExpr rmod = truncmod(op->a, op->b);
        if ((dtype == DataType::Int(32) || dtype == DataType::Int(64)) && support_bitwise_op_) {
          // (rmod >> shift) & b
          // -> (rmod >= 0 ? 0: -1) & b
          // -> rmod >= 0 ? 0 : b
          // std::cout << "[FLRD]   Return3" << std::endl;
          return rmod + (op->b & (rmod >> make_const(dtype, dtype.bits() - 1)));
        } else {
          auto ret = tir::SelectNode::make(rmod >= 0, rmod, rmod + op->b);
          // std::cout << "[FLRD]   Return 4 " << ret << std::endl;
          return ret;
        }
      }
    } else {
      // uncommon case
      DLOG(INFO) << "LowerFloorMod: Cannot decide the sign of divsor and divident";
      PrimExpr rmod = truncmod(op->a, op->b);
      // b > 0 && rmod >= 0 -> rmod
      // b > 0 && rmod < 0  -> rmod + b
      // b < 0 && rmod < 0 -> rmod
      // b < 0 && rmod > 0 -> rmod + b
      // std::cout << "[FLRD]   Return5" << std::endl;
      return tir::SelectNode::make((op->b >= 0 && rmod >= 0) || (op->b < 0 && rmod <= 0), rmod,
                                   rmod + op->b);
    }
  }

  PrimExpr VisitExpr_(const MaxNode* op) final {
    using namespace arith;
    PVar<PrimExpr> x, y;
    PVar<IntImm> c;
    auto e = GetRef<PrimExpr>(op);
    if (max(floordiv(x, y), c).Match(e) && c.Eval()->value >= 0 &&
        analyzer_->CanProveGreaterEqual(y.Eval(), 0)) {
      return max(VisitExpr(truncdiv(x, y).Eval()), c.Eval());
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const EQNode* op) final {
    using namespace arith;
    PVar<PrimExpr> x, y;
    auto e = GetRef<PrimExpr>(op);
    if ((floormod(x, y) == 0).Match(e)) {
      return VisitExpr((truncmod(x, y) == 0).Eval());
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const NENode* op) final {
    using namespace arith;
    PVar<PrimExpr> x, y;
    auto e = GetRef<PrimExpr>(op);
    if ((floormod(x, y) != 0).Match(e)) {
      return VisitExpr((truncmod(x, y) != 0).Eval());
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::aux_data_structure) {
      bool constraint_added = false;
      if (auto ufn = op->node.as<UninterpFunNode>()) {
        analyzer_->AddForallConstraintScoped(ufn->parameters, ufn->body >= 0);
        constraint_added = true;
      } else if (op->node.as<VarNode>()) {
        Var index = Var("idx", DataType::Int(32));
        analyzer_->AddForallConstraintScoped(
            {index},
            LoadNode::make(DataType::Int(32), Downcast<Var>(op->node), index, 1, kAll) >= 0);
        constraint_added = true;
      }

      Stmt ret = IRMutatorWithAnalyzer::VisitStmt_(op);
      if (constraint_added) {
        analyzer_->RemoveLastConstraintScoped();
      }
      return ret;
    } else {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }
  }

 private:
  PrimExpr SwapBroadcastCast(const PrimExpr& e) {
    // Try to change broadcast(cast(x)) to cast(broadcast(x))
    // For some targets, LLVM will generate more efficient FMA
    // instruction with the latter. For example, vmla vs. vmlal
    // on ARM.
    if (const BroadcastNode* bcast = e.as<BroadcastNode>()) {
      if (const CastNode* cast = bcast->value.as<CastNode>()) {
        auto should_swap = [&]() {
          // Maintain behaviour (int8 -> int16, fp16 -> fp32).
          if (cast->dtype.bits() == cast->value.dtype().bits() * 2) {
            return true;
          }
          // Check both operands are integer-like.
          if (!cast->dtype.is_uint() && !cast->dtype.is_int()) {
            return false;
          }
          if (!cast->value.dtype().is_uint() && !cast->value.dtype().is_int()) {
            return false;
          }
          // If both are integer-like, swap if we have a widening cast.
          return cast->dtype.bits() > cast->value.dtype().bits();
        };

        if (should_swap()) {
          PrimExpr new_bcast = BroadcastNode::make(cast->value, bcast->lanes);
          return CastNode::make(bcast->dtype, new_bcast);
        }
      }
    }
    return e;
  }

  PrimExpr MakeFMA(const PrimExpr& a, const PrimExpr& b, const PrimExpr& c, const AddNode* op) {
    // emit fma instruction: a * b + c
    PrimExpr lhs = SwapBroadcastCast(a);
    PrimExpr rhs = SwapBroadcastCast(b);

    if (fma_ != nullptr && op->dtype.is_float()) {
      PrimExpr r =
          (*fma_)(CallNode::make(op->dtype, "fma", {lhs, rhs, c}, CallNode::PureIntrinsic));
      if (r.defined()) return this->VisitExpr(r);
    } else {
      if (!lhs.same_as(a) || !rhs.same_as(b)) {
        PrimExpr mul = this->VisitExpr(MulNode::make(lhs, rhs));
        return AddNode::make(mul, this->VisitExpr(c));
      }
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  PrimExpr ApplyPattern(const std::string& name, const PrimExpr& e) {
    bool print = false;//name == "exp";
    if (print) std::cout << "[LI] Patterning " << name << " " << e << std::endl;
    for (size_t i = 0; i < patterns_.size(); ++i) {
      std::string& p = patterns_[i];
      size_t psize = p.length();
      p.resize(psize + name.length());
      name.copy(&p[0] + psize, name.length());
      const runtime::PackedFunc* f = runtime::Registry::Get(p);
      std::string name = p;
      p.resize(psize);
      // if pattern exists.
      if (f != nullptr) {
	if (print) std::cout << "[LI]  PatternFunc exists " << p << std::endl;
        PrimExpr r = (*f)(e);
        CHECK(r.defined()) << "intrinsic rule must always return valid Expr " << r << " " << name;
        if (!r.same_as(e)) {
          return this->VisitExpr(r);
        }
      }
    }
    return PrimExpr();
  }

  // patterns
  std::vector<std::string> patterns_;
  const PackedFunc* fma_{nullptr};
  bool support_bitwise_op_{true};
};

Stmt LowerIntrinStmt(Stmt stmt, const std::string& target) {
  // std::cout << "[LI] Lowering intrinsics" << std::endl;
  arith::Analyzer analyzer;
  return IntrinInjecter(&analyzer, target)(std::move(stmt));
}

LoweredFunc LowerIntrin(LoweredFunc f, const std::string& target) {
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  n->body = LowerIntrinStmt(n->body, target);
  return LoweredFunc(n);
}

// Register the api only for test purposes
TVM_REGISTER_GLOBAL("ir_pass._LowerIntrinStmt").set_body_typed(LowerIntrinStmt);

}  // namespace tir
}  // namespace tvm
