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
 * \file inline.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

// inliner to inline a function
// the result may not be SSA,
// ConvertSSA need to be applied after this pass
class IRInline final : public StmtExprMutator {
 public:
  IRInline(FunctionRef f, Array<Var> args, PrimExpr body, Map<Var, PrimExpr> const_vmap)
      : f_(f), args_(args), body_(body), const_vmap_(const_vmap) {}

  PrimExpr VisitExpr_(const CallNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();

    if (op->func == f_) {
      CHECK_EQ(op->value_index, 0);
      expr = body_;
      if (args_.size() != op->args.size()) {
        std::cout << std::endl;
      }
      CHECK_EQ(args_.size(), op->args.size()) << GetRef<PrimExpr>(op);

      bool has_side_effect = false;
      for (size_t i = 0; i < op->args.size(); ++i) {
        if (HasSideEffect(op->args[i])) has_side_effect = true;
      }
      if (has_side_effect) {
        for (size_t i = 0; i < args_.size(); ++i) {
          expr = LetNode::make(args_[i], op->args[i], expr);
        }
      } else {
        Map<Var, PrimExpr> vmap;
        for (const auto& it : const_vmap_) {
          vmap.Set(it.first, it.second);
        }
        for (size_t i = 0; i < args_.size(); ++i) {
          vmap.Set(args_[i], op->args[i]);
        }
        expr = Substitute(EvaluateNode::make(expr), vmap).as<EvaluateNode>()->value;
      }
      return expr;
    } else {
      return expr;
    }

    if (op->name == tvm::tir::intrinsic::tvm_if_then_else) {
      PrimExpr condition = this->VisitExpr(op->args[0]);
      if (ana.CanProve(condition == 1))
        return this->VisitExpr(op->args[1]);
      else if (ana.CanProve(condition == 0))
        return this->VisitExpr(op->args[2]);
      else
        return ExprMutator::VisitExpr_(op);
    }
    return expr;
  }

 private:
  FunctionRef f_;
  Array<Var> args_;
  PrimExpr body_;
  Map<Var, PrimExpr> const_vmap_;
  arith::Analyzer ana;
};

class SimplifyInlined final : public StmtExprMutator {
 public:
  PrimExpr VisitExpr_(const CallNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);

    if (op->name == tvm::tir::intrinsic::tvm_if_then_else) {
      PrimExpr condition = this->VisitExpr(op->args[0]);
      if (ana.CanProve(condition == 1)) {
        return this->VisitExpr(op->args[1]);
      } else if (ana.CanProve(condition == 0)) {
        return this->VisitExpr(op->args[2]);
      }
    }
    return expr;
  }

 private:
  arith::Analyzer ana;
};

Stmt Inline(Stmt stmt, FunctionRef f, Array<Var> args, PrimExpr body, Map<Var, PrimExpr> vmap) {
  CHECK_EQ(f->num_outputs(), 1) << "can only inline output single value operation";
  Stmt ret = IRInline(f, args, body, vmap)(std::move(stmt));
  Stmt simplified = SimplifyInlined()(std::move(ret));
  if (simplified.same_as(stmt)) return simplified;
  return ConvertSSA(simplified);
}
}  // namespace tir
}  // namespace tvm
