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
 * \file stmt_simplify.cc
 * \brief Statement simplifier based on analyzer
 */
#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/op.h>

#include "../tir/ir/var_replacer.h"
#include "ir_mutator_with_analyzer.h"
#include "const_fold.h"

namespace tvm {
namespace arith {

using namespace tir;

class VarExtentCollector : public StmtVisitor {
 public:
  void VisitStmt_(const ForNode* op) final {
    HandleExtent(op->loop_var, op->min, op->extent);
    StmtVisitor::VisitStmt(op->body);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      HandleExtent(Downcast<IterVar>(op->node)->var, 0, op->value);
    }
    StmtVisitor::VisitStmt(op->body);
  }

  void HandleExtent(Var v, PrimExpr min, PrimExpr extent) {
    CHECK(!range_map_.count(v.get())) << "Reused variable";
    range_map_[v.get()] = Range::make_by_min_extent(min, extent);
  }

  std::unordered_map<const Object*, Range> range_map_;
};

class StmtSimplifier : public IRMutatorWithAnalyzer {
 public:
  explicit StmtSimplifier(Analyzer* analyzer) : IRMutatorWithAnalyzer(analyzer) {}

  using Parent = IRMutatorWithAnalyzer;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  PrimExpr VisitExpr(const PrimExpr& expr) final {
    bool print = false;  //(expr.as<FloorDivNode>());
    if (print) std::cout << "[SIMPL] Expr " << expr << std::endl;
    std::unordered_map<const VarNode*, arith::IntSet> relaxable;
    for (auto var : VarCollector().collect(expr)) {
      if (print)
        std::cout << "[SIMPL]  Var " << var->name_hint << " " << def_count_.count(var) << " "
                  << extent_collector_.range_map_.count(var) << std::endl;
      if (!def_count_.count(var) && extent_collector_.range_map_.count(var)) {
        Range r = extent_collector_.range_map_.at(var);
        if (analyzer_->CanProveGreaterEqual(r->min, 0)) {
          relaxable[var] = arith::IntSet::range(r);
          if (print) std::cout << "[SIMPL]   Relaxable " << var->name_hint << " " << r << std::endl;
        }
      }
    }

    if (relaxable.size() > 0) {
      IntSet set = EvalSet(expr, relaxable);
      PrimExpr min_expr = analyzer_->Simplify(set.min());
      PrimExpr max_expr = analyzer_->Simplify(set.max());
      PrimExpr res_expr = analyzer_->Simplify(max_expr - min_expr);
      if (print) {
        std::cout << "[SIMPL]     Expr " << tir::Simplify(expr) << std::endl;
        std::cout << "[SIMPL]      Min " << min_expr << std::endl;
        std::cout << "[SIMPL]      Max " << max_expr << std::endl;
        std::cout << "[SIMPL]      Res " << res_expr << std::endl;
      }
      if (!(is_pos_inf(min_expr) || is_neg_inf(min_expr) || is_pos_inf(min_expr) || is_neg_inf(min_expr)) &&
	  analyzer_->CanProve(res_expr == 0)) {
        return min_expr;
      }
    }
    return analyzer_->Simplify(expr);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      CHECK_NE(iv->thread_tag.length(), 0U);
      if (!def_count_.count(iv->var.get())) {
        this->HandleDef(iv->var.get());
      }
    }
    return Parent::VisitStmt_(op);
  }

  Stmt Simplify(Stmt stmt) {
    // std::cout << "[SIMPL] FUCKFUCKFUCKFUCKFUCK" << std::endl;
    VarExtentCollector collector;
    extent_collector_(stmt);
    return operator()(std::move(stmt));
  }

  Stmt VisitStmt_(const ForNode* op) final {
    this->HandleDef(op->loop_var.get());
    analyzer_->Bind(op->loop_var, Range::make_by_min_extent(op->min, op->extent));
    With<ConstraintContext> ctx1(analyzer_, op->loop_var >= op->min);
    With<ConstraintContext> ctx2(analyzer_, op->loop_var < op->min + op->extent);
    return Parent::VisitStmt_(op);
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    this->HandleDef(op->var.get());
    PrimExpr value = this->VisitExpr(op->value);
    if (!tir::HasSideEffect(value)) {
      // it is fine to discard the let binding
      // because the call to simplify will always inline the var.
      analyzer_->Bind(op->var, value);
      return this->VisitStmt(op->body);
    }
    Stmt body = this->VisitStmt(op->body);
    if (value.same_as(op->value) && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = this->CopyOnWrite(op);
      n->value = std::move(value);
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    this->HandleDef(op->buffer_var.get());
    return Parent::VisitStmt_(op);
  }

  // eliminate useless stores
  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = Parent::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    if (const LoadNode* load = op->value.as<LoadNode>()) {
      if (load->buffer_var.same_as(op->buffer_var) && Equal(load->index, op->index)) {
        return EvaluateNode::make(0);
      }
    }
    return GetRef<Stmt>(op);
  }

  void HandleDef(const VarNode* v) {
    CHECK(!def_count_.count(v)) << "variable " << v->name_hint
                                << " has already been defined, the Stmt is not SSA";
    // std::cout << "[SIMPL] Define " << v->name_hint << std::endl;
    def_count_[v] = 1;
  }

  std::unordered_map<const VarNode*, int> def_count_;
  VarExtentCollector extent_collector_;
};

}  // namespace arith

namespace tir {

Stmt CanonicalSimplify(Stmt stmt, Map<Var, Range> vrange) {
  arith::Analyzer analyzer;
  for (auto kv : vrange) {
    analyzer.Bind(kv.first, kv.second);
  }
  return arith::StmtSimplifier(&analyzer).Simplify(std::move(stmt));
}

PrimExpr CanonicalSimplify(PrimExpr expr, Map<Var, Range> vrange) {
  arith::Analyzer analyzer;
  for (auto kv : vrange) {
    analyzer.Bind(kv.first, kv.second);
  }
  return analyzer.canonical_simplify(expr);
}

PrimExpr Simplify(PrimExpr expr, Map<Var, Range> vrange) {
  arith::Analyzer analyzer;
  for (auto kv : vrange) {
    analyzer.Bind(kv.first, kv.second);
  }
  expr = analyzer.Simplify(expr);
  return expr;
}

Stmt Simplify(Stmt stmt, Map<Var, Range> vrange) {
  return CanonicalSimplify(std::move(stmt), vrange);
}
}  // namespace tir
}  // namespace tvm
