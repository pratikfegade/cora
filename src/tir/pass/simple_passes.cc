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
 * \file simple_passes.cc
 * \brief Implementation of simple passes
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

class IRSideEffect : public ExprVisitor {
 public:
  void VisitExpr(const PrimExpr& e) final {
    if (has_side_effect_) return;
    ExprVisitor::VisitExpr(e);
  }

  void VisitExpr_(const CallNode* op) final {
    if (!op->is_pure()) {
      has_side_effect_ = true;
      return;
    } else {
      ExprVisitor::VisitExpr_(op);
    }
  }

  bool has_side_effect_{false};
};

bool HasSideEffect(const PrimExpr& e) {
  IRSideEffect v;
  v(e);
  return v.has_side_effect_;
}

class IRSubstitute : public StmtExprMutator {
 public:
  explicit IRSubstitute(const std::unordered_map<const VarNode*, PrimExpr>& smap) : smap_(smap) {
    auto it = smap.begin();
    while (it != smap.end()) {
      if (it->first == it->second.get()) {
        it = const_cast<std::unordered_map<const VarNode*, PrimExpr>&>(smap).erase(it);
      } else {
        ++it;
      }
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = smap_.find(op);
    if (it != smap_.end()) {
      // std::cout << "[REPL] " << op->name_hint << " " << it->second << std::endl;
      return it->second;
    } else {
      // std::cout << "[SAME] " << op->name_hint << std::endl;
      return GetRef<PrimExpr>(op);
    }
  }

 private:
  const std::unordered_map<const VarNode*, PrimExpr>& smap_;
};

Stmt Substitute(Stmt stmt, const std::unordered_map<const VarNode*, PrimExpr>& value_map) {
  if (value_map.size() == 0) return stmt;
  IRSubstitute substitute(value_map);
  // for (int i = 0; i < 10; ++i) {
  stmt = substitute(stmt);
  // }
  return std::move(stmt);
}

PrimExpr Substitute(PrimExpr expr, const std::unordered_map<const VarNode*, PrimExpr>& value_map) {
  if (value_map.size() == 0) return expr;
  // return IRSubstitute(value_map)(std::move(expr));

  IRSubstitute substitute(value_map);
  // for (int i = 0; i < 10; ++i) {
  expr = substitute(expr);
  // }
  return std::move(expr);
}

Stmt Substitute(Stmt stmt, const Map<Var, PrimExpr>& value_map) {
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  for (const auto& kv : value_map) {
    vmap[kv.first.get()] = kv.second;
  }
  return Substitute(stmt, vmap);
}

PrimExpr Substitute(PrimExpr expr, const Map<Var, PrimExpr>& value_map) {
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  for (const auto& kv : value_map) {
    vmap[kv.first.get()] = kv.second;
  }
  return Substitute(expr, vmap);
}

class VarTouchVisitor : public ExprVisitor {
 public:
  void VisitExpr(const PrimExpr& e) final {
    if (use_var_) return;
    ExprVisitor::VisitExpr(e);
  }

  void VisitExpr_(const VarNode* op) final { Handle(op); }

  void VisitExpr_(const LoadNode* op) final {
    Handle(op->buffer_var.get());
    ExprVisitor::VisitExpr_(op);
  }

  virtual void Handle(const VarNode* var) = 0;

  bool use_var_{false};
};

class ExprUseVarVisitor : public VarTouchVisitor {
 public:
  explicit ExprUseVarVisitor(const VarNode* var) : var_(var) {}

  void Handle(const VarNode* var) final {
    if (var == var_) use_var_ = true;
  }

 private:
  const VarNode* var_;
};

class ExprUseVSetVisitor : public VarTouchVisitor {
 public:
  explicit ExprUseVSetVisitor(const std::unordered_set<const VarNode*>& vset) : vset_(vset) {}

  void Handle(const VarNode* var) final {
    if (vset_.count(var)) use_var_ = true;
  }

 private:
  const std::unordered_set<const VarNode*>& vset_;
};

bool ExprUseVar(const PrimExpr& e, const Var& v) {
  ExprUseVarVisitor visitor(v.get());
  visitor(e);
  return visitor.use_var_;
}

bool ExprUseVar(const PrimExpr& e, const std::unordered_set<const VarNode*>& vset) {
  ExprUseVSetVisitor visitor(vset);
  visitor(e);
  return visitor.use_var_;
}

}  // namespace tir
}  // namespace tvm
