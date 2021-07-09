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
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

class LetInliner : public StmtExprMutator {
 public:
  Stmt VisitStmt_(const LetStmtNode* op) final {
    if (!HasSideEffect(op->value)) {
      var_value_[op->var.get()] = this->VisitExpr(op->value);
      Stmt body = this->VisitStmt(op->body);
      var_value_.erase(op->var.get());
      return body;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_value_.find(op);
    if (it != var_value_.end()) {
      return it->second;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

 private:
  std::unordered_map<const VarNode*, PrimExpr> var_value_;
};

Stmt InlineLets(Stmt stmt) { return LetInliner()(std::move(stmt)); }
}  // namespace tir
}  // namespace tvm
