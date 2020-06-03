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
 *  Loop peeling as in Halide pipeline.
 * \file peel_loop.cc
 */
// Peels the loop as in Halide pipeline.
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../arith/compute_expr.h"

namespace tvm {
namespace tir {

class LoopPeeler : public StmtExprMutator {
 public:
  explicit LoopPeeler() {}

  Stmt VisitStmt_(const ForNode* op) {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<ForNode>();

    if (op->for_type == ForType::Peeled) {
      return Peel(op);
    }
    return stmt;
  }

  Stmt Peel(const ForNode* op) {
    PrimExpr extent = GetExtent(op);

    // Create the body for the last iteration
    Map<Var, PrimExpr> vmap;
    vmap.Set(op->loop_var, op->min + extent - 1);
    Stmt last_iter = Substitute(op->body, vmap);

    // All iterations but the last one
    Stmt for_stmt = ForNode::make(op->loop_var, op->min, op->extent - 1, op->for_type,
                                  op->device_api, op->body);

    return SeqStmt({for_stmt, last_iter});
  }

 private:
  // returns the extent of the loop if it's a constant integer, otherwise return -1
  PrimExpr GetExtent(const ForNode* op) {
    // constant folding.
    PrimExpr extent = tir::Simplify(op->extent);
    return extent;
  }
};

Stmt PeelLoop(Stmt stmt) {
  Stmt ret = LoopPeeler()(stmt);
  if (!ret.same_as(stmt)) {
    return ConvertSSA(ret);
  } else {
    return ret;
  }
}

LoweredFunc PeelLoop(LoweredFunc f) {
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  Stmt body = f->body;
  body = PeelLoop(body);
  n->body = body;
  return LoweredFunc(n);
}

Stmt PeelLoopExplicitly(Stmt stmt) {
  const ForNode* op = stmt.as<ForNode>();
  if (!op) {
    LOG(FATAL) << "attempted to peel a non-loop statement";
  }
  return LoopPeeler().Peel(op);
}

}  // namespace tir
}  // namespace tvm
