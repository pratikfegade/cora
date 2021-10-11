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
 * \file inject_virtual_thread.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_set>

#include "../../arith/compute_expr.h"

namespace tvm {
namespace tir {

class ThreadVarRewriter : public StmtExprMutator {
 public:
  ThreadVarRewriter(Map<std::string, FunctionRef> vsub_, Array<FunctionRef> to_substitute_in_, bool substitute_) :
    vsub(vsub_), to_substitute_in(to_substitute_in_), substitute(substitute_) {}

  Stmt VisitStmt_(const ProducerConsumerNode* op) override {
    // std::cout << "[STV] Visiting " << op->func << std::endl;
    if (to_substitute_in.Contains(op->func)) {
      bool old_substitute = substitute;
      // std::cout << "[STV]  +In" << std::endl;
      substitute = true;
      Stmt body = this->VisitStmt(op->body);
      substitute = old_substitute;
      // std::cout << "[STV]  After " << op->func << " " << substitute << std::endl;
      return ProducerConsumerNode::make(op->func, op->is_producer, body);
    } else {
      bool old_substitute = substitute;
      // std::cout << "[STV]  -In" << std::endl;
      substitute = false;
      Stmt body = this->VisitStmt(op->body);
      substitute = old_substitute;
      // std::cout << "[STV]  After " << op->func << " " << substitute << std::endl;
      return ProducerConsumerNode::make(op->func, op->is_producer, body);
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) override {
    // if (op->name_hint == "blockIdx.y")
      // std::cout << "[STV] Var " << substitute << std::endl;
    if (substitute && vsub.count(op->name_hint)) {
      auto function = Downcast<UninterpFun>(vsub.at(op->name_hint));
      return UninterpFun::InlineUninterpFunCalls(
          function.MakeCallTo({GetRef<PrimExpr>(op)}, function->dimensions));
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

 private:
  Map<std::string, FunctionRef> vsub;
  Array<FunctionRef> to_substitute_in;
  bool substitute;
};

Stmt SubstituteThreadVars(Stmt stmt, Array<FunctionRef> to_substitute_in, Map<std::string, FunctionRef> vsub_map) {
  for (auto op: to_substitute_in) {
    // std::cout << "[STV] " << op << std::endl;
  }
  stmt = ThreadVarRewriter(vsub_map, to_substitute_in, false)(std::move(stmt));
  return stmt;
}

LoweredFunc SubstituteThreadVarsFunc(LoweredFunc func, Array<FunctionRef> to_substitute_in,
				     Map<std::string, FunctionRef> vsub_map) {
  auto n = make_object<LoweredFuncNode>(*func.operator->());
  n->body = ThreadVarRewriter(vsub_map, to_substitute_in, true)(func->body);
  return LoweredFunc(n);
}

}  // namespace tir
}  // namespace tvm
