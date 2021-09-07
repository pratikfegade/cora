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
  ThreadVarRewriter(Map<std::string, FunctionRef> vsub_) : vsub(vsub_) {}

  PrimExpr VisitExpr_(const VarNode* op) override {
    if (vsub.count(op->name_hint)) {
      auto function = Downcast<UninterpFun>(vsub.at(op->name_hint));
      return UninterpFun::InlineUninterpFunCalls(
          function.MakeCallTo({GetRef<PrimExpr>(op)}, function->dimensions));
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

 private:
  Map<std::string, FunctionRef> vsub;
};

Stmt SubstituteThreadVars(Stmt stmt, Map<std::string, FunctionRef> vsub_map) {
  stmt = ThreadVarRewriter(vsub_map)(std::move(stmt));
  return stmt;
}

}  // namespace tir
}  // namespace tvm
