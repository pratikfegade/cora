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
 * \file ir_util.cc
 * \brief Helper functions to construct and compose IR nodes.
 */
#include "ir_util.h"

#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

Map<Buffer, Buffer> ExtractPrepCode(const Stmt& full_body, Stmt* p_prep_code, Stmt* p_main_body) {
  class PrepCodeChecker : public StmtVisitor {
    void VisitStmt_(const LetStmtNode* op) final {
      scope_depth_++;
      this->VisitStmt(op->body);
      scope_depth_--;
    }
    void VisitStmt_(const AttrStmtNode* op) final {
      if (op->attr_key == attr::prep_code_scope) {
        CHECK(scope_depth_ == 0) << "Deeply nested prep code scope not allowed";
      }

      scope_depth_++;
      this->VisitStmt(op->body);
      scope_depth_--;
    }
    void VisitStmt_(const IfThenElseNode* op) final {
      scope_depth_++;
      this->VisitStmt(op->then_case);
      if (op->else_case.defined()) this->VisitStmt(op->else_case);
      scope_depth_--;
    }
    void VisitStmt_(const ForNode* op) final {
      scope_depth_++;
      this->VisitStmt(op->body);
      scope_depth_--;
    }
    void VisitStmt_(const AllocateNode* op) final {
      scope_depth_++;
      this->VisitStmt(op->body);
      scope_depth_--;
    }
    void VisitStmt_(const AssertStmtNode* op) final {
      scope_depth_++;
      this->VisitStmt(op->body);
      scope_depth_--;
    }
    void VisitStmt_(const ProducerConsumerNode* op) final {
      scope_depth_++;
      this->VisitStmt(op->body);
      scope_depth_--;
    }
    void VisitStmt_(const RealizeNode* op) final {
      scope_depth_++;
      this->VisitStmt(op->body);
      scope_depth_--;
    }

    int scope_depth_{0};
  };

  PrepCodeChecker()(full_body);

  if (auto op = full_body.as<SeqStmtNode>()) {
    if (auto attr = op->seq[0].as<AttrStmtNode>()) {
      if (attr->attr_key == attr::prep_code_scope) {
        *p_prep_code = op->seq[0];
        Array<Stmt> remaining;
        for (size_t i = 1; i < op->seq.size(); ++i) {
          remaining.push_back(op->seq[i]);
        }
        *p_main_body = SeqStmt(remaining);
        return Downcast<Map<Buffer, Buffer>>(attr->node);
      }
    }
  }

  *p_prep_code = EvaluateNode::make(0);
  *p_main_body = full_body;
}

Stmt MergeNest(const std::vector<Stmt>& nest, Stmt body) {
  // use reverse iteration
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    Stmt s = *ri;
    if (const auto* for_ = s.as<ForNode>()) {
      auto n = make_object<ForNode>(*for_);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* let = s.as<LetStmtNode>()) {
      auto n = make_object<LetStmtNode>(*let);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* attr = s.as<AttrStmtNode>()) {
      auto n = make_object<AttrStmtNode>(*attr);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* ite = s.as<IfThenElseNode>()) {
      auto n = make_object<IfThenElseNode>(*ite);
      CHECK(is_no_op(n->then_case));
      CHECK(!n->else_case.defined());
      n->then_case = body;
      body = Stmt(n);
    } else if (const auto* seq = s.as<SeqStmtNode>()) {
      auto n = make_object<SeqStmtNode>(*seq);
      CHECK(n->size() != 0 && is_no_op(n->seq[n->size() - 1]));
      n->seq.Set(n->size() - 1, body);
      body = Stmt(n);
    } else if (const auto* assert_ = s.as<AssertStmtNode>()) {
      auto n = make_object<AssertStmtNode>(*assert_);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* alloc = s.as<AllocateNode>()) {
      auto n = make_object<AllocateNode>(*alloc);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else {
      LOG(FATAL) << "not supported nest type";
    }
  }
  return body;
}

Stmt MergeNest(const std::vector<std::vector<Stmt>>& nest, Stmt body) {
  // std::cout << "[MN] In merge nest" << std::endl;
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    body = MergeNest(*ri, body);
  }
  return body;
}

}  // namespace tir
}  // namespace tvm
