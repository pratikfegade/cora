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
 * \file schedule_ops.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../../tir/ir/var_replacer.h"
#include "../../tir/pass/ir_util.h"
#include "../operation/op_util.h"
#include "function_generator.h"
#include "graph.h"
#include "schedule_utils.h"

namespace tvm {
namespace te {

using namespace tir;

Stmt MakePipeline(const Stage& s, const std::unordered_map<IterVar, Range>& dom_map,
                  const std::unordered_map<std::string, Range>& env_dom_map,
                  const std::unordered_map<std::string, IterVar>& env_var_map,
                  const std::unordered_map<const VarNode*, std::string>& bind_map,
                  const AttachPathWithStages& attach_path, Stmt consumer,
                  bool debug_keep_trivial_loop) {
  Stmt producer =
      s->op->BuildProvide(s, dom_map, env_dom_map, env_var_map, bind_map, attach_path.second,
                          attach_path.first, debug_keep_trivial_loop);

  if (producer.defined()) {
    producer = ProducerConsumerNode::make(s->op, true, producer);
  }
  if (s->double_buffer) {
    producer = AttrStmtNode::make(s->op, tir::attr::double_buffer_scope, 1, producer);
  }
  Stmt pipeline = producer;

  if (consumer.defined() && !is_no_op(consumer)) {
    consumer = ProducerConsumerNode::make(s->op, false, consumer);
    pipeline = SeqStmt({producer, consumer});
  }
  pipeline = s->op->BuildRealize(s, dom_map, pipeline);

  // use attribute to mark scope of the operation.
  pipeline =
      AttrStmtNode::make(s->op, tir::attr::realize_scope, StringImmNode::make(s->scope), pipeline);

  if (s->is_opengl) {
    pipeline =
        AttrStmtNode::make(s->op, tir::attr::opengl_stage_scope, StringImmNode::make(""), pipeline);
  }
  // if (s->op->name == "B.shared") {
  // std::cout << "[SO] Pipeline\n" << pipeline << std::endl;
  // }
  return pipeline;
}

// inject the operator's realization on the stmt.
class InjectAttach : public StmtMutator {
 public:
  InjectAttach(const Stage& stage, const Stage& attach_spec,
               const std::unordered_map<IterVar, Range>& dom_map,
               const std::unordered_map<std::string, Range>& env_dom_map,
               const std::unordered_map<std::string, IterVar>& env_var_map,
               const std::unordered_map<const VarNode*, std::string>& bind_map,
               const AttachPathWithStages& attach_path, bool debug_keep_trivial_loop)
      : stage_(stage),
        attach_spec_(attach_spec),
        dom_map_(dom_map),
        env_dom_map_(env_dom_map),
        env_var_map_(env_var_map),
        bind_map_(bind_map),
        attach_path_(attach_path),
        debug_keep_trivial_loop_(debug_keep_trivial_loop) {}

  Stmt VisitStmt(const Stmt& input_stmt) final {
    CHECK(input_stmt.defined());
    auto stmt = StmtMutator::VisitStmt(input_stmt);
    const AttrStmtNode* op = stmt.as<AttrStmtNode>();
    if (op != nullptr && op->attr_key == attr::loop_scope) {
      if ((attach_spec_->attach_type == kScope ||
           attach_spec_->attach_type == kSingleKernelScope) &&
          op->node == attach_spec_->attach_ivar) {
        CHECK(!found_attach) << "Find IterVar " << attach_spec_->attach_ivar
                             << " in multiple places in the IR " << input_stmt;
        found_attach = true;
        stmt =
            AttrStmtNode::make(op->node, op->attr_key, op->value,
                               MakePipeline(stage_, dom_map_, env_dom_map_, env_var_map_, bind_map_,
                                            attach_path_, op->body, debug_keep_trivial_loop_));
      }
    }
    return stmt;
  }
  // whether attach point is found
  bool found_attach{false};

 private:
  // The stage.
  const Stage& stage_;
  // The attach spec, may not contain op.
  const Stage& attach_spec_;
  // domain map
  const std::unordered_map<IterVar, Range>& dom_map_;
  const std::unordered_map<std::string, Range> env_dom_map_;
  const std::unordered_map<std::string, IterVar> env_var_map_;
  const std::unordered_map<const VarNode*, std::string>& bind_map_;
  const AttachPathWithStages& attach_path_;
  // Whether keep trivial loops with extent of 1 during lowering.
  // This is a debug feature for dataflow/axis analysis
  bool debug_keep_trivial_loop_;
};

// inject the operator's realization on the stmt.
class InjectScanStep : public StmtMutator {
 public:
  InjectScanStep(const Stage& stage, const Operation& scan_op,
                 const std::unordered_map<IterVar, Range>& dom_map,
                 const std::unordered_map<std::string, Range>& env_dom_map,
                 const std::unordered_map<std::string, IterVar>& env_var_map,
                 const std::unordered_map<const VarNode*, std::string>& bind_map,
                 const AttachPathWithStages& attach_path, bool is_init,
                 bool debug_keep_trivial_loop)
      : stage_(stage),
        scan_op_(scan_op),
        dom_map_(dom_map),
        env_dom_map_(env_dom_map),
        env_var_map_(env_var_map),
        bind_map_(bind_map),
        attach_path_(attach_path),
        is_init_(is_init),
        debug_keep_trivial_loop_(debug_keep_trivial_loop) {}

  Stmt VisitStmt(const Stmt& input_stmt) final {
    CHECK(input_stmt.defined());
    auto stmt = StmtMutator::VisitStmt(input_stmt);
    // update
    const AttrStmtNode* op = stmt.as<AttrStmtNode>();
    if (op != nullptr && ((op->attr_key == attr::scan_update_scope && !is_init_) ||
                          (op->attr_key == attr::scan_init_scope && is_init_))) {
      if (op->node.same_as(scan_op_)) {
        // std::cout << "[OPS] Injecting " << stage_->op << " at " << scan_op_ << std::endl;
        found_attach = true;
        stmt =
            AttrStmtNode::make(op->node, op->attr_key, op->value,
                               MakePipeline(stage_, dom_map_, env_dom_map_, env_var_map_, bind_map_,
                                            attach_path_, op->body, debug_keep_trivial_loop_));
      }
    }
    return stmt;
  }

  // whether attach point is found
  bool found_attach{false};

 private:
  // the operations to be carried
  const Stage& stage_;
  const Operation& scan_op_;
  // domain map
  const std::unordered_map<IterVar, Range>& dom_map_;
  const std::unordered_map<std::string, Range> env_dom_map_;
  const std::unordered_map<std::string, IterVar> env_var_map_;
  const std::unordered_map<const VarNode*, std::string>& bind_map_;
  const AttachPathWithStages& attach_path_;
  // whether it is init.
  bool is_init_;
  // Whether keep trivial loops with extent of 1 during lowering.
  // This is a debug feature for dataflow/axis analysis
  bool debug_keep_trivial_loop_;
};

// inject the operator's realization on the stmt.
class InjectConditionalStep : public StmtMutator {
 public:
  InjectConditionalStep(const Stage& stage, const Operation& conditional_op,
                        const std::unordered_map<IterVar, Range>& dom_map,
                        const std::unordered_map<std::string, Range>& env_dom_map,
                        const std::unordered_map<std::string, IterVar>& env_var_map,
                        const std::unordered_map<const VarNode*, std::string>& bind_map,
                        const AttachPathWithStages& attach_path, bool is_else,
                        bool debug_keep_trivial_loop)
      : stage_(stage),
        conditional_op_(conditional_op),
        dom_map_(dom_map),
        env_dom_map_(env_dom_map),
        env_var_map_(env_var_map),
        bind_map_(bind_map),
        attach_path_(attach_path),
        is_else_(is_else),
        debug_keep_trivial_loop_(debug_keep_trivial_loop) {}

  Stmt VisitStmt(const Stmt& input_stmt) final {
    CHECK(input_stmt.defined());
    auto stmt = StmtMutator::VisitStmt(input_stmt);
    // update
    const AttrStmtNode* op = stmt.as<AttrStmtNode>();
    if (op != nullptr && ((op->attr_key == attr::conditional_then_scope && !is_else_) ||
                          (op->attr_key == attr::conditional_else_scope && is_else_))) {
      if (op->node.same_as(conditional_op_)) {
        // std::cout << "[OPS] Injecting " << stage_->op << " at " << conditional_op_ <<
        // std::endl;
        found_attach = true;
        stmt =
            AttrStmtNode::make(op->node, op->attr_key, op->value,
                               MakePipeline(stage_, dom_map_, env_dom_map_, env_var_map_, bind_map_,
                                            attach_path_, op->body, debug_keep_trivial_loop_));
      }
    }
    return stmt;
  }

  // whether attach point is found
  bool found_attach{false};

 private:
  // the operations to be carried
  const Stage& stage_;
  const Operation& conditional_op_;
  // domain map
  const std::unordered_map<IterVar, Range>& dom_map_;
  const std::unordered_map<std::string, Range> env_dom_map_;
  const std::unordered_map<std::string, IterVar> env_var_map_;
  const std::unordered_map<const VarNode*, std::string>& bind_map_;
  const AttachPathWithStages& attach_path_;
  // whether it is init.
  bool is_else_;
  // Whether keep trivial loops with extent of 1 during lowering.
  // This is a debug feature for dataflow/axis analysis
  bool debug_keep_trivial_loop_;
};

// inject the operator's realization on the stmt.
class InjectSingleKernelInput : public StmtMutator {
 public:
  InjectSingleKernelInput(const Stage& stage, const Operation& single_kernel_op,
                          const std::unordered_map<IterVar, Range>& dom_map,
                          const std::unordered_map<std::string, Range>& env_dom_map,
                          const std::unordered_map<std::string, IterVar>& env_var_map,
                          const std::unordered_map<const VarNode*, std::string>& bind_map,
                          const AttachPathWithStages& attach_path, bool is_init,
                          bool debug_keep_trivial_loop)
      : stage_(stage),
        single_kernel_op_(single_kernel_op),
        dom_map_(dom_map),
        env_dom_map_(env_dom_map),
        env_var_map_(env_var_map),
        bind_map_(bind_map),
        attach_path_(attach_path),
        is_init_(is_init),
        debug_keep_trivial_loop_(debug_keep_trivial_loop) {}

  Stmt VisitStmt(const Stmt& input_stmt) final {
    CHECK(input_stmt.defined());
    auto stmt = StmtMutator::VisitStmt(input_stmt);
    // update
    const AttrStmtNode* op = stmt.as<AttrStmtNode>();
    if (op != nullptr && ((op->attr_key == attr::single_kernel_input_scope))) {
      if (op->node.same_as(single_kernel_op_)) {
        found_attach = true;
        stmt =
            AttrStmtNode::make(op->node, op->attr_key, op->value,
                               MakePipeline(stage_, dom_map_, env_dom_map_, env_var_map_, bind_map_,
                                            attach_path_, op->body, debug_keep_trivial_loop_));
        // std::cout << "[BODY] " << stmt << std::endl;
      }
    }
    return stmt;
  }

  // whether attach point is found
  bool found_attach{false};

 private:
  // the operations to be carried
  const Stage& stage_;
  const Operation& single_kernel_op_;
  // domain map
  const std::unordered_map<IterVar, Range>& dom_map_;
  const std::unordered_map<std::string, Range> env_dom_map_;
  const std::unordered_map<std::string, IterVar> env_var_map_;
  const std::unordered_map<const VarNode*, std::string>& bind_map_;
  const AttachPathWithStages& attach_path_;
  // whether it is init.
  bool is_init_;
  // Whether keep trivial loops with extent of 1 during lowering.
  // This is a debug feature for dataflow/axis analysis
  bool debug_keep_trivial_loop_;
};

// Postprocessing of schedule op
// Replace the init and update's expression by scan's buffer.
class SchedulePostProc : public StmtExprMutator {
 public:
  Stmt VisitStmt_(const ProducerConsumerNode* op) final {
    auto it = replace_op_.find(op->func.get());
    if (it != replace_op_.end()) {
      Stmt body = this->VisitStmt(op->body);
      if (it->second.defined()) {
        return ProducerConsumerNode::make(it->second, op->is_producer, body);
      } else {
        return body;
      }
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const LetStmtNode* op) final {
    if (!HasSideEffect(op->value)) {
      var_value_[op->var.get()] = this->VisitExpr(op->value);
      return this->VisitStmt(op->body);
      // return StmtExprMutator::VisitStmt_(op);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::loop_scope || op->attr_key == attr::scan_init_scope) {
      return this->VisitStmt(op->body);
    } else if (op->attr_key == attr::scan_update_scope) {
      const ScanOpNode* scan = op->node.as<ScanOpNode>();
      CHECK(scan);
      var_value_[scan->scan_axis->var.get()] = op->value;
      return this->VisitStmt(op->body);
    } else if (op->attr_key == attr::thread_extent) {
      // delete duplicated thread extent attr
      auto it = thread_extent_scope_.find(op->node.get());
      if (it != thread_extent_scope_.end()) {
        CHECK(is_zero(tir::Simplify(it->second - op->value)));
        return this->VisitStmt(op->body);
      } else {
        thread_extent_scope_[op->node.get()] = op->value;
        Stmt ret = StmtExprMutator::VisitStmt_(op);
        thread_extent_scope_.erase(op->node.get());
        return ret;
      }
    } else if (op->attr_key == tir::attr::realize_scope ||
               op->attr_key == tir::attr::double_buffer_scope) {
      auto it = replace_op_.find(op->node.get());
      if (it != replace_op_.end()) {
        if (it->second.defined()) {
          Stmt ret = AttrStmtNode::make(it->second, op->attr_key, op->value, op->body);
          return this->VisitStmt(ret);
        } else {
          return this->VisitStmt(op->body);
        }
      }
    } else if (op->attr_key == tir::attr::buffer_bind_scope) {
      Array<ObjectRef> tuple = Downcast<Array<ObjectRef>>(op->node);
      Tensor tensor = Downcast<Tensor>(tuple[1]);
      auto it = replace_op_.find(tensor->op.get());
      if (it != replace_op_.end()) {
        if (it->second.defined()) {
          return AttrStmtNode::make(
              Array<ObjectRef>{tuple[0], it->second.output(tensor->value_index)}, op->attr_key,
              op->value, this->VisitStmt(op->body));
        } else {
          return this->VisitStmt(op->body);
        }
      }
    } else if (op->attr_key == tir::attr::buffer_dim_align) {
      Tensor tensor = Downcast<Tensor>(op->node);
      auto it = replace_op_.find(tensor->op.get());
      if (it != replace_op_.end()) {
        if (it->second.defined()) {
          return AttrStmtNode::make(it->second.output(tensor->value_index), op->attr_key, op->value,
                                    this->VisitStmt(op->body));
        } else {
          return this->VisitStmt(op->body);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const RealizeNode* op) final {
    Region processed_bounds;

    bool to_relax = false;
    if (op2stage_cache_.size() > 0) {
      CHECK(op2stage_cache_.count(op->func.get())) << op->func;
      to_relax = !op2stage_cache_.at(op->func.get()).is_ancestor_attached_at_root();
    }

    // std::cout << "[RRB] Func " << op->func << " " << to_relax << std::endl;

    for (const auto& bound : op->bounds) {
      if (to_relax) {
        auto extent1 = bound->extent;
        auto extent2 = this->VisitExpr(extent1);
        auto extent3 = Simplify(extent2);
        auto extent4 = UninterpFun::RelaxUninterpCallsMaxInclusive(extent3, false);

        // std::cout << "[RRB]  E1 " << extent1 << std::endl;
        // std::cout << "[RRB]   2 " << extent2 << std::endl;
        // std::cout << "[RRB]   3 " << extent3 << std::endl;
        // std::cout << "[RRB]   4 " << extent4 << std::endl;

        Range replaced = Range::make_by_min_extent(
            UninterpFun::InlineUninterpFunCalls(this->VisitExpr(bound->min)),
            UninterpFun::InlineUninterpFunCalls(extent4));
        processed_bounds.push_back(replaced);

      } else {
        Range replaced = Range::make_by_min_extent(
            UninterpFun::InlineUninterpFunCalls(this->VisitExpr(bound->min)),
            UninterpFun::InlineUninterpFunCalls(to_relax
                                                    ? UninterpFun::RelaxUninterpCallsMaxInclusive(
                                                          Simplify(this->VisitExpr(bound->extent)))
                                                    : Simplify(this->VisitExpr(bound->extent))));
        processed_bounds.push_back(replaced);
      }
    }

    TensorKey key{op->func, op->value_index};
    auto it = replace_realize_.find(key);
    if (it != replace_realize_.end()) {
      if (it->second.defined()) {
        Stmt ret = RealizeNode::make(it->second->op, it->second->value_index, op->dtype,
                                     processed_bounds, op->condition, op->body);
        return this->VisitStmt(ret);
      } else {
        return this->VisitStmt(op->body);
      }
    } else {
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      const auto realize = stmt.as<RealizeNode>();
      return RealizeNode::make(realize->func, realize->value_index, realize->dtype,
                               processed_bounds, realize->condition, realize->body);
    }
  }

  Stmt VisitStmt_(const ProvideNode* op) final {
    TensorKey key{op->func, op->value_index};
    auto it = replace_buffer_.find(key);
    if (it != replace_buffer_.end()) {
      const Tensor& dst = it->second;
      // std::cout << "[PP] Replacing " << op->func << " " << dst->op << " "
      // << dst->op->attrs.count("no_sync") << std::endl;

      Stmt ret = ProvideNode::make(dst->op, dst->value_index, op->value, op->args,
                                   op->custom_realize_bounds);
      return this->VisitStmt(ret);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->call_type == CallNode::Halide) {
      TensorKey key{op->func, op->value_index};
      auto it = replace_buffer_.find(key);
      if (it != replace_buffer_.end()) {
        const Tensor& dst = it->second;
        // std::cout << "[PP] Replacing " << op->func << " " << dst->op << std::endl;
        PrimExpr ret =
            CallNode::make(op->dtype, dst->op->name, op->args, op->call_type, op->arg_dims, dst->op,
                           dst->value_index, op->custom_realize_bounds);
        return this->VisitExpr(ret);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_value_.find(op);
    if (it != var_value_.end()) {
      return it->second;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  void InitToReplaceForEnvelopeOps(Schedule& sch) {
    this->thread_extent_scope_.clear();
    this->var_value_.clear();
    this->replace_buffer_.clear();
    this->replace_realize_.clear();
    this->replace_op_.clear();
    this->op2stage_cache_.clear();

    for (Stage s : sch->stages) {
      for (auto kv : s->iter_var_attrs) {
        // Update bind thread information.
        if (kv.second->bind_thread.defined()) {
          const Var& from = kv.first->var;
          const Var& to = kv.second->bind_thread->var;
          CHECK(!var_value_.count(from.get()));
          var_value_[from.get()] = to;
        }
      }
      // Specially add replacements for scan op.
      if (const ScanOpNode* scan = s->op.as<ScanOpNode>()) {
        for (size_t i = 0; i < scan->update.size(); ++i) {
          Tensor t = s->op.output(i);
          AddReplace(scan->init[i], t);
          AddReplace(scan->update[i], t);
          AddReplace(scan->state_placeholder[i], t);
          // std::cout << "[PP] Adding replacement Sc " << scan->init[i]->op << " " << t->op
          //           << std::endl;
          // std::cout << "[PP] Adding replacement Sc " << scan->update[i]->op << " " << t->op
          //           << std::endl;
          // std::cout << "[PP] Adding replacement Sc " << scan->state_placeholder[i]->op << " "
          //           << t->op << std::endl;
        }
      }

      // and for ConditionalOp
      if (const ConditionalOpNode* conditional = s->op.as<ConditionalOpNode>()) {
        for (size_t i = 0; i < conditional->then_case.size(); ++i) {
          Tensor t = s->op.output(i);
          AddReplace(conditional->then_case[i], t);
          AddReplace(conditional->else_case[i], t);
        }
      }

      // and for SpecializationEnvelopeOp
      if (const SpecializationEnvelopeOpNode* scanEnv = s->op.as<SpecializationEnvelopeOpNode>()) {
        for (int i = 0; i < scanEnv->num_outputs(); ++i) {
          Tensor t = s->op.output(i);
          for (auto input : scanEnv->inputs) {
            AddReplace(input[i], t);
          }
        }
      }

      if (const SingleKernelEnvelopeOpNode* scanEnv = s->op.as<SingleKernelEnvelopeOpNode>()) {
        for (int i = 0; i < scanEnv->num_outputs(); ++i) {
          Tensor t = s->op.output(i);
          Tensor input = scanEnv->inputs[i];
          AddReplace(input, t);
        }
      }
    }

    sch->InvalidateCache();
    sch->InitCache();
    op2stage_cache_ = sch->op2stage_cache_;
  }

  void InitToReplaceOriginOps(const Schedule& sch) {
    this->thread_extent_scope_.clear();
    this->var_value_.clear();
    this->replace_buffer_.clear();
    this->replace_realize_.clear();
    this->replace_op_.clear();
    this->op2stage_cache_.clear();

    for (Stage s : sch->stages) {
      // This must be checked for all ops, including scan.
      if (!s->op.same_as(s->origin_op)) {
        for (int i = 0; i < s->op->num_outputs(); ++i) {
          Tensor target = s->origin_op.output(i);
          AddReplace(s->op.output(i), target, target, s->origin_op);
          // std::cout << "[PP] Adding replacement Or " << s->op << " " << s->origin_op <<
          // std::endl;
        }
      }
    }
  }

 private:
  void AddReplace(Tensor src, Tensor dst, Tensor repl_realize = Tensor(),
                  Operation repl_op = Operation()) {
    TensorKey key{src->op, src->value_index};
    replace_buffer_[key] = dst;
    replace_realize_[key] = repl_realize;
    replace_op_[src->op.get()] = repl_op;
  }
  // The thread extent scope.
  std::unordered_map<const Object*, PrimExpr> thread_extent_scope_;
  // The scan value
  std::unordered_map<const VarNode*, PrimExpr> var_value_;
  // buffer replacement
  std::unordered_map<TensorKey, Tensor> replace_buffer_;
  // buffere realization to be replaced
  std::unordered_map<TensorKey, Tensor> replace_realize_;
  // replace producer consumer.
  std::unordered_map<const Object*, Operation> replace_op_;
  // replace producer consumer.
  std::unordered_map<const Object*, Stage> op2stage_cache_;
};

class EnvThreadReplacer : public StmtExprMutator {
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      // delete duplicated thread extent attr
      IterVar thread = Downcast<IterVar>(op->node);
      std::string name = thread->var->name_hint;
      if (isCudaThread(thread) || isCPUEnvThread(thread)) {
        if (!env_thread_map.count(name)) {
          env_thread_map[name] = thread->var;
          env_dom_map[name] = thread->dom;
          Stmt body = StmtExprMutator::VisitStmt(op->body);
          env_thread_map.erase(name);
          env_dom_map.erase(name);
          return AttrStmtNode::make(op->node, op->attr_key, op->value, body, op->hfuse_group_id);
        } else {
          return StmtExprMutator::VisitStmt(op->body);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode* op) {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    if (print) {
      std::cout << "[EnvTh] Visiting " << GetRef<Stmt>(op) << std::endl;
      print = false;
    }
    return ret;
  }

  PrimExpr VisitExpr_(const VarNode* op) {
    if (env_thread_map.count(op->name_hint)) {
      if (var_dom_map.count(op) && env_dom_map.count(op->name_hint) &&
          env_dom_map.at(op->name_hint).defined()) {
        Range old_range = var_dom_map.at(op);
        Range new_range = env_dom_map.at(op->name_hint);
        PrimExpr old_extent =
            arith::Simplify(UninterpFun::InlineUninterpFunCalls(old_range->extent));
        PrimExpr new_extent =
            arith::Simplify(UninterpFun::InlineUninterpFunCalls(new_range->extent));
        if (!ana.CanProve(new_extent >= old_extent)) {
          std::unordered_map<const VarNode*, IntSet> is_var_dom_map;
          for (auto it : var_dom_map) {
            is_var_dom_map[it.first] = IntSet::range(it.second);
          }
          IntSet evaled = EvalSet(processExtent(old_extent), is_var_dom_map);
          PrimExpr max_old_extent =
              arith::Simplify(UninterpFun::InlineUninterpFunCalls(evaled.max()));
          if (new_extent.dtype() != max_old_extent.dtype() ||
              !ana.CanProve(new_extent >= max_old_extent)) {
            CHECK(false) << "[EnvTh] BADBAD " << op->name_hint << " " << old_extent << " "
                         << new_extent << " " << max_old_extent << std::endl;
            // std::cout << "[EnvTh] BADBAD " << op->name_hint << " " << old_extent << " " <<
            // new_extent
            // 	      << " " << max_old_extent << std::endl;
            print = true;
          }
        }
      }
      return env_thread_map.at(op->name_hint);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr processExtent(PrimExpr e) {
    class ExtentProcessor : public ExprMutator {
     public:
      explicit ExtentProcessor(std::unordered_map<std::string, Var> env_thread_map_,
                               std::unordered_map<const VarNode*, std::string> bind_map_)
          : env_thread_map(env_thread_map_), bind_map(bind_map_) {}

      PrimExpr VisitExpr_(const VarNode* op) {
        if (bind_map.count(op)) {
          return env_thread_map.at(bind_map.at(op));
        } else
          return GetRef<PrimExpr>(op);
      };

     private:
      std::unordered_map<std::string, Var> env_thread_map;
      std::unordered_map<const VarNode*, std::string> bind_map;
    };

    ExtentProcessor extentProcessor(env_thread_map, bind_map);
    auto ret = arith::Simplify(extentProcessor(e));
    return ret;
  }

  bool print = false;
  arith::Analyzer ana;
  std::unordered_map<std::string, Var> env_thread_map;
  std::unordered_map<std::string, Range> env_dom_map;
  std::unordered_map<const VarNode*, Range> var_dom_map;
  std::unordered_map<const VarNode*, std::string> bind_map;

 public:
  EnvThreadReplacer(Map<IterVar, Range> dom_map,
                    std::unordered_map<const VarNode*, std::string> bind_map_)
      : bind_map(bind_map_) {
    for (auto it : dom_map) {
      var_dom_map[it.first->var.as<VarNode>()] = it.second;
    }
  }
};

class SimplifyFusionFunctions : public StmtExprMutator {
 public:
  SimplifyFusionFunctions(const Schedule& sch) {
    for (auto stage : sch->stages) {
      for (auto rel : stage->relations) {
        if (auto rrel = rel.as<RaggedFuseNode>()) {
          fused_functions.insert(rrel->fused_to_inner_uf.get());
          fused_functions.insert(rrel->fused_to_outer_uf.get());
          fused_functions.insert(rrel->outer_inner_to_fused_uf.get());
        }
      }
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) override {
    if (auto ufun = op->func.as<UninterpFunNode>()) {
      if (fused_functions.count(ufun)) {
        bool args_zero = true;
        for (PrimExpr arg : op->args) {
          arg = this->VisitExpr(arg);
          if (!is_zero(arg)) args_zero = false;
        }
        if (args_zero) return IntImm(DataType::Int(32), 0);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

 private:
  std::unordered_set<const Object*> fused_functions;
};

Stmt ScheduleOps(Schedule sch, InferBoundsResult bounds, bool debug_keep_trivial_loop,
                 bool distinct_device, bool debug_fill_function_bodies,
                 Array<Buffer> afuns_needed_for) {
  Map<IterVar, Range> dom_map_ = bounds->bounds;
  Map<Stage, Map<std::string, Range>> env_dom_map_ = bounds->env_bounds;
  Map<Stage, Map<std::string, IterVar>> env_var_map_ = bounds->env_vars;
  std::unordered_map<IterVar, Range> dom_map = as_unordered_map(dom_map_);

  std::unordered_map<const VarNode*, std::string> bind_map;

  std::cout << "[SCHED] Scheding ops" << std::endl;
  for (Stage stage : sch->stages) {
    std::cout << "[SCHED]  op " << stage->op << std::endl;
    for (auto kv : stage->iter_var_attrs) {
      if (kv.second->bind_thread.defined()) {
        bind_map[kv.first->var.as<VarNode>()] = kv.second->bind_thread->var->name_hint;
      }
    }
  }

  // Generate A functions for all layouts
  FunctionGenerator function_generator(sch, dom_map, distinct_device, debug_fill_function_bodies,
                                       afuns_needed_for);
  function_generator.GenerateAFunctions();
  PrimExpr afun_buf_size = function_generator.GetCurrentAggregateBufferSize();
  // Map<Buffer, Buffer> prep_buffer_map;
  // AFunGenerator generator(sch);
  // Stmt a_fun_stmt = generator.GenerateAndSetAFuns(&prep_buffer_map);

  sch.freeze_tensor_dimensions(dom_map_);

  Stmt body = Stmt();
  // scan init and scan updates
  std::unordered_map<Operation, Operation> scan_init;
  std::unordered_map<Operation, Operation> single_kernel_inputs;
  for (Stage s : sch->stages) {
    if (const ScanOpNode* scan = s->op.as<ScanOpNode>()) {
      for (Tensor t : scan->init) {
        if (scan_init.count(t->op)) {
          CHECK(scan_init.at(t->op).same_as(s->op))
              << "Scan init tensor can only belong to one scan";
        } else {
          scan_init[t->op] = s->op;
        }
      }
    } else if (const SingleKernelEnvelopeOpNode* single_kernel =
                   s->op.as<SingleKernelEnvelopeOpNode>()) {
      for (const auto& t : single_kernel->inputs) {
        if (single_kernel_inputs.count(t->op)) {
          CHECK(single_kernel_inputs.at(t->op).same_as(s->op))
              << "Scan envelope input tensor can only belong to one scan envelope";
        } else {
          single_kernel_inputs[t->op] = s->op;
        }
      }
    }
  }
  // verify correctness of group.
  for (Stage g : sch->groups) {
    CHECK(!g->op.defined());
    CHECK_EQ(g->leaf_iter_vars.size(), 0U);
  }

  // Throw errors if a non-root op has been given a ragged layout. An
  // ideal solution would be to actually make the layout dense for
  // such operations, rather than throwing an error.
  // for (auto stage : sch->stages) {
  //   if (stage.is_ancestor_attached_at_root()) continue;
  //   for (size_t i = 0; i < static_cast<size_t>(stage->op->num_outputs()); ++i) {
  //     Modes layout = stage->op->output_layout(i);
  //     CHECK(!layout.defined() || !layout->is_ragged())
  //         << "The operation " << stage->op
  //         << " which is attached at a non-root position has been asked to have a ragged "
  //            "layout. That is not yet supported ";
  //   }
  // }

  std::unordered_map<int, std::unordered_set<const StageNode*>> hfuse_groups;
  std::unordered_map<const Object*, int> hfuse_group_mapping;

  for (auto s : sch->stages) {
    for (size_t i = 0; i < s->leaf_iter_vars.size(); ++i) {
      auto iv = s->leaf_iter_vars[i];
      IterVarAttr it_attr;
      if (s->iter_var_attrs.count(iv)) {
        it_attr = s->iter_var_attrs[iv];
        if (it_attr.defined() && it_attr->hfuse_group_id >= 0) {
          CHECK(s.is_ancestor_attached_at_root())
              << "Only stages attached at the root can be hfused";
          CHECK_EQ(i, 0) << "Only the outermost leaf itervars can be hfused";
          CHECK(!hfuse_group_mapping.count(s.get())) << "One op cannot be hfused multiple times";
          CHECK(it_attr->bind_thread.defined() || iv->iter_type == kParallelized)
              << "We only allow hfusion for parallel loops right now as storage_rewrite does not "
                 "handle the sequential loop case correctly ";
          hfuse_group_mapping[s.get()] = it_attr->hfuse_group_id;
          auto it = hfuse_groups.find(it_attr->hfuse_group_id);
          if (it == hfuse_groups.end()) {
            hfuse_groups[it_attr->hfuse_group_id] = {s.operator->()};
          } else {
            it->second.insert(s.operator->());
          }
        }
      }
    }
  }

  std::vector<Stage> stage_order;
  std::unordered_set<const Object*> inserted;
  for (size_t i = sch->stages.size(); i != 0; --i) {
    Stage s = sch->stages[i - 1];
    if (!inserted.count(s.get())) {
      stage_order.push_back(s);
      inserted.insert(s.get());
    }
    auto it = hfuse_group_mapping.find(s.get());
    if (it != hfuse_group_mapping.end()) {
      for (auto sn : hfuse_groups[it->second]) {
        if (!inserted.count(sn)) {
          stage_order.push_back(GetRef<Stage>(sn));
          inserted.insert(sn);
        }
      }
    }
  }
  AttachPathWithStages attach_path = CreateAttachPathWithStages(sch);

  // std::cout << "[SO] Original Order " << std::endl;
  // for (auto s : sch->stages) {
  //   std::cout << "[SO]   " << s << std::endl;
  // }
  // std::cout << "[SO] New Order " << std::endl;
  // for (auto s : stage_order) {
  //   std::cout << "[SO]   " << s << std::endl;
  // }

  // std::cout << "[SO] Generating code" << std::endl;
  // reverse the post DFS order.
  int previous_hfuse_group_id = -1;
  // for (size_t i = stage_order.size(); i != 0; --i) {
  // Stage s = stage_order[i - 1];
  for (size_t i = 0; i < stage_order.size(); ++i) {
    Stage s = stage_order[i];

    int current_hfuse_group_id = -1;
    auto it = hfuse_group_mapping.find(s.get());
    if (it != hfuse_group_mapping.end()) {
      current_hfuse_group_id = it->second;
    }

    if (previous_hfuse_group_id < 0 && current_hfuse_group_id < 0) {
      // std::cout << "[SO]  Continue none for " << s << std::endl;
    } else if (previous_hfuse_group_id < 0 && current_hfuse_group_id >= 0) {
      // std::cout << "[SO]  Start for " << s << std::endl;
    } else if (previous_hfuse_group_id >= 0 && current_hfuse_group_id < 0) {
      // std::cout << "[SO]  End for " << s << std::endl;
      body = AttrStmtNode::make(IntImm(0), attr::hfuse_group, 0, body);
    } else {
      // std::cout << "[SO]  Continue for " << s << std::endl;
    }

    // std::cout << "[SO] Body\n" << body << std::endl;

    CHECK_NE(s->attach_type, kInline) << "call schedule.normalize before scheduleops";
    CHECK(s->op.defined());
    // no need to specify place holder op.
    if (s->op.as<PlaceholderOpNode>()) continue;
    // Remove grouping sugar, get the real attach spec.
    Stage attach_spec = s.GetAttachSpec();
    // std::cout << "[OPS] Stage " << s << std::endl;

    std::unordered_map<std::string, Range> env_dom_map = as_unordered_map(env_dom_map_.at(s));
    std::unordered_map<std::string, IterVar> env_var_map = as_unordered_map(env_var_map_.at(s));

    if (scan_init.count(s->op)) {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      CHECK(body.defined());
      InjectScanStep mu(s, scan_init.at(s->op), dom_map, env_dom_map, env_var_map, bind_map,
                        attach_path, true, debug_keep_trivial_loop);
      body = mu(std::move(body));
      CHECK(mu.found_attach) << "did not find attachment point for scan.init";
    } else if (attach_spec->attach_type == kSingleKernelScope) {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      CHECK(body.defined());
      InjectSingleKernelInput mu(s, attach_spec->attach_stage->op, dom_map, env_dom_map,
                                 env_var_map, bind_map, attach_path, true, debug_keep_trivial_loop);
      body = mu(std::move(body));
      CHECK(mu.found_attach) << "did not find attachment point for scan.update";
    } else if (attach_spec->attach_type == kScanUpdate) {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      // Handle scan update
      CHECK(body.defined());
      InjectScanStep mu(s, attach_spec->attach_stage->op, dom_map, env_dom_map, env_var_map,
                        bind_map, attach_path, false, debug_keep_trivial_loop);
      body = mu(std::move(body));
      CHECK(mu.found_attach) << "did not find attachment point for scan.update";
    } else if (attach_spec->attach_type == kConditionalThen) {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      // Handle scan update
      CHECK(body.defined());
      InjectConditionalStep mu(s, attach_spec->attach_stage->op, dom_map, env_dom_map, env_var_map,
                               bind_map, attach_path, false, debug_keep_trivial_loop);
      body = mu(std::move(body));
      CHECK(mu.found_attach) << "did not find attachment point for scan.update";
    } else if (attach_spec->attach_type == kConditionalElse) {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      // Handle scan update
      CHECK(body.defined());
      InjectConditionalStep mu(s, attach_spec->attach_stage->op, dom_map, env_dom_map, env_var_map,
                               bind_map, attach_path, true, debug_keep_trivial_loop);
      body = mu(std::move(body));
      CHECK(mu.found_attach) << "did not find attachment point for scan.update";
    } else if (attach_spec->attach_type == kInlinedAlready) {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      // do nothing
    } else if (attach_spec->attach_type == kGroupRoot) {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      CHECK(!s->group.defined());
      body = MakePipeline(s, dom_map, env_dom_map, env_var_map, bind_map, attach_path, body,
                          debug_keep_trivial_loop);
    } else {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      // CHECK_EQ(attach_spec->attach_type, kScope) << s;
      CHECK(attach_spec->attach_type == kScope || attach_spec->attach_type == kSingleKernelScope)
          << s;
      CHECK(body.defined());
      InjectAttach mutator(s, attach_spec, dom_map, env_dom_map, env_var_map, bind_map, attach_path,
                           debug_keep_trivial_loop);
      // std::cout << "[BODY] "  << body << std::endl;
      body = mutator(std::move(body));
      CHECK(mutator.found_attach) << "did not find attachment point for " << s << " in "
                                  << attach_spec->attach_stage->op << " x "
                                  << attach_spec->attach_ivar << ", body:\n"
                                  << body;
    }

    previous_hfuse_group_id = current_hfuse_group_id;
  }

  // std::cout << "Body after gen " << body << std::endl;
  body = function_generator.SimplifyFusionFunctions(body);
  // std::cout << "Body after function simpl " << body << std::endl;
  // exit(0);
  function_generator.GenerateFusionFunctions();
  body = function_generator.CreateBody(body);

  PrimExpr total_buf_size = function_generator.GetCurrentAggregateBufferSize();

  // std::cout << "BUFFER_SIZES " << afun_buf_size << " " << (total_buf_size - afun_buf_size)
  // << std::endl;

  // std::cout << "Body after function gen " << body << std::endl;
  sch->InvalidateCache();
  sch->InitCache();
  SchedulePostProc post_proc;
  post_proc.InitToReplaceForEnvelopeOps(sch);
  Stmt ret1 = post_proc(std::move(body));
  // std::cout << "Body after postproc1 " << ret1 << std::endl;
  sch->InvalidateCache();
  sch->InitCache();
  post_proc.InitToReplaceOriginOps(sch);
  Stmt ret2 = post_proc(std::move(ret1));
  // EnvThreadReplacer env_replace(dom_map_, bind_map);
  // Stmt ret3 = env_replace(std::move(ret2));
  Stmt ret3 = std::move(ret2);
  // std::cout << "Body after postproc2 " << ret2 << std::endl;
  return UninterpFun::InlineUninterpFunCalls(ret3);
}

TVM_REGISTER_GLOBAL("schedule.ScheduleOps").set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args.size() == 2)
    *ret = ScheduleOps(args[0], args[1], false, true, true, {});
  else
    *ret = ScheduleOps(args[0], args[1], args[2], args[3], args[4], args[5]);
});

}  // namespace te
}  // namespace tvm
