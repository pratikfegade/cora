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
                  const std::unordered_map<const VarNode*, std::string>& bind_map, Stmt consumer,
                  bool debug_keep_trivial_loop) {
  Stmt producer =
      s->op->BuildProvide(s, dom_map, env_dom_map, env_var_map, bind_map, debug_keep_trivial_loop);
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

  // if (s->op->name == "unified") {
  // std::cout << "[NoRe] " << pipeline << std::endl;
  // }

  // use attribute to mark scope of the operation.
  pipeline =
      AttrStmtNode::make(s->op, tir::attr::realize_scope, StringImmNode::make(s->scope), pipeline);

  if (s->is_opengl) {
    pipeline =
        AttrStmtNode::make(s->op, tir::attr::opengl_stage_scope, StringImmNode::make(""), pipeline);
  }
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
               bool debug_keep_trivial_loop)
      : stage_(stage),
        attach_spec_(attach_spec),
        dom_map_(dom_map),
        env_dom_map_(env_dom_map),
        env_var_map_(env_var_map),
        bind_map_(bind_map),
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
        stmt = AttrStmtNode::make(op->node, op->attr_key, op->value,
                                  MakePipeline(stage_, dom_map_, env_dom_map_, env_var_map_,
                                               bind_map_, op->body, debug_keep_trivial_loop_));
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
                 const std::unordered_map<const VarNode*, std::string>& bind_map, bool is_init,
                 bool debug_keep_trivial_loop)
      : stage_(stage),
        scan_op_(scan_op),
        dom_map_(dom_map),
        env_dom_map_(env_dom_map),
        env_var_map_(env_var_map),
        bind_map_(bind_map),
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
        stmt = AttrStmtNode::make(op->node, op->attr_key, op->value,
                                  MakePipeline(stage_, dom_map_, env_dom_map_, env_var_map_,
                                               bind_map_, op->body, debug_keep_trivial_loop_));
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
                        bool is_else, bool debug_keep_trivial_loop)
      : stage_(stage),
        conditional_op_(conditional_op),
        dom_map_(dom_map),
        env_dom_map_(env_dom_map),
        env_var_map_(env_var_map),
        bind_map_(bind_map),
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
        // std::cout << "[OPS] Injecting " << stage_->op << " at " << conditional_op_ << std::endl;
        found_attach = true;
        stmt = AttrStmtNode::make(op->node, op->attr_key, op->value,
                                  MakePipeline(stage_, dom_map_, env_dom_map_, env_var_map_,
                                               bind_map_, op->body, debug_keep_trivial_loop_));
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
                          bool is_init, bool debug_keep_trivial_loop)
      : stage_(stage),
        single_kernel_op_(single_kernel_op),
        dom_map_(dom_map),
        env_dom_map_(env_dom_map),
        env_var_map_(env_var_map),
        bind_map_(bind_map),
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
        stmt = AttrStmtNode::make(op->node, op->attr_key, op->value,
                                  MakePipeline(stage_, dom_map_, env_dom_map_, env_var_map_,
                                               bind_map_, op->body, debug_keep_trivial_loop_));
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
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  // // Bad hacky place for this
  // arith::Analyzer ana;
  // PrimExpr VisitExpr_(const MulNode* op) final {
  //   if (ana.CanProve(op->a == 0.0f) || ana.CanProve(op->b == 0.0f)) return 0.0f;
  //   return GetRef<PrimExpr>(op);
  // }

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

    for (const auto& bound : op->bounds) {
      Range replaced = Range::make_by_min_extent(
          this->VisitExpr(UninterpFun::InlineUninterpFunCalls(bound->min)),
          this->VisitExpr(UninterpFun::InlineUninterpFunCalls(bound->extent)));
      processed_bounds.push_back(replaced);
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
      Stmt ret = ProvideNode::make(dst->op, dst->value_index, op->value, op->args);
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
        PrimExpr ret = CallNode::make(op->dtype, dst->op->name, op->args, op->call_type, dst->op,
                                      dst->value_index);
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

  void InitToReplaceForEnvelopeOps(const Schedule& sch) {
    this->thread_extent_scope_.clear();
    this->var_value_.clear();
    this->replace_buffer_.clear();
    this->replace_realize_.clear();
    this->replace_op_.clear();

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
            // std::cout << "[PP] Adding replacement Sp " << input[i]->op << " " << t->op <<
            // std::endl;
          }
        }
      }

      if (const SingleKernelEnvelopeOpNode* scanEnv = s->op.as<SingleKernelEnvelopeOpNode>()) {
        for (int i = 0; i < scanEnv->num_outputs(); ++i) {
          Tensor t = s->op.output(i);
          Tensor input = scanEnv->inputs[i];
          AddReplace(input, t);
          // std::cout << "[PP] Adding replacement Sk " << input->op << " " << t->op << std::endl;
        }
      }
    }
  }

  void InitToReplaceOriginOps(const Schedule& sch) {
    this->thread_extent_scope_.clear();
    this->var_value_.clear();
    this->replace_buffer_.clear();
    this->replace_realize_.clear();
    this->replace_op_.clear();

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
          return AttrStmtNode::make(op->node, op->attr_key, op->value, body);
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

class RaggedFusionBoundStmtsGenerator : public StmtExprMutator {
 public:
  RaggedFusionBoundStmtsGenerator(Schedule& sch_, std::unordered_map<IterVar, Range>& dom_map_)
      : sch(sch_), dom_map(dom_map_), count(0) {}

  Stmt generate(Stmt main_body) {
    Array<Stmt> fusion_stmts;
    for (Stage s : sch->stages) {
      std::vector<const RaggedFuseNode*> ragged_fuse_relations = get_ragged_fusion_relations(s);
      for (auto rel : ragged_fuse_relations) {
        Stmt fusion_stmt = generate_fusion_statements(s, rel);
        fusion_stmts.push_back(fusion_stmt);
      }
    }

    if (fusion_stmts.size() > 0) {
      fusion_stmts.push_back(main_body);
      return SeqStmt(fusion_stmts);
    } else {
      return main_body;
    }
  }

 private:
  std::vector<const RaggedFuseNode*> get_ragged_fusion_relations(Stage& stage) {
    std::vector<const RaggedFuseNode*> rels;
    for (auto rel : stage->relations) {
      if (auto frel = rel.as<RaggedFuseNode>()) {
        rels.push_back(frel);
      }
    }
    return rels;
  }

  PrimExpr root_ivs_fused(Stage& stage, Array<IterVar> fused_ivs) {
    // Modes loop_layout = stage->op->loop_layout();
    // if (!loop_layout.defined() || loop_layout->a_funs.size() == 0) {
    //   return NullValue<PrimExpr>();
    // }

    // std::unordered_map<const Object*, IterVar> root_vars_to_ivs;
    // Array<IterVar> root_vars = stage->op->root_iter_vars();
    // for (auto rv : stage->op->root_iter_vars()) {
    //   root_vars_to_ivs[rv->var.get()] = rv;
    // }

    // Array<PrimExpr> var_values = get_iter_var_values(fused_ivs, stage);
    // Array<Dimension> fused_dims;
    // Array<PrimExpr> fused_vars;
    // // These checks can be made less conservative
    // for (size_t i = 0; i < fused_ivs.size(); ++i) {
    //   Var var = fused_ivs[i]->var;
    //   PrimExpr value = var_values[i];
    //   std::cout << "[GFS]  Value " << var << " " << value << std::endl;
    //   auto var_node = value.as<VarNode>();
    //   if (var_node == nullptr || !root_vars_to_ivs.count(var_node)) {
    //     return NullValue<PrimExpr>();
    //   }
    //   IterVar root_iv = root_vars_to_ivs.at(var_node);
    //   fused_dims.push_back(loop_layout->dimensions[root_vars.GetIdx(root_iv)]);
    //   fused_vars.push_back(root_iv->var);
    // }

    // for (auto dim : fused_dims) {
    //   for (auto dependent_dim : loop_layout->get_dependent_dimensions(dim)) {
    //     if (!fused_dims.Contains(dependent_dim)) return NullValue<PrimExpr>();
    //   }
    // }

    // PrimExpr fused_var_val = loop_layout->ComputePosition("mummy", fused_vars, fused_dims);
    // std::cout << "[GFS]   Fused var val " << fused_var_val << std::endl;
    // return fused_var_val;
    return NullValue<PrimExpr>();
  }

  Stmt generate_fusion_statements(Stage& stage, const RaggedFuseNode* rel) {
    // std::cout << "[GFS] Generating fusion for " << stage << std::endl;
    CHECK(stage.is_ancestor_attached_at_root());

    IterVar outer = rel->outer;
    IterVar inner = rel->inner;
    IterVar fused = rel->fused;

    PrimExpr fused_var_val = root_ivs_fused(stage, {outer, inner});

    PrimExpr outer_extent = Simplify(UninterpFun::InlineUninterpFunCalls(
        UninterpFun::RelaxComplexUninterpCalls(dom_map.at(outer)->extent)));
    PrimExpr inner_extent = Simplify(UninterpFun::InlineUninterpFunCalls(
        UninterpFun::RelaxComplexUninterpCalls(dom_map.at(inner)->extent)));
    PrimExpr fused_extent = outer_extent * inner_extent;

    // std::cout << "[GFS]   Extents " << outer_extent << " " << inner_extent << " " << fused_extent
    // << std::endl;

    // Allocate buffers
    Buffer fused_to_inner =
        decl_buffer({fused_extent}, DataType::Int(32), "fi" + std::to_string(count));
    Buffer fused_to_outer =
        decl_buffer({fused_extent}, DataType::Int(32), "fo" + std::to_string(count));
    Buffer outer_inner_to_fused =
        decl_buffer({outer_extent, inner_extent}, DataType::Int(32), "oif" + std::to_string(count));
    Buffer fused_val = decl_buffer({1}, DataType::Int(32), "f" + std::to_string(count));
    count++;

    // Compute the outer and inner variables in terms of the root itervars
    PrimExpr outer_value = get_iter_var_values({outer}, stage)[0];
    PrimExpr inner_value = get_iter_var_values({inner}, stage)[0];

    Array<IterVar> root_vars_to_loop;
    VarCollector collector;
    collector.collect(outer_value);
    collector.collect(inner_value);
    auto vars = collector.getCollected();

    for (auto iv : stage->op->root_iter_vars()) {
      if (vars.count(iv->var.get())) root_vars_to_loop.push_back(iv);
    }

    std::vector<Stmt> for_loops;
    Stmt no_op = EvaluateNode::make(0);
    for (auto rv : root_vars_to_loop) {
      for_loops.push_back(ForNode::make(rv->var, rv->dom->min, rv->dom->extent, ForType::Serial,
                                        DeviceAPI::None, no_op));
    }

    Stmt body = NullValue<Stmt>();
    {
      PrimExpr fused_val_load = fused_val.vload({0}, DataType::Int(32));
      Stmt fused_store = EvaluateNode::make(0);
      if (!fused_var_val.defined()) {
        fused_store = outer_inner_to_fused.vstore({outer_value, inner_value}, fused_val_load);
      }
      Stmt outer_store = fused_to_outer.vstore({fused_val_load}, outer_value);
      Stmt inner_store = fused_to_inner.vstore({fused_val_load}, inner_value);
      Stmt fused_incr = fused_val.vstore({0}, fused_val_load + 1);
      body = SeqStmt({fused_store, outer_store, inner_store, fused_incr});
    }

    body = MergeNest(for_loops, body);
    body = SeqStmt({fused_val.vstore({0}, 0), body});
    body =
        AttrStmtNode::make(fused_to_inner->data, attr::storage_scope, StringImmNode::make("global"),
                           AllocateNode::make(fused_to_inner->data, DataType::Int(32),
                                              {fused_extent}, IntImm(DataType::Bool(1), 1), body));
    body =
        AttrStmtNode::make(fused_to_outer->data, attr::storage_scope, StringImmNode::make("global"),
                           AllocateNode::make(fused_to_outer->data, DataType::Int(32),
                                              {fused_extent}, IntImm(DataType::Bool(1), 1), body));
    body = AttrStmtNode::make(
        outer_inner_to_fused->data, attr::storage_scope, StringImmNode::make("global"),
        AllocateNode::make(outer_inner_to_fused->data, DataType::Int(32), {fused_extent},
                           IntImm(DataType::Bool(1), 1), body));
    body = AttrStmtNode::make(fused_val->data, attr::storage_scope, StringImmNode::make("global"),
                              AllocateNode::make(fused_val->data, DataType::Int(32), {1},
                                                 IntImm(DataType::Bool(1), 1), body));

    // std::cout << "[GFS]  Stmt\n" << body << std::endl;

    auto init_uf = [&](UninterpFun uf, PrimExpr max_extent, Buffer loadee,
                       PrimExpr body = NullValue<PrimExpr>()) {
      UninterpFunNode* uf_node = const_cast<UninterpFunNode*>(uf.as<UninterpFunNode>());
      Array<PrimExpr> extents;
      for (auto param : uf->parameters) extents.push_back(param);
      if (body.defined()) {
        uf_node->SetBody(body);
      } else {
        uf_node->SetBody(loadee.vload(extents, DataType::Int(32)));
      }
      uf_node->SetRange(Range::make_by_min_extent(0, max_extent));
    };

    init_uf(rel->fused_to_outer_uf, outer_extent, fused_to_outer);
    init_uf(rel->fused_to_inner_uf, inner_extent, fused_to_inner);
    init_uf(rel->outer_inner_to_fused_uf, fused_extent, outer_inner_to_fused, fused_var_val);

    return body;
  }

  Array<PrimExpr> get_iter_var_values(Array<IterVar> vars, Stage& stage) {
    std::unordered_map<IterVar, PrimExpr> state;
    for (auto rv : stage->op->root_iter_vars()) {
      state[rv] = rv->var;
    }
    PassDownIndex(stage, dom_map, &state, true);
    Array<PrimExpr> results;
    for (auto var : vars) {
      CHECK(state.count(var));
      results.push_back(state.at(var));
    }
    return results;
  }

  Schedule& sch;
  std::unordered_map<IterVar, Range>& dom_map;
  int count;
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

Stmt ScheduleOps(Schedule sch, InferBoundsResult bounds, bool debug_keep_trivial_loop) {
  Map<IterVar, Range> dom_map_ = bounds->bounds;
  Map<Stage, Map<std::string, Range>> env_dom_map_ = bounds->env_bounds;
  Map<Stage, Map<std::string, IterVar>> env_var_map_ = bounds->env_vars;

  std::unordered_map<const VarNode*, std::string> bind_map;

  for (Stage stage : sch->stages) {
    for (auto kv : stage->iter_var_attrs) {
      if (kv.second->bind_thread.defined()) {
        bind_map[kv.first->var.as<VarNode>()] = kv.second->bind_thread->var->name_hint;
      }
    }
  }

  // Generate A functions for all layouts
  AFunGenerator generator(sch);
  Stmt a_fun_stmt = generator.GenerateAndSetAFuns();

  sch.freeze_tensor_dimensions(dom_map_);

  Stmt body = Stmt();
  std::unordered_map<IterVar, Range> dom_map = as_unordered_map(dom_map_);
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
  for (auto stage : sch->stages) {
    if (stage.is_ancestor_attached_at_root()) continue;
    for (size_t i = 0; i < stage->op->num_outputs(); ++i) {
      Modes layout = stage->op->output_layout(i);
      CHECK(!layout.defined() || !layout->is_ragged())
          << "The operation " << stage->op
          << " which is attached at a non-root position has been asked to have a ragged "
             "layout. That is not yet supported ";
    }
  }

  // reverse the post DFS order.
  for (size_t i = sch->stages.size(); i != 0; --i) {
    Stage s = sch->stages[i - 1];

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
      InjectScanStep mu(s, scan_init.at(s->op), dom_map, env_dom_map, env_var_map, bind_map, true,
                        debug_keep_trivial_loop);
      body = mu(std::move(body));
      CHECK(mu.found_attach) << "did not find attachment point for scan.init";
    } else if (attach_spec->attach_type == kSingleKernelScope) {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      CHECK(body.defined());
      InjectSingleKernelInput mu(s, attach_spec->attach_stage->op, dom_map, env_dom_map,
                                 env_var_map, bind_map, true, debug_keep_trivial_loop);
      body = mu(std::move(body));
      CHECK(mu.found_attach) << "did not find attachment point for scan.update";
    } else if (attach_spec->attach_type == kScanUpdate) {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      // Handle scan update
      CHECK(body.defined());
      InjectScanStep mu(s, attach_spec->attach_stage->op, dom_map, env_dom_map, env_var_map,
                        bind_map, false, debug_keep_trivial_loop);
      body = mu(std::move(body));
      CHECK(mu.found_attach) << "did not find attachment point for scan.update";
    } else if (attach_spec->attach_type == kConditionalThen) {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      // Handle scan update
      CHECK(body.defined());
      InjectConditionalStep mu(s, attach_spec->attach_stage->op, dom_map, env_dom_map, env_var_map,
                               bind_map, false, debug_keep_trivial_loop);
      body = mu(std::move(body));
      CHECK(mu.found_attach) << "did not find attachment point for scan.update";
    } else if (attach_spec->attach_type == kConditionalElse) {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      // Handle scan update
      CHECK(body.defined());
      InjectConditionalStep mu(s, attach_spec->attach_stage->op, dom_map, env_dom_map, env_var_map,
                               bind_map, true, debug_keep_trivial_loop);
      body = mu(std::move(body));
      CHECK(mu.found_attach) << "did not find attachment point for scan.update";
    } else if (attach_spec->attach_type == kInlinedAlready) {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      // do nothing
    } else if (attach_spec->attach_type == kGroupRoot) {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      CHECK(!s->group.defined());
      body = MakePipeline(s, dom_map, env_dom_map, env_var_map, bind_map, body,
                          debug_keep_trivial_loop);
    } else {
      // std::cout << "[OPS]  " << __LINE__ << std::endl;
      // CHECK_EQ(attach_spec->attach_type, kScope) << s;
      CHECK(attach_spec->attach_type == kScope || attach_spec->attach_type == kSingleKernelScope)
          << s;
      CHECK(body.defined());
      InjectAttach mutator(s, attach_spec, dom_map, env_dom_map, env_var_map, bind_map,
                           debug_keep_trivial_loop);
      // std::cout << "[BODY] "  << body<< std::endl;
      body = mutator(std::move(body));
      CHECK(mutator.found_attach) << "did not find attachment point for " << s << " in "
                                  << attach_spec->attach_stage->op << " x "
                                  << attach_spec->attach_ivar << ", body:\n"
                                  << body;
    }
    // if (s->op->name == "c_next_h" || s->op->name == "cl_next_h" || s->op->name == "cl_hz_gate") {
    // std::cout << "Body after " << s->op << " " << body << std::endl;
    // }
  }

  body = SeqStmt({a_fun_stmt, body});

  body = SimplifyFusionFunctions(sch)(body);

  std::cout << "Before fusion merge\n" << body << std::endl;

  RaggedFusionBoundStmtsGenerator fusion_generator(sch, dom_map);
  Stmt ret0 = fusion_generator.generate(body);

  sch->InvalidateCache();
  sch->InitCache();
  SchedulePostProc post_proc;
  post_proc.InitToReplaceForEnvelopeOps(sch);
  Stmt ret1 = post_proc(std::move(ret0));
  // std::cout << "Body after postproc1 " << ret1 << std::endl;
  sch->InvalidateCache();
  sch->InitCache();
  post_proc.InitToReplaceOriginOps(sch);
  Stmt ret2 = post_proc(std::move(ret1));
  EnvThreadReplacer env_replace(dom_map_, bind_map);
  Stmt ret3 = env_replace(std::move(ret2));
  // std::cout << "Body after postproc2 " << ret2 << std::endl;
  return UninterpFun::InlineUninterpFunCalls(ret3);
  // return ret3;
}

TVM_REGISTER_GLOBAL("schedule.ScheduleOps").set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args.size() == 2)
    *ret = ScheduleOps(args[0], args[1], false);
  else
    *ret = ScheduleOps(args[0], args[1], args[2]);
});

}  // namespace te
}  // namespace tvm
