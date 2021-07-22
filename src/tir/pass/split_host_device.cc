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
 * \file split_host_device.cc
 * \brief Split device function from host.
 */
#include <tvm/runtime/module.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/lowered_func.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>

namespace tvm {
namespace tir {

// use/def analysis, also delete unreferenced lets
class IRUseDefAnalysis : public StmtExprMutator {
 public:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      CHECK_NE(iv->thread_tag.length(), 0U);
      // thread_extent can appear multiple times
      // use the first appearance as def.
      if (!use_count_.count(iv->var.get())) {
        this->HandleDef(iv->var.get());
        thread_axis_.push_back(iv);
        thread_extent_.push_back(op->value);
      }

      PrimExpr value = op->value;
      if (visit_thread_extent_) {
        value = this->VisitExpr(value);
      }
      Stmt body = this->VisitStmt(op->body);
      if (value.same_as(op->value) && body.same_as(op->body)) {
        return GetRef<Stmt>(op);
      }
      return AttrStmtNode::make(op->node, op->attr_key, value, body);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    this->HandleDef(op->var.get());
    Stmt body = this->VisitStmt(op->body);
    // eliminate unreferenced let
    if (use_count_.at(op->var.get()) == 0 && !HasSideEffect(op->value)) {
      return body;
    } else {
      PrimExpr value = this->VisitExpr(op->value);
      if (body.same_as(op->body) && value.same_as(op->value)) {
        return GetRef<Stmt>(op);
      } else {
        return LetStmtNode::make(op->var, value, body);
      }
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    this->HandleDef(op->loop_var.get());
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    // std::cout << "[API] Allocating " << op->buffer_var << " " << op->buffer_var.get() <<
    // std::endl;
    this->HandleDef(op->buffer_var.get());
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    this->HandleUse(op->buffer_var);
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const LetNode* op) final {
    this->HandleDef(op->var.get());
    PrimExpr body = this->VisitExpr(op->body);
    // eliminate unreferenced let
    if (use_count_.at(op->var.get()) == 0 && !HasSideEffect(op->value)) {
      return body;
    } else {
      PrimExpr value = this->VisitExpr(op->value);
      if (body.same_as(op->body) && value.same_as(op->value)) {
        return GetRef<PrimExpr>(op);
      } else {
        return LetNode::make(op->var, value, body);
      }
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    this->HandleUse(GetRef<PrimExpr>(op));
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    this->HandleUse(op->buffer_var);
    return StmtExprMutator::VisitExpr_(op);
  }

  void HandleDef(const VarNode* v) {
    CHECK(!def_count_.count(v)) << "variable " << v->name_hint
                                << " has already been defined, the Stmt is not SSA";
    CHECK(!use_count_.count(v)) << "variable " << v->name_hint
                                << " has been used before definition!";
    // std::cout << "[API] Defining " << v->name_hint << " " << v << std::endl;
    use_count_[v] = 0;
    def_count_[v] = 1;
  }

  void HandleUse(const PrimExpr& v) {
    CHECK(v.as<VarNode>());
    Var var = Downcast<Var>(v);
    auto it = use_count_.find(var.get());
    if (it != use_count_.end()) {
      if (it->second >= 0) {
        ++it->second;
      }
    } else {
      // std::cout << "[API] Undefined " << v << " " << v.get() << std::endl;
      undefined_.push_back(var);
      use_count_[var.get()] = -1;
    }
  }

  // The fields are publically readible to
  // be accessible to the users.
  bool visit_thread_extent_{true};
  Array<Var> undefined_;
  Array<IterVar> thread_axis_;
  Array<PrimExpr> thread_extent_;
  std::unordered_map<const VarNode*, int> use_count_;
  std::unordered_map<const VarNode*, int> def_count_;
};

class HostDeviceSplitter : public StmtMutator {
 public:
  Stmt VisitStmt_(const AllocateNode* op) final {
    handle_data_type_[op->buffer_var.get()] = make_const(op->dtype, 0);
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::pipeline_exec_scope ||
        op->attr_key == attr::device_scope) {
      Stmt body = op->body;
      for (auto it = aux_data_structures_.rbegin(); it != aux_data_structures_.rend(); ++it) {
        body = AttrStmtNode::make(*it, attr::aux_data_structure, 0, body);
      }
      return SplitDeviceFunc(AttrStmtNode::make(op->node, op->attr_key, op->value, body));
    } else if (op->attr_key == attr::aux_data_structure) {
      aux_data_structures_.push_back(op->node);
      Stmt ret = StmtMutator::VisitStmt_(op);
      aux_data_structures_.pop_back();
      return ret;
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }

  Array<LoweredFunc> Split(LoweredFunc f) {
    CHECK_EQ(f->func_type, kMixedFunc);
    for (auto kv : f->handle_data_type) {
      handle_data_type_[kv.first.get()] = kv.second;
    }
    name_ = f->name;
    ObjectPtr<LoweredFuncNode> n = make_object<LoweredFuncNode>(*f.operator->());
    n->body = operator()(f->body);
    n->func_type = kHostFunc;
    Array<LoweredFunc> ret{LoweredFunc(n)};
    for (LoweredFunc x : device_funcs_) {
      ret.push_back(x);
    }
    return ret;
  }

 private:
  Stmt SplitDeviceFunc(Stmt body) {
    std::ostringstream os;
    os << name_ << "_kernel" << device_funcs_.size();
    ObjectPtr<LoweredFuncNode> n = make_object<LoweredFuncNode>();
    // isolate the device function.
    IRUseDefAnalysis m;
    m.visit_thread_extent_ = false;
    n->body = m(std::move(body));
    n->name = os.str();
    n->func_type = kDeviceFunc;
    n->thread_axis = m.thread_axis_;
    // Strictly order the arguments: Var pointers, positional arguments.
    for (Var v : m.undefined_) {
      if (v.dtype().is_handle()) {
        n->args.push_back(v);
        // mark handle data type.
        auto it = handle_data_type_.find(v.get());
        if (it != handle_data_type_.end()) {
          n->handle_data_type.Set(v, it->second);
        }
      }
    }
    for (Var v : m.undefined_) {
      if (!v.dtype().is_handle()) {
        n->args.push_back(v);
      }
    }
    LoweredFunc f_device(n);
    Array<PrimExpr> call_args;
    call_args.push_back(StringImmNode::make(f_device->name));
    for (Var arg : n->args) {
      call_args.push_back(arg);
    }
    for (PrimExpr ext : m.thread_extent_) {
      call_args.push_back(ext);
    }
    device_funcs_.emplace_back(f_device);
    return EvaluateNode::make(CallNode::make(DataType::Int(32), intrinsic::tvm_call_packed,
                                             call_args, CallNode::Intrinsic));
  }

  // function name
  std::string name_;
  // the device functions
  std::vector<LoweredFunc> device_funcs_;
  std::vector<ObjectRef> aux_data_structures_;
  std::unordered_map<const VarNode*, PrimExpr> handle_data_type_;
};

Array<Var> UndefinedVars(const Stmt& stmt, const Array<Var>& args) {
  IRUseDefAnalysis m;
  for (Var arg : args) {
    m.use_count_[arg.get()] = 0;
  }
  m(stmt);
  return m.undefined_;
}

class ReplaceRemainingAuxBuffers : public StmtExprMutator {
 public:
  Array<LoweredFunc> Replace(Array<LoweredFunc> funcs) {
    Array<LoweredFunc> new_funcs;
    LoweredFunc host_func = funcs[0];
    in_host_code = true;
    new_funcs.push_back(Replace(host_func));
    in_host_code = false;
    for (size_t i = 1; i < funcs.size(); ++i) {
      new_funcs.push_back(Replace(funcs[i]));
    }
    return new_funcs;
  }

  LoweredFunc Replace(LoweredFunc f) {
    auto n = make_object<LoweredFuncNode>(*f.operator->());
    n->body = this->VisitStmt(f->body);
    return LoweredFunc(n);
  }

 private:
  Stmt VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == attr::prep_code_scope) {
      CHECK(in_host_code);
      Map<Buffer, Buffer> buf_mapping = Downcast<Map<Buffer, Buffer>>(op->node);
      for (auto it : buf_mapping) {
        host_to_dev[it.first->data.operator->()] = it.second->data;
        dev_to_host[it.second->data.operator->()] = it.first->data;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const LoadNode* op) override {
    return LoadNode::make(op->dtype, ReplaceBufferVar(op->buffer_var), this->VisitExpr(op->index),
                          this->VisitExpr(op->predicate), op->sync_type);
  }

  Stmt VisitStmt_(const StoreNode* op) override {
    return StoreNode::make(ReplaceBufferVar(op->buffer_var), this->VisitExpr(op->value),
                           this->VisitExpr(op->index), this->VisitExpr(op->predicate),
                           op->sync_type);
  }

  Var ReplaceBufferVar(Var var) {
    auto op = var.operator->();
    if (in_host_code && dev_to_host.count(op)) {
      return dev_to_host.at(op);
    } else if (!in_host_code && host_to_dev.count(op)) {
      return host_to_dev.at(op);
    }
    return var;
  }

  std::unordered_map<const VarNode*, Var> host_to_dev;
  std::unordered_map<const VarNode*, Var> dev_to_host;

  bool in_host_code{true};
};

Array<LoweredFunc> SplitHostDevice(LoweredFunc func, std::string grid_sync_str) {
  Array<LoweredFunc> ret = HostDeviceSplitter().Split(func);
  // std::cout << "[SYNC] Sync str " << grid_sync_str << std::endl;
  if (grid_sync_str.size() > 0) {
    for (size_t i = 1; i < ret.size(); ++i) {
      if (grid_sync_str[i - 1] == '1') {
        auto* op = const_cast<LoweredFuncNode*>(static_cast<const LoweredFuncNode*>(ret[i].get()));
        op->grid_sync_type = kCoopGroup;
        // std::cout << "[SYNC] SplitHost Setting coop sync " << ret[i] << std::endl;
      } else {
        auto* op = const_cast<LoweredFuncNode*>(static_cast<const LoweredFuncNode*>(ret[i].get()));
        op->grid_sync_type = kTVM;
        // std::cout << "[SYNC] SplitHost Setting tvm sync " << ret[i] << std::endl;
      }
    }
  }
  return ReplaceRemainingAuxBuffers().Replace(ret);
}

}  // namespace tir
}  // namespace tvm
