#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_equality.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/lowered_func.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <unordered_set>

#include "../ir/var_replacer.h"
#include "ir_util.h"

namespace tvm {
namespace tir {
class LoadCollector : public StmtExprVisitor {
 public:
  LoadCollector() { scope_loops_.push_back(nullptr); }

  std::unordered_map<const ForNode*, std::vector<const LoadNode*>> GetHoistableLoads(Stmt stmt) {
    this->VisitStmt(stmt);
    return hoistable_loads_;
  }

 private:
  void VisitStmt_(const ForNode* op) override {
    // if (op->loop_var->name_hint == "iV_23_f.o") {
    //   std::cout << "[HL] Loop previously: " << op->body << std::endl;
    // }

    scope_loops_.push_back(op);
    this->VisitStmt(op->body);
    scope_loops_.pop_back();
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::aux_data_structure) {
      if (auto buf = op->node.as<VarNode>()) {
        hoistable_buffers_.insert(buf);
        this->VisitStmt(op->body);
        hoistable_buffers_.erase(buf);
      } else {
        StmtExprVisitor::VisitStmt_(op);
      }
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const StoreNode* op) override {
    // std::cout << "[HL] Store " << GetRef<Stmt>(op) << std::endl;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const LoadNode* op) override {
    // std::cout << "[HL]   Visiting load " << GetRef<PrimExpr>(op) << " S " << scope_loops_.size()
    // << std::endl;
    if (hoistable_buffers_.count(op->buffer_var.operator->())) {
      // std::cout << "[HL]    Hoistable buf" << std::endl;
      std::unordered_set<const VarNode*> used_vars = VarCollector().collect(op->index);
      int i = scope_loops_.size() - 1;
      for (; i > 0; --i) {
        auto loop_var = scope_loops_[i]->loop_var.operator->();
        if (used_vars.count(loop_var)) {
          break;
        }
      }

      // std::cout << "[HL]     I " << i << std::endl;
      if (i < scope_loops_.size() - 1 || (scope_loops_.size() == 1 && i == 0)) {
        Var loop_var = i == 0 ? NullValue<Var>() : scope_loops_[i]->loop_var;
        // std::cout << "[HL]    Load " << GetRef<PrimExpr>(op) << " in " << scope_loops_.back()
        // << " hoisted to " << loop_var << std::endl;
        hoistable_loads_[scope_loops_[i]].push_back(op);
      }
      this->VisitExpr(op->index);
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  std::vector<const ForNode*> scope_loops_;
  std::unordered_map<const ForNode*, std::vector<const LoadNode*>> hoistable_loads_;
  std::unordered_set<const VarNode*> hoistable_buffers_;
};

class LoadHoister : public StmtExprMutator {
 public:
  LoadHoister(std::unordered_map<const ForNode*, std::vector<const LoadNode*>> hoistable_loads)
      : hoistable_loads_(hoistable_loads) {}

  Stmt HoistLoads(Stmt stmt) {
    // if (hoistable_loads_.count(nullptr)) {
    //   return AddLetsAndVisit(stmt, hoistable_loads_[nullptr]);
    // } else {
    return this->VisitStmt(stmt);
    // }
  }

 private:
  Stmt AddLetsAndVisit(Stmt body, std::vector<const LoadNode*> loads, bool visit) {
    std::vector<Stmt> let_nest;
    Stmt noop = EvaluateNode::make(0);
    std::unordered_map<PrimExpr, Var, ExprHash, ExprEquality> added_loads;
    for (auto load_node : loads) {
      PrimExpr load = GetRef<PrimExpr>(load_node);
      if (added_loads.count(load)) {
        continue;
        load_vars_[load] = added_loads[load];
      }
      Var load_var = Var("var" + std::to_string(count_++), load.dtype());
      load_vars_[load] = load_var;
      let_nest.push_back(LetStmtNode::make(load_var, load, noop));
      added_loads[load] = load_var;
    }
    return MergeNest(let_nest, visit ? this->VisitStmt(body) : body);
  }

  PrimExpr VisitExpr_(const LoadNode* op) override {
    PrimExpr load = GetRef<PrimExpr>(op);
    if (load_vars_.count(load)) return load_vars_[load];
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const ForNode* op) override {
    Stmt ret;
    std::vector<const LoadNode*> outermost_loads;
    if (!outermost_done_) {
      if (hoistable_loads_.count(nullptr)) {
        outermost_loads = hoistable_loads_[nullptr];
      }
      outermost_done_ = true;
    }
    Stmt stmt;
    if (hoistable_loads_.count(op)) {
      Stmt body = AddLetsAndVisit(op->body, hoistable_loads_[op], true);
      stmt = ForNode::make(op->loop_var, this->VisitExpr(op->min), this->VisitExpr(op->extent),
                           op->for_type, op->device_api, body, op->hfuse_group_id);
    } else {
      stmt = StmtExprMutator::VisitStmt_(op);
    }
    if (outermost_loads.size() > 0) {
      ret = AddLetsAndVisit(stmt, outermost_loads, true);
    } else {
      ret = stmt;
    }

    // if (op->loop_var->name_hint == "iV_23_f.o") {
    // std::cout << "[HL] Loop after " << ret << std::endl;
    // }
    return ret;
  }

  std::unordered_map<const ForNode*, std::vector<const LoadNode*>> hoistable_loads_;
  std::unordered_map<PrimExpr, Var, ExprHash, ExprEquality> load_vars_;
  bool outermost_done_{false};
  int count_{0};
};

LoweredFunc HoistLoads(LoweredFunc f) {
  // std::cout << "[HL] Hoisting loads" << std::endl;
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  LoadCollector load_collector;
  auto hoistable_loads = load_collector.GetHoistableLoads(f->body);
  n->body = LoadHoister(hoistable_loads).HoistLoads(f->body);
  return LoweredFunc(n);
}
}  // namespace tir
}  // namespace tvm
#undef COUT
