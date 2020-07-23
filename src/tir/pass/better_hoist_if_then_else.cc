#include <tvm/arith/analyzer.h>
#include <tvm/arith/z3_analyzer.h>
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_equality.h>
#include <tvm/tir/lowered_func.h>
#include <tvm/tir/stmt_functor.h>

#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "../../arith/interval_set.h"
#include "../../runtime/thread_storage_scope.h"

#define COUT std::cout << "[RIfR] "
namespace tvm {
namespace tir {
class ConsecutiveIfFuser : public StmtMutator {
  Stmt VisitStmt_(const SeqStmtNode* op) override {
    Array<Stmt> seq = op->seq;

    std::vector<size_t> run_starts;
    std::vector<size_t> run_ends;
    FindConsecutiveFusableIfRuns(seq, run_starts, run_ends);
    CHECK_EQ(run_starts.size(), run_ends.size());

    Array<Stmt> new_seq;

    size_t pos = 0;
    size_t next_run_pos = 0;
    while (pos < seq.size()) {
      if (next_run_pos < run_starts.size() && pos == run_starts[next_run_pos]) {
        // A new run starts here. Fuse
        size_t start = pos;
        size_t end = run_ends[next_run_pos];

        Array<Stmt> fusables_visited;
        for (size_t i = start; i < end; ++i) {
          fusables_visited.push_back(StmtMutator::VisitStmt(seq[i]));
        }

        Stmt fused_if = FuseFusableIfs(fusables_visited);
        new_seq.push_back(fused_if);

        pos = end;
        next_run_pos++;
      } else {
        new_seq.push_back(StmtMutator::VisitStmt(seq[pos]));
        pos++;
      }
    }
    if (new_seq.size() == 1) {
      return new_seq[0];
    } else {
      return SeqStmt(new_seq);
    }
  }

  Stmt FuseFusableIfs(Array<Stmt> seq) {
    PrimExpr condition = seq[0].as<IfThenElseNode>()->condition;
    Array<Stmt> bodies;
    for (size_t i = 0; i < seq.size(); ++i) {
      auto if_node = seq[i].as<IfThenElseNode>();
      bodies.push_back(if_node->then_case);
    }

    return IfThenElseNode::make(condition, SeqStmt(bodies));
  }

  void FindConsecutiveFusableIfRuns(const Array<Stmt>& seq, std::vector<size_t>& run_starts,
                                    std::vector<size_t>& run_ends) {
    // std::cout << "[FIF] Find fusable run " << std::endl;
    size_t current_run_start = 0;
    if (seq.size() <= 1) return;
    Stmt previous = seq[0];
    size_t i;
    for (i = 1; i < seq.size(); ++i) {
      Stmt current = seq[i];
      if (FusableIfs(previous, current)) {
        // std::cout << "[FIF]  Equal" << std::endl;
      } else {
        /* A run ends here */
        if (current_run_start == i - 1) { /* Single runs are trivial */
        } else {
          // std::cout << "[FIF]  Found run " << current_run_start << " " << i << std::endl;
          run_starts.push_back(current_run_start);
          run_ends.push_back(i);
        }
        current_run_start = i;
      }
      previous = current;
    }
    if (current_run_start == i - 1) { /* Single runs are trivial */
    } else {
      // std::cout << "[FIF]  Found run " << current_run_start << " " << i << std::endl;
      run_starts.push_back(current_run_start);
      run_ends.push_back(i);
    }
  }

  bool FusableIfs(Stmt s1, Stmt s2) {
    auto if1 = s1.as<IfThenElseNode>();
    auto if2 = s2.as<IfThenElseNode>();
    if (if1 && if2) {
      if (!if1->else_case.defined() && !if2->else_case.defined()) {
        tvm::tir::ExprEquality equals_checker;
        // std::cout << "[FIF]   Checking " << std::endl;
        // std::cout << "[FIF]      " << if1->condition << std::endl;
        // std::cout << "[FIF]      " << if2->condition << std::endl;
        bool ret = equals_checker.VisitExpr(if1->condition, if2->condition);
        // std::cout << "[FIF]      Result: " << ret << std::endl;
        return ret;
      }
      return false;
    }
    return false;
  }
};

class ProducerConsumerNodesRemover : public StmtMutator {
  Stmt VisitStmt_(const ProducerConsumerNode* op) override {
    return StmtMutator::VisitStmt(op->body);
  }
  Stmt VisitStmt_(const SeqStmtNode* op) override {
    Array<Stmt> new_seq;
    for (auto stmt : op->seq) {
      new_seq.push_back(StmtMutator::VisitStmt(stmt));
    }
    return SeqStmt::Flatten(new_seq);
  }
};

class DuplicateNestedIfsRemover : public StmtMutator {
  Stmt VisitStmt_(const IfThenElseNode* op) override {
    if (!op->else_case.defined()) {
      auto nested_if = op->then_case.as<IfThenElseNode>();
      if (nested_if && !nested_if->else_case.defined()) {
        if (ExprEquality().VisitExpr(op->condition, nested_if->condition)) {
          Stmt body = StmtMutator::VisitStmt(nested_if->then_case);
          return IfThenElseNode::make(op->condition, body);
        }
      }
    }
    return StmtMutator::VisitStmt_(op);
  }
};

class IfHoister : public StmtMutator {
  Stmt VisitStmt_(const ForNode* op) override {
    // std::cout << "[HOIST]   " << op->loop_var << std::endl;
    if (auto ite = op->body.as<IfThenElseNode>()) {
      if (Hoistable(op, ite)) {
        if (ite->else_case.defined()) {
          Stmt for_body = StmtMutator::VisitStmt(ite->then_case);
          Stmt then_loop = ForNode::make(op->loop_var, op->min, op->extent, op->for_type,
                                         op->device_api, for_body);
          Stmt else_loop = ForNode::make(op->loop_var, op->min, op->extent, op->for_type,
                                         op->device_api, for_body);
          Stmt if_stmt = IfThenElseNode::make(ite->condition, then_loop, else_loop);
          return if_stmt;
        } else {
          Stmt for_body = StmtMutator::VisitStmt(ite->then_case);
          Stmt for_loop = ForNode::make(op->loop_var, op->min, op->extent, op->for_type,
                                        op->device_api, for_body);
          Stmt if_stmt = IfThenElseNode::make(ite->condition, for_loop);
          return if_stmt;
        }
      }
    } else if (auto seq = op->body.as<SeqStmtNode>()) {
      // std::cout << "[HOIST]     Seq " << seq->size() << std::endl;
    }
    return StmtMutator::VisitStmt_(op);
  }

  bool Hoistable(const ForNode* for_loop, const IfThenElseNode* if_stmt) {
    auto stored_vars = GetAllStoredVars(GetRef<Stmt>(for_loop));
    bool ret = !ReadsVariablesFromSet(if_stmt->condition, stored_vars);
    // std::cout << "[HOIST]   " << if_stmt->condition << " " << ret << std::endl;
    return ret;
  }

  std::unordered_set<const VarNode*> GetAllStoredVars(Stmt stmt) {
    class StoredVarFinder : public StmtVisitor {
      void VisitStmt_(const LetStmtNode* op) override { (*p_set).insert(op->var.get()); }
      void VisitStmt_(const ForNode* op) override { (*p_set).insert(op->loop_var.get()); }
      void VisitStmt_(const AllocateNode* op) override { (*p_set).insert(op->buffer_var.get()); }
      void VisitStmt_(const StoreNode* op) override { (*p_set).insert(op->buffer_var.get()); }

      std::unordered_set<const VarNode*>* p_set;

     public:
      StoredVarFinder(std::unordered_set<const VarNode*>* p_set_) : p_set(p_set_) {}
    };

    std::unordered_set<const VarNode*> ret;
    StoredVarFinder finder(&ret);
    finder(stmt);
    return ret;
  }

  bool ReadsVariablesFromSet(PrimExpr expr, std::unordered_set<const VarNode*> set) {
    class AccessedVariablesChecker : public ExprVisitor {
      void VisitExpr_(const VarNode* op) override {
        auto set = *p_set;
        if (set.find(op) != set.end()) {
          this->found = true;
        }
      }

      std::unordered_set<const VarNode*>* p_set;

     public:
      AccessedVariablesChecker(std::unordered_set<const VarNode*>* p_set_)
          : p_set(p_set_), found(false) {}
      bool found;
    };

    AccessedVariablesChecker checker(&set);
    checker(expr);
    return checker.found;
  }
};

class RedundantIfRemover : public StmtExprMutator {
  Stmt VisitStmt_(const ForNode* op) override {
    setConstraint(op->loop_var, Range::make_by_min_extent(op->min, op->extent));
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    removeConstraint(op->loop_var);
    return ret;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      Var var = Downcast<IterVar>(op->node)->var;
      Range range = Range::make_by_min_extent(0, op->value);
      setConstraint(var, range);
      Stmt ret = StmtExprMutator::VisitStmt_(op);
      removeConstraint(var);
      return ret;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const IfThenElseNode* op) override {
    arith::Analyzer analyzer;

    for (const auto& expr : input_constraints) {
      CHECK(expr.dtype().is_bool());
      analyzer.AddConstraint(expr);
    }
    for (const auto& it : constraints) {
      analyzer.Bind(GetRef<Var>(it.first), it.second);
    }

    bool redundant = analyzer.CanProve(op->condition);
    if (redundant) {
      // std::cout << "[RIF] Redundant " << op->condition << std::endl;
      return StmtExprMutator::VisitStmt(op->then_case);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const SelectNode* op) override {
    arith::Analyzer analyzer;

    for (const auto& expr : input_constraints) {
      CHECK(expr.dtype().is_bool());
      analyzer.AddConstraint(expr);
    }
    for (const auto& it : constraints) {
      analyzer.Bind(GetRef<Var>(it.first), it.second);
    }

    bool redundant = analyzer.CanProve(op->condition);
    if (redundant) {
      // std::cout << "[RIF] Redundant " << op->condition << std::endl;
      return StmtExprMutator::VisitExpr(op->true_value);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

 public:
  RedundantIfRemover(const Array<PrimExpr>& input_constraints_)
      : input_constraints(input_constraints_) {}

 private:
  const Array<PrimExpr>& input_constraints;
  std::unordered_map<const VarNode*, Range> constraints;

  void setConstraint(const Var& var, const Range& range) {
    // std::cout << "[RIF] Binding " << var << " " << range << std::endl;
    constraints[var.as<VarNode>()] = range;
  }

  void removeConstraint(const Var& var) {
    // std::cout << "[RIF] Unbinding " << var << std::endl;
    constraints.erase(var.as<VarNode>());
  }
};

Stmt BetterHoistIfThenElseStmt(Stmt stmt, std::string target, Array<PrimExpr> constraints) {
  // std::cout << "[STMT] Hoisting" << std::endl;
  // if (target != "cuda") return stmt;
  stmt = ProducerConsumerNodesRemover()(stmt);
  for (int i = 0; i < 5; ++i) {
    // std::cout << "[STMT0] " << stmt << std::endl;
    stmt = DuplicateNestedIfsRemover()(stmt);
    // std::cout << "[STMT1] " << stmt << std::endl;
    stmt = ConsecutiveIfFuser()(stmt);
    // std::cout << "[STMT2] " << stmt << std::endl;
    stmt = IfHoister()(stmt);
    // std::cout << "[STMT3] " << stmt << std::endl;
    stmt = RedundantIfRemover(constraints)(stmt);
    // std::cout << "[STMT4] " << stmt << std::endl;
  }
  return stmt;
}

LoweredFunc BetterHoistIfThenElse(LoweredFunc f, std::string target, Array<PrimExpr> constraints) {
  // if (target != "cuda") return f;
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  n->body = BetterHoistIfThenElseStmt(f->body, target, constraints);
  return LoweredFunc(n);
}

LoweredFunc RemoveRedundantIfsFromFunc(LoweredFunc f, std::string target,
                                       Array<PrimExpr> constraints) {
  // if (target != "cuda") return f;
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  Stmt body = f->body;
  body = RedundantIfRemover(constraints)(body);
  n->body = body;
  return LoweredFunc(n);
}

Stmt RemoveRedundantIfs(Stmt stmt, Array<PrimExpr> constraints) {
  return RedundantIfRemover(constraints)(stmt);
}
}  // namespace tir
}  // namespace tvm
#undef COUT
