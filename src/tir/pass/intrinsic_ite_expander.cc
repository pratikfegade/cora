#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include "../../arith/interval_set.h"
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace tir {
  class InlineIfThenElseExpander {
    void CollectIfElseExprs()  {
      class StoreValueVisitor : public ExprVisitor {
    	void VisitExpr_(const CallNode* op) override {
    	  if (op->call_type == CallNode::CallType::PureIntrinsic &&
    	      op->name == tir::intrinsic::tvm_if_then_else) {
    	    (*p_if_else_exprs)[outer_store].push_back(op);
    	  }
    	}
      public:
    	StoreValueVisitor(const StoreNode* outer_store_,
    			  std::unordered_map<const StoreNode*, std::vector<const CallNode*>>* p_if_else_exprs_) :
    	  outer_store(outer_store_), p_if_else_exprs(p_if_else_exprs_) {}
    	const StoreNode* outer_store;
    	std::unordered_map<const StoreNode*, std::vector<const CallNode*>>* p_if_else_exprs;
      };

      class StoreVisitor : public StmtVisitor {
    	void VisitStmt_(const StoreNode* op) override {
    	  StoreValueVisitor value_visitor(op, this->p_if_else_exprs);
    	  value_visitor(op->value);
    	  StmtVisitor::VisitStmt_(op);
    	}

    	std::unordered_map<const StoreNode*, std::vector<const CallNode*>>* p_if_else_exprs;

      public:
    	StoreVisitor(std::unordered_map<const StoreNode*, std::vector<const CallNode*>>* p_if_else_exprs_) :
    	  p_if_else_exprs(p_if_else_exprs_) {}
      };

      StoreVisitor visitor(&(this->if_else_exprs));
      visitor(this->stmt);
    }

    Stmt ExpandITEInOneStoreNode(const StoreNode* store_node) {
      class ITEBranchExprReplacer: public StmtExprMutator {
    	PrimExpr VisitExpr_(const CallNode* op) override {
    	  if (to_replace == op) {
    	    CHECK(op->name == tir::intrinsic::tvm_if_then_else);
    	    if (if_branch) {
    	      return op->args[1];
    	    }
    	    else {
    	      return op->args[2];
    	    }
    	  }
    	  else {
    	    return ExprMutator::VisitExpr_(op);
    	  }
    	}

    	const CallNode* to_replace;
    	const bool if_branch;
      public:
    	ITEBranchExprReplacer(const CallNode* to_replace_, const bool if_branch_) :
    	  to_replace(to_replace_), if_branch(if_branch_) {}
      };

      Stmt stmt = GetRef<Stmt>(store_node);

      for (auto ite: this->if_else_exprs[store_node]) {
	// std::cout << "Stmt: " << stmt;
	ITEBranchExprReplacer if_branch_replacer(ite, true);
	ITEBranchExprReplacer else_branch_replacer(ite, false);
	Stmt new_stmt = IfThenElseNode::make(ite->args[0],
					     if_branch_replacer(stmt),
					     else_branch_replacer(stmt));
	stmt = new_stmt;
      }
      // std::cout << "Stmt: " << stmt;
      return stmt;
    }

    Stmt ExpandIfThenElseExprInner() {
      class ITEExpander : public StmtMutator {
	Stmt VisitStmt_(const StoreNode* op) override {
	  if (outer_expander->if_else_exprs.find(op) != outer_expander->if_else_exprs.end()) {
	    return outer_expander->ExpandITEInOneStoreNode(op);
	  }
	  else {
	    return StmtMutator::VisitStmt_(op);
	  }
	}

	InlineIfThenElseExpander* outer_expander;
      public:
	ITEExpander(InlineIfThenElseExpander* outer_expander_) :
	  outer_expander(outer_expander_) {}
      };

      ITEExpander expander(this);
      return expander(this->stmt);
    }

  public:
    InlineIfThenElseExpander(Stmt stmt_) : stmt(stmt_) {}

    Stmt ExpandIfThenElseExpr() {
      CollectIfElseExprs();
      return ExpandIfThenElseExprInner();
    }

    Stmt stmt;
    std::unordered_map<const StoreNode*, std::vector<const CallNode*>> if_else_exprs;
  };

  Stmt ExpandIntrinsicITE(Stmt stmt) {
    // std::cout << "Better hoisting ifelse" << std::endl;
    return ConvertSSA(InlineIfThenElseExpander(stmt).ExpandIfThenElseExpr());
  }
}
}
