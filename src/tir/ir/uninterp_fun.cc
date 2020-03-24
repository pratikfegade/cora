#include <tvm/runtime/registry.h>
#include <tvm/tir/uninterp_fun.h>
#include <tvm/tir/expr_functor.h>

namespace tvm {
  namespace tir {
    UninterpFun UninterpFunNode::make(std::string fname,
				      Range range,
				      Array<Var> parameters,
				      PrimExpr body) {
      ObjectPtr<UninterpFunNode> n = make_object<UninterpFunNode>();
      n->fname = fname;
      n->range = range;
      n->parameters = parameters;
      n->body = body;
      return UninterpFun(n);
    }

    size_t UninterpFunNode::arity() const {
      return this->parameters.size();
    }

    int UninterpFunNode::num_outputs() const {
      return 1;
    }

    class ComplexExprChecker : public ExprVisitor {
    public:
      void VisitExpr_(const CallNode* op) final {
	complex = true;
      }

      bool complex{false};
    };

    bool UninterpFunNode::is_complex() const {
      ComplexExprChecker checker;
      checker(this->body);
      return checker.complex;
    }

    class IndexVariableReplacer: ExprMutator {
      const std::unordered_map<const VarNode*, PrimExpr> replace_map_;

    public:
      explicit IndexVariableReplacer(const std::unordered_map<const VarNode*, PrimExpr> replace_map) : replace_map_(replace_map) {}

      PrimExpr VisitExpr_(const VarNode* op) override {
	if (replace_map_.count(op) > 0) {
	  return replace_map_.at(op);
	}
	else return ExprMutator::VisitExpr_(op);
      }

      PrimExpr Replace(const PrimExpr expr) {
	return VisitExpr(expr);
      }
    };

    const PrimExpr UninterpFunNode::substitute(Array<PrimExpr> arguments) const {
      std::unordered_map<const VarNode*, PrimExpr> replace_map;
      CHECK_EQ(this->parameters.size(), arguments.size());
      for (size_t i = 0; i < arguments.size(); ++i) {
    	const VarNode* var_node = this->parameters[i].get();
    	replace_map[var_node] = arguments[i];
      }
      IndexVariableReplacer ivr(replace_map);
      return ivr.Replace(this->body);
    }

    PrimExpr UninterpFun::InlineUninterpFunCalls(PrimExpr e) {
      class UninterpInliner: ExprMutator {
	PrimExpr VisitExpr_(const CallNode* op) {
	  if (op->func.as<UninterpFunNode>()) {
	    UninterpFun ufun = Downcast<UninterpFun, FunctionRef>(op->func);
	    Array<PrimExpr> arguments;
	    for (auto arg: op->args) {
	      arguments.push_back(this->VisitExpr(arg));
	    }
	    return ufun->substitute(arguments);
	  }
	  else {
	    return ExprMutator::VisitExpr_(op);
	  }
	}

      public:
	PrimExpr Inline(PrimExpr e) {
	  return this->VisitExpr(e);
	}
      };

      UninterpInliner ui;
      return ui.Inline(e);
    }

    Range UninterpFun::InlineUninterpFunCalls(Range r) {
      return Range::make_by_min_extent(UninterpFun::InlineUninterpFunCalls(r->min),
				       UninterpFun::InlineUninterpFunCalls(r->extent));
    }

    TVM_REGISTER_NODE_TYPE(UninterpFunNode);
    TVM_REGISTER_GLOBAL("tir.UninterpFun")
    .set_body_typed([](std::string fname, Range range, Array<Var> parameters, PrimExpr body) {
		      return UninterpFunNode::make(fname, range, parameters, body);
		    });
  }
}
