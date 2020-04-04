#include <tvm/runtime/registry.h>
#include <tvm/tir/uninterp_fun.h>
#include <tvm/te/dimension.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/expr_equality.h>
#include "var_replacer.h"
#include <vector>

namespace tvm {
  namespace tir {
    UninterpFun UninterpFunNode::make(std::string fname,
				      Range range,
				      Array<Var> parameters,
				      PrimExpr body) {
      Array<tvm::te::Dimension> no_dimensions;
      for (size_t i = 0; i < parameters.size(); ++i) {
	no_dimensions.push_back(tvm::te::Dimension::NoDimension);
      }
      return UninterpFunNode::make(fname, range, no_dimensions, parameters, body);
    }

    UninterpFun UninterpFunNode::make(std::string fname,
				      Range range,
				      Array<tvm::te::Dimension> dimensions,
				      Array<Var> parameters,
				      PrimExpr body) {
      ObjectPtr<UninterpFunNode> n = make_object<UninterpFunNode>();
      n->fname = fname;
      n->range = range;
      n->dimensions = dimensions;
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

    const PrimExpr UninterpFunNode::substitute(Array<PrimExpr> arguments) const {
      std::unordered_map<const VarNode*, PrimExpr> replace_map;
      CHECK_EQ(this->parameters.size(), arguments.size());
      for (size_t i = 0; i < arguments.size(); ++i) {
    	const VarNode* var_node = this->parameters[i].get();
    	replace_map[var_node] = arguments[i];
      }
      return VarReplacer(replace_map)(this->body);
    }

    const PrimExpr UninterpFunNode::substitute(Array<PrimExpr> args, Array<tvm::te::Dimension> arg_dims) const {
      if (args.size() != arg_dims.size()) {
	std::cout << "Really?" << std::endl;
      }
      // CHECK_EQ(args.size(), arg_dims.size());
      Map<tvm::te::Dimension, PrimExpr> arg_dim_map;
      for (size_t i = 0; i < args.size(); ++i) {
	arg_dim_map.Set(arg_dims[i], args[i]);
      }

      std::unordered_map<const VarNode*, PrimExpr> replace_map;
      for (size_t i = 0; i < this->parameters.size(); ++i) {
    	auto param = this->parameters[i].get();
	auto param_dim = this->dimensions[i];
	CHECK(arg_dim_map.count(param_dim) > 0) << param_dim->name;
    	replace_map[param] = arg_dim_map.at(param_dim);
      }
      return VarReplacer(replace_map)(this->body);
    }

    int UninterpFunNode::GetArgPos(Var var) const {
      size_t i = 0;
      for (; i < this->parameters.size(); ++i) {
	if (var.same_as(this->parameters[i])) return i;
      }
      return i;
    }

    UninterpFun UninterpFunNode::AddDummyArgument(size_t pos) const {
      Array<Var> parameters;
      for (size_t j = 0; j < this->arity(); ++j) {
	if (pos == j) {
	  parameters.push_back(Var("ufp_f" + std::to_string(j), DataType::Int(32)));
	}
	parameters.push_back(this->parameters[j]);
      }
      if (pos == this->arity()) {
	parameters.push_back(Var("ufp_f" + std::to_string(pos), DataType::Int(32)));
      }
      return UninterpFunNode::make(this->fname, this->range, parameters, this->body);
    }

    UninterpFun UninterpFunNode::FunWithNewParams(Array<PrimExpr> param_exprs, Array<Var> new_params) const {
      CHECK_EQ(this->parameters.size(), param_exprs.size());
      std::unordered_map<const VarNode*, PrimExpr> replace_map;
      for (size_t i = 0; i < this->parameters.size(); ++i) {
	replace_map[this->parameters[i].as<VarNode>()] = param_exprs[i];
      }

      PrimExpr new_body = VarReplacer(replace_map)(this->body);
      return UninterpFunNode::make(this->fname, this->range, new_params, new_body);
    }

    PrimExpr UninterpFun::InlineUninterpFunCalls(PrimExpr e) {
      class UninterpInliner: ExprMutator {
	PrimExpr VisitExpr_(const CallNode* op) {
	  if (op->func.as<UninterpFunNode>()) {
	    CHECK(op->argument_dimensions.defined());
	    UninterpFun ufun = Downcast<UninterpFun, FunctionRef>(op->func);
	    Array<PrimExpr> arguments;
	    for (auto arg: op->args) {
	      arguments.push_back(this->VisitExpr(arg));
	    }
	    return ufun->substitute(arguments, op->argument_dimensions);
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

    ArgMappingAndEquality UninterpFun::CheckEquality(UninterpFun f1, UninterpFun f2) {
      PrimExpr e1 = f1->body;
      PrimExpr e2 = f2->body;

      // std::cout << "Comparing " << e1 << " " << e2 << std::endl;

      class VarCollector: public ExprVisitor {
	void VisitExpr_(const VarNode* op) {
	  variables.push_back(GetRef<Var>(op));
	}
      public:
	Array<Var> variables;
      };

      class VarReplacer: public ExprMutator {
	PrimExpr VisitExpr_(const VarNode* op) {
	  if (current_index == replacement.size()) {
	    this->overrun = true;
	    return ExprMutator::VisitExpr_(op);
	  }

	  Var replacement_var =  replacement[current_index++];
	  replace_map.Set(GetRef<Var>(op), replacement_var);
	  // std::cout << "Mapping " << GetRef<Var>(op) << " to " << replacement_var << std::endl;
	  return replacement_var;
	}

	Array<Var> replacement;
	size_t current_index;

      public:
	bool overrun;
	Map<Var, Var> replace_map;
	VarReplacer(Array<Var> replacement_) : replacement(replacement_), current_index(0), overrun(false) {}
      };

      VarCollector collector;
      collector(e1);
      VarReplacer replacer(collector.variables);
      PrimExpr replaced_e2 = replacer(e2);

      if (replacer.overrun) {
	return { false, replacer.replace_map };
      }

      // std::cout << "Replaced " << e2 << " to " << replaced_e2 << std::endl;

      bool ret = tir::ExprEquality().VisitExpr(replaced_e2, e1);
      // std::cout << "Returned " << ret << std::endl;
      return { ret, replacer.replace_map };
    }

    TVM_REGISTER_NODE_TYPE(UninterpFunNode);
    TVM_REGISTER_GLOBAL("tir.UninterpFun")
    .set_body_typed([](std::string fname, Range range, Array<Var> parameters, Array<te::Dimension> dims, PrimExpr body) {
        return UninterpFunNode::make(fname, range, dims, parameters, body);
      });
  }
}
