#include <tvm/runtime/registry.h>
#include <tvm/te/dimension.h>
#include <tvm/tir/expr_equality.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/uninterp_fun.h>

#include <vector>

#include "var_replacer.h"

namespace tvm {
namespace tir {
UninterpFun UninterpFunNode::make(std::string fname, Range range, Array<Var> parameters,
                                  PrimExpr body) {
  std::cout << "Y u using dis?" << std::endl;
  CHECK(false);
  Array<tvm::te::Dimension> no_dimensions;
  for (size_t i = 0; i < parameters.size(); ++i) {
    no_dimensions.push_back(tvm::te::Dimension::NoDimension);
  }
  return UninterpFunNode::make(fname, range, no_dimensions, parameters, body);
}

UninterpFun UninterpFunNode::make(std::string fname, Range range,
                                  Array<tvm::te::Dimension> dimensions, Array<Var> parameters,
                                  PrimExpr body) {
  CHECK_EQ(parameters.size(), dimensions.size());
  ObjectPtr<UninterpFunNode> n = make_object<UninterpFunNode>();
  n->fname = fname;
  n->range = range;
  n->dimensions = dimensions;
  n->parameters = parameters;
  n->body = body;
  return UninterpFun(n);
}

size_t UninterpFunNode::arity() const { return this->parameters.size(); }

int UninterpFunNode::num_outputs() const { return 1; }

class ComplexExprChecker : public ExprVisitor {
 public:
  void VisitExpr_(const CallNode* op) final { complex = true; }

  bool complex{false};
};

bool UninterpFunNode::is_complex() const {
  ComplexExprChecker checker;
  checker(this->body);
  return checker.complex;
}

const PrimExpr UninterpFunNode::substitute(Array<PrimExpr> args,
                                           Array<tvm::te::Dimension> arg_dims) const {
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

void UninterpFunNode::SetBody(PrimExpr expr) { this->body = expr; }

UninterpFun UninterpFunNode::FunWithNewParams(Array<PrimExpr> param_exprs,
                                              Array<Var> new_params) const {
  CHECK_EQ(this->parameters.size(), param_exprs.size());
  std::unordered_map<const VarNode*, PrimExpr> replace_map;
  for (size_t i = 0; i < this->parameters.size(); ++i) {
    replace_map[this->parameters[i].as<VarNode>()] = param_exprs[i];
  }

  PrimExpr new_body = VarReplacer(replace_map)(this->body);
  return UninterpFunNode::make(this->fname, this->range, new_params, new_body);
}

PrimExpr UninterpFun::InlineUninterpFunCalls(PrimExpr e) {
  class UninterpInliner : ExprMutator {
    PrimExpr VisitExpr_(const CallNode* op) {
      if (op->func.as<UninterpFunNode>()) {
        CHECK(op->argument_dimensions.defined());
        UninterpFun ufun = Downcast<UninterpFun, FunctionRef>(op->func);
        Array<PrimExpr> arguments;
        for (auto arg : op->args) {
          arguments.push_back(this->VisitExpr(arg));
        }
        return ufun->substitute(arguments, op->argument_dimensions);
      } else {
        return ExprMutator::VisitExpr_(op);
      }
    }

   public:
    PrimExpr Inline(PrimExpr e) { return this->VisitExpr(e); }
  };

  UninterpInliner ui;
  return ui.Inline(e);
}

Range UninterpFun::InlineUninterpFunCalls(Range r) {
  return Range::make_by_min_extent(UninterpFun::InlineUninterpFunCalls(r->min),
                                   UninterpFun::InlineUninterpFunCalls(r->extent));
}

Map<Dimension, PrimExpr> UninterpFun::InvertCall(PrimExpr expr, UninterpFun ufun) {
  if (auto call = expr.as<CallNode>()) {
    if (call->func == ufun) {
      CHECK_EQ(call->args.size(), call->argument_dimensions.size());
      Map<Dimension, PrimExpr> ret;
      for (size_t i = 0; i < call->args.size(); ++i) {
        ret.Set(call->argument_dimensions[i], call->args[i]);
      }
      return ret;
    }
  }
  return {};
}

ArgMappingAndEquality UninterpFun::CheckEquality(UninterpFun f1, UninterpFun f2) {
  PrimExpr e1 = f1->body;
  PrimExpr e2 = f2->body;

  class VarCollector : public ExprVisitor {
    void VisitExpr_(const VarNode* op) { variables.push_back(GetRef<Var>(op)); }

   public:
    Array<Var> variables;
  };

  class VarReplacer : public ExprMutator {
    PrimExpr VisitExpr_(const VarNode* op) {
      if (current_index == replacement.size()) {
        this->overrun = true;
        return ExprMutator::VisitExpr_(op);
      }

      Var replacement_var = replacement[current_index++];
      replace_map.Set(GetRef<Var>(op), replacement_var);
      return replacement_var;
    }

    Array<Var> replacement;
    size_t current_index;

   public:
    bool overrun;
    Map<Var, Var> replace_map;
    VarReplacer(Array<Var> replacement_)
        : replacement(replacement_), current_index(0), overrun(false) {}
  };

  VarCollector collector;
  collector(e1);
  VarReplacer replacer(collector.variables);
  PrimExpr replaced_e2 = replacer(e2);

  if (replacer.overrun) {
    return {false, replacer.replace_map};
  }

  bool ret = tir::ExprEquality().VisitExpr(replaced_e2, e1);
  return {ret, replacer.replace_map};
}

PrimExpr UninterpFun::MakeCallTo(UninterpFun f, Array<PrimExpr> args, Array<Dimension> arg_dims) {
  for (const auto& dim : f->dimensions) {
    CHECK(arg_dims.Contains(dim)) << dim->name << " " << f->body;
  }
  return CallNode::make(DataType::Int(32), f->fname, args, CallNode::UninterpFunCall, arg_dims, f,
                        0);
}

PrimExpr UninterpFun::RelaxComplexUninterpCalls(PrimExpr expr) {
  class Relaxer : public ExprMutator {
    PrimExpr VisitExpr_(const CallNode* op) {
      if (auto ufun = op->func.as<UninterpFunNode>()) {
        if (ufun->is_complex()) {
          return ufun->range->extent;
        }
      }
      return ExprMutator::VisitExpr_(op);
    }
  };

  return Relaxer()(expr);
}

TVM_REGISTER_NODE_TYPE(UninterpFunNode);
TVM_REGISTER_GLOBAL("tir.UninterpFun")
    .set_body_typed([](std::string fname, Range range, Array<Var> parameters,
                       Array<te::Dimension> dims, PrimExpr body) {
      return UninterpFunNode::make(fname, range, dims, parameters, body);
    });
}  // namespace tir
}  // namespace tvm
