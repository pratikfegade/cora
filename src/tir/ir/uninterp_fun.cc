#include <tvm/arith/int_set.h>
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/cache_info.h>
#include <tvm/te/dimension.h>
#include <tvm/tir/expr_equality.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/uf_equality.h>
#include <tvm/tir/uninterp_fun.h>

#include <vector>

#include "../../arith/interval_set.h"
#include "../../arith/projection_set.h"
#include "var_replacer.h"

namespace tvm {
namespace tir {
Map<te::Dimension, arith::IntSet> ProjectInverse(arith::IntSet range_set, UninterpFun fun) {
  if (range_set.is_nothing()) {
    Map<te::Dimension, arith::IntSet> ret;
    for (auto dim : fun->dimensions) {
      ret.Set(dim, arith::IntervalSet::Empty());
    }
    return ret;
  }
  if (auto s_proj = range_set.as<arith::ProjectionSetNode>()) {
    auto mapping_and_equals = CheckUninterpFunEquality(s_proj->ufun, fun);
    // std::cout << "[PI]  " << mapping_and_equals.equals << " " << s_proj->ufun->body << " " <<
    // fun->body << std::endl;
    if (mapping_and_equals.equals) {
      return Map<te::Dimension, arith::IntSet>(s_proj->arguments);
    }
  }
  return {};
}

bool UfBodyEquality::VisitExpr_(const CallNode* op1, const CallNode* op2) {
  FunctionRef f1 = op1->func;
  FunctionRef f2 = op2->func;

  Array<PrimExpr> args1;
  Array<PrimExpr> args2;
  if (f1 != f2) {
    // std::cout << "[UFEQ] Checking " << GetRef<PrimExpr>(op2) << " " << GetRef<PrimExpr>(op1) <<
    // f1
    // << " " << f2 << std::endl;
    // for (auto it : cacheTensorInfos) {
    // std::cout << "[UFEQ]   Map " << it.first << " " << it.second->orig << std::endl;
    // }
    bool present1 = cacheTensorInfos.count(f1);
    bool present2 = cacheTensorInfos.count(f2);

    if (!present1 && !present2)
      return false;
    else if (present1 && present2) {
      te::CacheInfo ci1 = cacheTensorInfos.at(f1);
      te::CacheInfo ci2 = cacheTensorInfos.at(f2);

      if (ci1->orig != ci2->orig) return false;
      if (op1->args.size() != op2->args.size()) return false;

      PrimExpr ve1 = op1->args[op1->args.size() - 1];
      PrimExpr ve2 = op2->args[op2->args.size() - 1];

      if (ve1.as<IntImmNode>() && ve2.as<IntImmNode>()) {
        int v1 = ve1.as<IntImmNode>()->value;
        int v2 = ve2.as<IntImmNode>()->value;
        Map<Dimension, Dimension> m1 = ci1->variantMappings[v1];
        Map<Dimension, Dimension> m2 = ci2->variantMappings[v2];
        if (m1 != m2) return false;
      } else {
        return false;
      }

      for (size_t i = 0; i < op2->args.size() - 1; ++i) {
        args1.push_back(op1->args[i]);
        args2.push_back(op2->args[i]);
      }
    } else if (!present2 && present1) {
      std::swap(f1, f2);
      std::swap(op1, op2);
    }

    // std::cout << "[UFEQ]   1 " << std::endl;

    // !present1 && present2
    if (op1->args.size() != op2->args.size() - 1) return false;
    // std::cout << "[UFEQ]   2 " << std::endl;

    te::CacheInfo ci2 = cacheTensorInfos.at(f2);
    PrimExpr ve2 = op2->args[op2->args.size() - 1];
    // std::cout << "[UFEQ]   3 " << std::endl;

    if (ve2.as<IntImmNode>()) {
      // std::cout << "[UFEQ]   4 " << std::endl;
      int v2 = ve2.as<IntImmNode>()->value;
      Map<Dimension, Dimension> m2 = ci2->variantMappings[v2];
      for (const auto& it : m2) {
        if (it.first != it.second) return false;
      }
    } else {
      return false;
    }

    args1 = Array<PrimExpr>(op1->args);
    for (size_t i = 0; i < op2->args.size() - 1; ++i) {
      args2.push_back(op2->args[i]);
    }
  } else {
    args1 = Array<PrimExpr>(op1->args);
    args2 = Array<PrimExpr>(op2->args);
  }

  return tir::ExprEquality::VisitArray(
      args1, args2,
      [this](const PrimExpr& e1, const PrimExpr& e2) { return this->VisitExpr(e1, e2); });
}

Map<FunctionRef, te::CacheInfo> UfBodyEquality::cacheTensorInfos =
    NullValue<Map<FunctionRef, te::CacheInfo>>();
ArgMappingAndEquality CheckUninterpFunEquality(UninterpFun f1, UninterpFun f2) {
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

  bool ret = UfBodyEquality().VisitExpr(replaced_e2, e1);
  return {ret, replacer.replace_map};
}

ArgMappingAndEquality UninterpFun::CheckEquality(UninterpFun f1, UninterpFun f2) {
  CHECK(false)
      << "Do not use this for checking UF equality. This does not deal with cached tensors";
  return {};
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
  CHECK_EQ(n->parameters.size(), n->dimensions.size());
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
    if (arg_dim_map.count(param_dim) == 0) {
      std::cout << param_dim->name;
    }
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

class UninterpCallInliner : StmtExprMutator {
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
  Stmt Inline(Stmt e) { return this->VisitStmt(e); }
};

PrimExpr UninterpFun::InlineUninterpFunCalls(PrimExpr e) {
  UninterpCallInliner ui;
  return ui.Inline(e);
}

Stmt UninterpFun::InlineUninterpFunCalls(Stmt e) {
  UninterpCallInliner ui;
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
