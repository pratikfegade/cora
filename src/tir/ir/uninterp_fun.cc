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
RaggedFusionInfo RaggedFusionInfoNode::make(IterVar outer, IterVar inner, IterVar fused,
                                            FunctionRef fused_to_outer_uf,
                                            FunctionRef fused_to_inner_uf,
                                            FunctionRef outer_inner_to_fused_uf) {
  auto n = make_object<RaggedFusionInfoNode>();
  n->outer = outer;
  n->inner = inner;
  n->fused = fused;
  n->fused_to_outer_uf = fused_to_outer_uf;
  n->fused_to_inner_uf = fused_to_inner_uf;
  n->outer_inner_to_fused_uf = outer_inner_to_fused_uf;
  return RaggedFusionInfo(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<UninterpFunNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const UninterpFunNode*>(node.get());
      p->stream << "UninterpFun(" << op->fname << ", " << op->body << ", " << op << ")";
    });

Map<te::Dimension, arith::IntSet> ProjectInverse(arith::IntSet range_set, UninterpFun fun) {
  CHECK_EQ(fun->dimensions.size(), fun->parameters.size());
  if (range_set.is_nothing()) {
    Map<te::Dimension, arith::IntSet> ret;
    for (auto dim : fun->dimensions) {
      ret.Set(dim, arith::IntervalSet::Empty());
    }
    return ret;
  }
  if (auto s_proj = range_set.as<arith::ProjectionSetNode>()) {
    auto mapping_and_equals = UninterpFun::CheckEquality(s_proj->ufun, fun);
    // std::cout << "[PI]  " << mapping_and_equals.equals << " " << s_proj->ufun->body << " " <<
    // fun->body << std::endl;
    if (mapping_and_equals.equals) {
      return Map<te::Dimension, arith::IntSet>(s_proj->arguments);
    }
  }
  return {};
}

bool UfBodyEquality::VisitExpr_(const CallNode* op1, const CallNode* op2) const {
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

  bool ret = UfBodyEquality().VisitExpr(replaced_e2, e1);
  return {ret, replacer.replace_map};
}

UninterpFun UninterpFunNode::make(std::string fname, Range range,
                                  Array<tvm::te::Dimension> dimensions, Array<Var> parameters,
                                  PrimExpr body, UninterpFunType type) {
  CHECK(parameters.size() == dimensions.size());
  // if (dimensions.size() == 0 && parameters.size() > 0) {
  //   std::cout << "[UF] No dim UF " << fname << std::endl;
  // }
  ObjectPtr<UninterpFunNode> n = make_object<UninterpFunNode>();
  // if (fname == "s2") {
  //   CHECK(dimensions.defined());
  //   std::cout << "[UF] bd_afun found " << n.get() << " " << dimensions.size() << std::endl;
  // }
  n->fname = fname;
  n->range = range;
  n->dimensions = dimensions;
  n->parameters = parameters;
  n->body = body;
  n->type = type;
  return UninterpFun(n);
}

UninterpFun UninterpFunNode::from_constant(std::string fname, PrimExpr val, UninterpFunType type) {
  return UninterpFunNode::make(fname, Range::make_by_min_extent(val, 1), {}, {}, val, type);
}

bool UninterpFunNode::is_constant() const { return body.defined() && body.as<IntImmNode>(); }

size_t UninterpFunNode::arity() const { return this->parameters.size(); }

int UninterpFunNode::num_outputs() const { return 1; }

class ComplexExprChecker : public ExprVisitor {
 public:
  void VisitExpr_(const CallNode* op) final { complex = true; }
  void VisitExpr_(const LoadNode* op) final { complex = true; }

  bool complex{false};
};

bool UninterpFunNode::is_complex() const {
  if (!this->body.defined()) return true;
  ComplexExprChecker checker;
  checker(this->body);
  return checker.complex;
}

const PrimExpr UninterpFunNode::substitute(Array<PrimExpr> args,
                                           Array<tvm::te::Dimension> arg_dims) const {
  std::unordered_map<const VarNode*, PrimExpr> replace_map;
  // if (args.size() != arg_dims.size()) {
  // std::cout << "Really?" << std::endl;
  // }
  CHECK_EQ(args.size(), arg_dims.size());
  Map<tvm::te::Dimension, PrimExpr> arg_dim_map;
  for (size_t i = 0; i < args.size(); ++i) {
    arg_dim_map.Set(arg_dims[i], args[i]);
  }

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

void UninterpFunNode::SetBody(PrimExpr expr) {
  // std::cout << "[UFUN] Setting body for " << fname << "  " << expr << " " << this << std::endl;
  this->body = expr;
}

void UninterpFunNode::SetRange(Range r) { this->range = r; }

class UninterpCallInliner : StmtExprMutator {
  PrimExpr VisitExpr_(const CallNode* op) {
    if (op->func.as<UninterpFunNode>()) {
      bool print = false;  // op->name == "lens";
      if (print) std::cout << "[IUF] Found call " << GetRef<PrimExpr>(op) << std::endl;
      CHECK(op->arg_dims.defined());
      UninterpFun ufun = Downcast<UninterpFun, FunctionRef>(op->func);
      if (only_simple && ufun->is_complex()) return ExprMutator::VisitExpr_(op);
      if (!ufun->body.defined()) return ExprMutator::VisitExpr_(op);
      Array<PrimExpr> arguments;
      for (auto arg : op->args) {
        arguments.push_back(this->VisitExpr(arg));
      }
      if (print) std::cout << "[IUF]  Substituting" << std::endl;
      return ufun->substitute(arguments, op->arg_dims);
    } else {
      if (op->custom_realize_bounds.size() > 0) {
        Array<Range> new_bounds;
        for (auto r : op->custom_realize_bounds) {
          new_bounds.push_back(
              Range::make_by_min_extent(this->VisitExpr(r->min), this->VisitExpr(r->extent)));
        }
        Array<PrimExpr> new_args;
        for (auto arg : op->args) {
          new_args.push_back(this->VisitExpr(arg));
        }
        return CallNode::make(op->dtype, op->name, new_args, op->call_type, op->arg_dims, op->func,
                              op->value_index, new_bounds);
      }

      return ExprMutator::VisitExpr_(op);
    }
  }

 public:
  PrimExpr Inline(PrimExpr e) { return this->VisitExpr(e); }
  Stmt Inline(Stmt e) { return this->VisitStmt(e); }
  bool only_simple{false};

  UninterpCallInliner(bool only_simple_) : only_simple(only_simple_) {}
};

PrimExpr UninterpFun::InlineUninterpFunCalls(PrimExpr e, bool only_simple) {
  UninterpCallInliner ui(only_simple);
  return ui.Inline(e);
}

Stmt UninterpFun::InlineUninterpFunCalls(Stmt e, bool only_simple) {
  UninterpCallInliner ui(only_simple);
  return ui.Inline(e);
}

Range UninterpFun::InlineUninterpFunCalls(Range r, bool only_simple) {
  return Range::make_by_min_extent(UninterpFun::InlineUninterpFunCalls(r->min, only_simple),
                                   UninterpFun::InlineUninterpFunCalls(r->extent, only_simple));
}

Map<Dimension, PrimExpr> UninterpFun::InvertCall(PrimExpr expr, UninterpFun ufun) {
  if (auto call = expr.as<CallNode>()) {
    if (call->func == ufun) {
      CHECK_EQ(call->args.size(), call->arg_dims.size());
      Map<Dimension, PrimExpr> ret;
      for (size_t i = 0; i < call->args.size(); ++i) {
        ret.Set(call->arg_dims[i], call->args[i]);
      }
      return ret;
    }
  }
  return {};
}

const PrimExpr UninterpFun::MakeCallTo(Array<PrimExpr> args, Array<Dimension> arg_dims,
                                       DataType dtype) const {
  auto self = (*this).operator->();
  for (const auto& dim : self->dimensions) {
    if (!arg_dims.Contains(dim)) {
      std::cout << dim->name << " " << *this << std::endl;
    }
    CHECK(arg_dims.Contains(dim)) << dim->name << " " << (*this);
  }
  return CallNode::make(dtype.is_handle() ? DataType::Int(32) : dtype, self->fname, args,
                        CallNode::UninterpFunCall, arg_dims, *this, 0);
}

PrimExpr UninterpFun::RelaxUninterpCallsMaxInclusive(PrimExpr expr, bool complex_only) {
  class Relaxer : public ExprMutator {
   public:
    Relaxer(bool complex_only) : complex_only_(complex_only) {}
    bool max = true;
    bool complex_only_;

    PrimExpr VisitExpr_(const CallNode* op) {
      if (auto ufun = op->func.as<UninterpFunNode>()) {
        // std::cout << "[RUF]    UF " << op->func << " " << complex_only_ << " " <<
        // ufun->is_complex()
        //           << std::endl;
        if (!complex_only_ || ufun->is_complex()) {
          return max ? ufun->range->max_inclusive() : ufun->range->min;
        }
      }
      return ExprMutator::VisitExpr_(op);
    }

    PrimExpr VisitExpr_(const SubNode* op) {
      PrimExpr av = this->VisitExpr(op->a);
      max = !max;
      PrimExpr bv = this->VisitExpr(op->b);
      max = !max;
      return (av - bv);
    }
  };
  // std::cout << "[RUF]   Relaxing uf " << expr << " " << complex_only << std::endl;

  return Relaxer(complex_only)(expr);
}

TVM_REGISTER_NODE_TYPE(UninterpFunNode);
TVM_REGISTER_GLOBAL("tir.UninterpFun")
    .set_body_typed([](std::string fname, Range range, Array<Var> parameters,
                       Array<te::Dimension> dims, PrimExpr body, int type) {
      return UninterpFunNode::make(fname, range, dims, parameters, body,
                                   static_cast<UninterpFunNode::UninterpFunType>(type));
    });
}  // namespace tir
}  // namespace tvm
