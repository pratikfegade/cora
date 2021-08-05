#include <tvm/arith/z3_analyzer.h>
#include <tvm/tir/op.h>

#include <stdexcept>
#include <utility>

namespace tvm {
namespace arith {
using namespace tir;

/*************************************************************************/
z3fun Z3Converter::GetOrCreateZ3Fun(const Var& v) {
  auto it = z3_funs.find(v);
  if (it != z3_funs.end()) return it->second;
  z3::sort_vector params(ctx);
  params.push_back(ctx.int_sort());
  auto z3name = v->name_hint + std::to_string(index++);
  z3fun fun = std::make_shared<z3::func_decl>(z3::function(z3name, params, ctx.int_sort()));
  // std::cout << "[Z3] Creating function " << v << " " << z3name << std::endl;
  return (z3_funs[v] = fun);
}

z3fun Z3Converter::GetOrCreateZ3Fun(const FunctionRef& f, const std::string& name, int arity) {
  // std::cout << "[Z3] Looking for function " << f << std::endl;
  auto ufn = f.as<UninterpFunNode>();
  if (ufn && ufn->body.defined()) {
    auto it = z3_ufuns.find(Downcast<UninterpFun>(f));
    if (it != z3_ufuns.end()) {
      // std::cout << "[Z3]  Found1 " << it->second->name() << std::endl;
      return it->second;
    }

    z3::sort_vector params(ctx);
    for (int i = 0; i < arity; ++i) {
      params.push_back(ctx.int_sort());
    }
    auto z3name = name + std::to_string(index++);
    z3fun fun = std::make_shared<z3::func_decl>(z3::function(z3name, params, ctx.int_sort()));
    // std::cout << "[Z3] Creating function1 " << f << " " << z3name << std::endl;
    return (z3_ufuns[Downcast<UninterpFun>(f)] = fun);
  } else {
    auto it = z3_funs.find(f);
    if (it != z3_funs.end()) {
      // std::cout << "[Z3]  Found2 " << it->second->name() << std::endl;
      return it->second;
    }

    z3::sort_vector params(ctx);
    for (int i = 0; i < arity; ++i) {
      params.push_back(ctx.int_sort());
    }
    auto z3name = name + std::to_string(index++);
    z3fun fun = std::make_shared<z3::func_decl>(z3::function(z3name, params, ctx.int_sort()));
    // std::cout << "[Z3] Creating function2 " << f << " " << z3name << std::endl;
    return (z3_funs[f] = fun);
  }
}

z3expr Z3Converter::VisitExpr_(const VarNode* op) {
  auto key = GetRef<PrimExpr>(op);
  auto it = z3_exprs.find(key);
  if (it != z3_exprs.end()) return it->second;
  auto z3var =
      std::make_shared<z3::expr>(ctx.int_const((op->name_hint + std::to_string(index++)).c_str()));
  z3_exprs[key] = z3var;
  return z3var;
}
z3expr Z3Converter::VisitExpr_(const SizeVarNode* op) {
  auto key = GetRef<PrimExpr>(op);
  auto it = z3_exprs.find(key);
  if (it != z3_exprs.end()) return it->second;
  auto z3var =
      std::make_shared<z3::expr>(ctx.int_const((op->name_hint + std::to_string(index++)).c_str()));
  z3_exprs[key] = z3var;
  return z3var;
}
z3expr Z3Converter::VisitExpr_(const LoadNode* op) {
  auto key = GetRef<PrimExpr>(op);
  auto it = z3_exprs.find(key);
  if (it != z3_exprs.end()) return it->second;

  z3::func_decl fun = *GetOrCreateZ3Fun(op->buffer_var);
  z3::expr arg = *this->VisitExpr(op->index);
  return (z3_exprs[key] = std::make_shared<z3::expr>(fun(arg)));
}
z3expr Z3Converter::VisitRightShift(const CallNode* op) {
  z3::expr a = *this->VisitExpr(op->args[0]);
  z3::expr b = *this->VisitExpr(op->args[1]);
  return std::make_shared<z3::expr>(a / z3::pw(2, b));
}
z3expr Z3Converter::VisitExpr_(const CallNode* op) {
  if (op->is_intrinsic(CallNode::shift_right)) {
    return VisitRightShift(op);
  } else if (op->is_intrinsic(CallNode::likely)) {
    CHECK_EQ(op->args.size(), 1);
    return this->VisitExpr(op->args[0]);
  } else if (op->is_pure() && (op->dtype.is_int() || op->dtype.is_uint())) {
    auto key = GetRef<PrimExpr>(op);
    auto it = z3_exprs.find(key);
    if (it != z3_exprs.end()) {
      return it->second;
    }

    auto ufn = op->func.as<UninterpFunNode>();
    if (ufn && ufn->body.defined() && !ufn->is_complex()) {
      return this->VisitExpr(ufn->substitute(op->args, op->argument_dimensions));
    }

    z3::func_decl fun = *GetOrCreateZ3Fun(op->func, op->name, op->args.size());
    z3::expr_vector args(ctx);
    for (auto arg : op->args) {
      args.push_back(*this->VisitExpr(arg));
    }
    // std::cout << "[Z3] Function " << op->func << " " << GetRef<PrimExpr>(op) << std::endl;
    auto res = std::make_shared<z3::expr>(fun(args));
    z3_exprs[key] = res;
    return res;
  } else {
    throw std::invalid_argument("Cannot convert this expression to a Z3 expression");
  }
}
z3expr Z3Converter::VisitExpr_(const CastNode* op) {
  auto key = GetRef<PrimExpr>(op);
  auto it = z3_exprs.find(key);
  if (it != z3_exprs.end()) return it->second;
  return this->VisitExpr(op->value);
}
z3expr Z3Converter::VisitExpr_(const NotNode* op) {
  auto key = GetRef<PrimExpr>(op);
  auto it = z3_exprs.find(key);
  if (it != z3_exprs.end()) return it->second;
  std::cout << "[Z3] Not " << GetRef<PrimExpr>(op) << " " << op->a.dtype() << std::endl;
  return std::make_shared<z3::expr>(!*this->VisitExpr(op->a));
}
z3expr Z3Converter::VisitExpr_(const IntImmNode* op) {
  auto key = GetRef<PrimExpr>(op);
  auto it = z3_exprs.find(key);
  if (it != z3_exprs.end()) return it->second;
  return std::make_shared<z3::expr>(ctx.int_val(op->value));
}
z3expr Z3Converter::VisitExpr_(const SelectNode* op) {
  auto key = GetRef<PrimExpr>(op);
  auto it = z3_exprs.find(key);
  if (it != z3_exprs.end()) return it->second;
  return std::make_shared<z3::expr>(z3::ite(*this->VisitExpr(op->condition),
                                            *this->VisitExpr(op->true_value),
                                            *this->VisitExpr(op->false_value)));
}

#define BINOP_CREATE_Z3(TVM_OP, OP_FUN)                                         \
  z3expr Z3Converter::VisitExpr_(const TVM_OP* op) {                            \
    auto key = GetRef<PrimExpr>(op);                                            \
    auto it = z3_exprs.find(key);                                               \
    if (it != z3_exprs.end()) return it->second;                                \
    return (z3_exprs[key] = std::make_shared<z3::expr>(                         \
                z3::OP_FUN(*this->VisitExpr(op->a), *this->VisitExpr(op->b)))); \
  }

BINOP_CREATE_Z3(AddNode, operator+)
BINOP_CREATE_Z3(SubNode, operator-)
BINOP_CREATE_Z3(MulNode, operator*)
BINOP_CREATE_Z3(DivNode, operator/)
BINOP_CREATE_Z3(ModNode, operator%)
// BINOP_CREATE_Z3(ModNode, mod)
BINOP_CREATE_Z3(FloorDivNode, operator/)
// BINOP_CREATE_Z3(FloorModNode, operator%)
BINOP_CREATE_Z3(FloorModNode, mod)
BINOP_CREATE_Z3(MinNode, min)
BINOP_CREATE_Z3(MaxNode, max)
BINOP_CREATE_Z3(EQNode, operator==)
BINOP_CREATE_Z3(NENode, operator!=)
BINOP_CREATE_Z3(LTNode, operator<)
BINOP_CREATE_Z3(LENode, operator<=)
BINOP_CREATE_Z3(GTNode, operator>)
BINOP_CREATE_Z3(GENode, operator>=)
BINOP_CREATE_Z3(AndNode, operator&&)
BINOP_CREATE_Z3(OrNode, operator||)

#undef BINOP_CREATE_Z3

z3expr Z3Converter::VisitExprDefault_(const Object* op) {
  throw std::invalid_argument("Cannot convert this expression to a Z3 expression");
}

/*************************************************************************/

z3::expr Z3Analyzer::ConvertToZ3(const PrimExpr& expr) {
  return (*this->converter)(expr)->simplify();
}

void Z3Analyzer::Bind(const Var& var, const Range& range) { this->Update(var, range, false); }

void Z3Analyzer::Update(const Var& var, const PrimExpr& expr, bool overwrite) {
  if (!expr->dtype.is_int() && !expr->dtype.is_uint()) return;
  this->Update(var, expr, expr + 1, overwrite);
}

void Z3Analyzer::Update(const Var& var, const Range& range, bool overwrite) {
  this->Update(var, range->min, range->min + range->extent, overwrite);
}

void Z3Analyzer::Update(const Var& var, const PrimExpr& min, const PrimExpr& max, bool overwrite) {
  try {
    // std::cout << "[Z3] Const: " << var << " " << min << " " << max << std::endl;

    z3::expr z3min = ConvertToZ3(min);
    z3::expr z3max = ConvertToZ3(max);
    z3::expr z3var = ConvertToZ3(var);

    if (!var_constraints.count(var.get()) || overwrite) {
      var_constraints[var.get()] = std::make_shared<z3::expr_vector>(ctx);
    }
    var_constraints.at(var.get())->push_back(z3var >= z3min);
    var_constraints.at(var.get())->push_back(z3var < z3max);
  } catch (const std::invalid_argument& e) {
    return;
  } catch (const z3::exception& e) {
    return;
  }
}

void Z3Analyzer::AddConstraint(const PrimExpr& constraint) {
  // std::cout << "[Z3]  Adding constraint " << constraint << std::endl;
  if (auto imm = constraint.as<IntImmNode>()) {
    this->general_constraints->push_back(ctx.bool_val(imm != 0));
  } else if (constraint.dtype().is_bool()) {
    z3::expr z3constraint = ConvertToZ3(constraint);
    // std::cout << "[Z3]   Constraint " << z3constraint << std::endl;
    this->general_constraints->push_back(z3constraint);
  } else {
    this->general_constraints->push_back(ctx.bool_val(true));
  }
}

void Z3Analyzer::AddForallConstraint(const Array<Var>& forall_vars,
                                     const PrimExpr& constraint_body) {
  if (constraint_body.as<IntImmNode>()) {
    this->general_constraints->push_back(ctx.bool_val(true));
  } else {
    z3::expr z3constraint_body = ConvertToZ3(constraint_body);
    z3::expr_vector z3forall_vars(ctx);

    for (auto var : forall_vars) {
      z3forall_vars.push_back(ConvertToZ3(var));
    }

    // std::cout << "[Z3] ForallConstraint: " << constraint_body << std::endl;
    z3::expr z3constraint = z3::forall(z3forall_vars, z3constraint_body);
    this->general_constraints->push_back(z3constraint);
  }
}

void Z3Analyzer::RemoveLastConstraint() { this->general_constraints->pop_back(); }

bool Z3Analyzer::CanProveInternal_(z3::expr& antecedent, z3::expr& consequent, bool print) {
  z3::solver solver(ctx);

  try {
    z3::params p(ctx);
    p.set(":timeout", 500u);
    // p.set(":produce-proofs", true);
    // p.set(":produce-models", true);
    solver.set(p);

    z3::expr to_prove = z3::implies(antecedent, consequent).simplify();

    solver.add(!to_prove);
    if (solver.check() == z3::unsat) {
      // std::cout << "[Z3] Proof " << solver.proof() << std::endl;
      return true;
    } else {
      // if (print) std::cout << "[Z3] Cannot prove. Have model.\n" << solver.get_model() <<
      // std::endl;
      return false;
    }
  } catch (const std::invalid_argument& e) {
    return false;
  } catch (const z3::exception& e) {
    return false;
  }
}

bool Z3Analyzer::CanProve(const PrimExpr& cond) {
  z3::expr antecedent = ctx.bool_val(true);
  for (auto it : var_constraints) {
    for (auto expr : *it.second) {
      antecedent = antecedent && expr;
    }
  }

  for (auto expr : *this->general_constraints) {
    antecedent = antecedent && expr;
  }

  z3::expr false_expr = ctx.bool_val(false);
  CHECK(!CanProveInternal_(antecedent, false_expr, false))
      << "Invalid constraints added to the solver";

  z3::expr consequent = ConvertToZ3(cond);

  return CanProveInternal_(antecedent, consequent, true);
}
}  // namespace arith
}  // namespace tvm
