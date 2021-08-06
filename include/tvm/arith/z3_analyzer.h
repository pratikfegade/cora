#ifndef TVM_ARITH_Z3_ANALYZER_H_
#define TVM_ARITH_Z3_ANALYZER_H_

#include <tvm/arith/int_set.h>
#include <tvm/ir/expr.h>
#include <tvm/support/with.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include "z3++.h"

namespace tvm {
/*! \brief namespace of arithmetic analysis. */
namespace arith {
using namespace tir;

using z3expr = std::shared_ptr<z3::expr>;
using z3exprvec = std::shared_ptr<z3::expr_vector>;
using z3fun = std::shared_ptr<z3::func_decl>;

class Z3Converter : public tir::ExprFunctor<z3expr(const PrimExpr&)> {
 public:
  z3fun GetOrCreateZ3Fun(const Var& v);
  z3fun GetOrCreateZ3Fun(const FunctionRef& f, const std::string& name, int arity);
  z3expr VisitExpr_(const VarNode* op) override;
  z3expr VisitExpr_(const SizeVarNode* op) override;
  z3expr VisitExpr_(const LoadNode* op) override;
  z3expr VisitRightShift(const CallNode* op);
  z3expr VisitExpr_(const CallNode* op) override;
  z3expr VisitExpr_(const CastNode* op) override;
  z3expr VisitExpr_(const NotNode* op) override;
  z3expr VisitExpr_(const IntImmNode* op) override;
  z3expr VisitExpr_(const SelectNode* op) override;

#define BINOP_DECLARE_CONVERTER_FUN(TVM_OP, OP_FUN) z3expr VisitExpr_(const TVM_OP* op) override;

  BINOP_DECLARE_CONVERTER_FUN(AddNode, operator+)
  BINOP_DECLARE_CONVERTER_FUN(SubNode, operator-)
  BINOP_DECLARE_CONVERTER_FUN(MulNode, operator*)
  BINOP_DECLARE_CONVERTER_FUN(DivNode, operator/)
  BINOP_DECLARE_CONVERTER_FUN(ModNode, operator%)
  BINOP_DECLARE_CONVERTER_FUN(FloorDivNode, operator/)
  BINOP_DECLARE_CONVERTER_FUN(FloorModNode, operator%)
  BINOP_DECLARE_CONVERTER_FUN(MinNode, min)
  BINOP_DECLARE_CONVERTER_FUN(MaxNode, max)
  BINOP_DECLARE_CONVERTER_FUN(EQNode, operator==)
  BINOP_DECLARE_CONVERTER_FUN(NENode, operator!=)
  BINOP_DECLARE_CONVERTER_FUN(LTNode, operator<)
  BINOP_DECLARE_CONVERTER_FUN(LENode, operator<=)
  BINOP_DECLARE_CONVERTER_FUN(GTNode, operator>)
  BINOP_DECLARE_CONVERTER_FUN(GENode, operator>=)
  BINOP_DECLARE_CONVERTER_FUN(AndNode, operator&&)
  BINOP_DECLARE_CONVERTER_FUN(OrNode, operator||)
#undef BINOP_DECLARE_CONVERTER_FUN

  z3expr VisitExprDefault_(const Object* op) override;

  Z3Converter(z3::context& ctx_) : ctx(ctx_) {
    // std::cout << "[Z3] -----" << std::endl;
  }

  class UfHasher {
   public:
    size_t operator()(UninterpFun uf) const {
      size_t hash = 0;
      return hash;
    }
  };

  class UfEquality {
   public:
    bool operator()(UninterpFun uf1, UninterpFun uf2) const {
      return UninterpFun::CheckEquality(uf1, uf2).equals;
    }
  };

  class ObjectRefHasher {
   public:
    size_t operator()(const ObjectRef& o) const { return std::hash<const Object*>()(o.get()); }
  };

  class ObjectRefEquality {
   public:
    bool operator()(const ObjectRef& o1, const ObjectRef& o2) const { return o1 == o2; }
  };

 private:
  z3::context& ctx;
  std::unordered_map<PrimExpr, z3expr, ObjectRefHasher, ObjectRefEquality> z3_exprs;
  std::unordered_map<ObjectRef, z3fun, ObjectRefHasher, ObjectRefEquality> z3_funs;
  std::unordered_map<UninterpFun, z3fun, UfHasher, UfEquality> z3_ufuns;
  int index = 0;
};

class Z3Analyzer {
 public:
  Z3Analyzer() {
    this->converter = std::unique_ptr<Z3Converter>(new Z3Converter(ctx));
    this->general_constraints = std::make_shared<z3::expr_vector>(ctx);
    InitCall_();
  }

  void Bind(const Var& var, const Range& range);
  void Update(const Var& var, const Range& range, bool overwrite);
  void Update(const Var& var, const PrimExpr& expr, bool overwrite);
  void Update(const Var& var, const PrimExpr& min, const PrimExpr& max, bool overwrite);
  void AddConstraint(const PrimExpr& constraint);
  void AddForallConstraint(const Array<Var>& forall_vars, const PrimExpr& constraint_body);
  void RemoveLastConstraint();
  z3::expr ConvertToZ3(const PrimExpr& expr);
  bool CanProve(const PrimExpr& cond);

 private:
  bool CanProveInternal_(z3::expr& antecedent, z3::expr& consequent, bool print);
  void InitCall_();

  z3::context ctx;
  std::unique_ptr<Z3Converter> converter;
  std::unordered_map<const Object*, z3exprvec> var_constraints;
  z3exprvec general_constraints;
};
}  // namespace arith
}  // namespace tvm
#endif
