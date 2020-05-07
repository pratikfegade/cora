#ifndef TVM_ARITH_Z3_ANALYZER_H_
#define TVM_ARITH_Z3_ANALYZER_H_

#include <tvm/support/with.h>
#include <tvm/ir/expr.h>
#include <tvm/arith/int_set.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

#include <vector>
#include <unordered_map>
#include <memory>
#include <limits>

#include "z3++.h"

namespace tvm {
/*! \brief namespace of arithmetic analysis. */
namespace arith {
using namespace tir;

using z3expr = std::shared_ptr<z3::expr>;
using z3exprvec = std::shared_ptr<z3::expr_vector>;
using z3fun = std::shared_ptr<z3::func_decl>;

class Z3Converter: public tir::ExprFunctor<z3expr(const PrimExpr&)> {
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

#define BINOP_DECLARE_CONVERTER_FUN(TVM_OP, OP_FUN)                                 \
  z3expr VisitExpr_(const TVM_OP* op) override;                			    \

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

  Z3Converter(std::unordered_map<const Object*, z3expr>& z3_exprs_,
	      std::unordered_map<const Object*, z3fun>& z3_funs_, z3::context& ctx_) :
    z3_exprs(z3_exprs_), z3_funs(z3_funs_), ctx(ctx_) {}

private:
  std::unordered_map<const Object*, z3expr>& z3_exprs;
  std::unordered_map<const Object*, z3fun>& z3_funs;
  z3::context& ctx;
  int index = 0;
};


class Z3Analyzer {
public:
  std::unordered_map<const Object*, z3expr> z3_exprs;
  std::unordered_map<const Object*, z3fun> z3_funs;
  z3::context ctx;
  Z3Converter converter;

  Z3Analyzer() : converter(z3_exprs, z3_funs, ctx) {
    // std::cout << "New analyzer" << std::endl;
  }

  void Bind(const Var& var, const Range& range);
  void Update(const Var& var, const Range& range, bool overwrite);
  void Update(const Var& var, const PrimExpr& expr, bool overwrite);
  void Update(const Var& var, const PrimExpr& min, const PrimExpr& max, bool overwrite);
  z3::expr ConvertToZ3(const PrimExpr& expr);
  bool CanProve(const PrimExpr& cond);

private:
  std::unordered_map<const Object*, z3exprvec> var_constraints;
};
}
}
#endif
