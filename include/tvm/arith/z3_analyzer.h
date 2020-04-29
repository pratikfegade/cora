#ifndef TVM_ARITH_Z3_ANALYZER_H_
#define TVM_ARITH_Z3_ANALYZER_H_

#include <tvm/support/with.h>
#include <tvm/ir/expr.h>
#include <tvm/arith/int_set.h>

#include <vector>
#include <unordered_map>
#include <memory>
#include <limits>

#include "z3++.h"

namespace tvm {
/*! \brief namespace of arithmetic analysis. */
namespace arith {

using z3expr = std::shared_ptr<z3::expr>;
using z3fun = std::shared_ptr<z3::func_decl>;

class Z3Analyzer {
public:
  std::unordered_map<const Object*, z3expr> z3_exprs;
  std::unordered_map<const Object*, z3fun> z3_funs;
  z3::context ctx;

  Z3Analyzer() : constraints(ctx) {}

  void Bind(const Var& var, const Range& range);

  z3::expr ConvertToZ3(const PrimExpr& expr);

  bool CanProve(const PrimExpr& cond);

private:
  z3::expr_vector constraints;
};
}
}
#endif
