#ifndef TVM_TIR_UF_EQUALITY_H_
#define TVM_TIR_UF_EQUALITY_H_

#include <tvm/arith/int_set.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/container.h>
#include <tvm/te/cache_info.h>
#include <tvm/te/dimension.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_equality.h>

#include <vector>

namespace tvm {
namespace tir {
Map<te::Dimension, arith::IntSet> ProjectInverse(arith::IntSet range_set, UninterpFun fun);

class UfBodyEquality : public ExprEquality {
  bool VisitExpr_(const CallNode* op1, const CallNode* op2) const override;

 public:
  UfBodyEquality() {}
  static Map<FunctionRef, te::CacheInfo> cacheTensorInfos;
};

}  // namespace tir
}  // namespace tvm

#endif
