#ifndef TVM_TIR_UF_EQUALITY_H_
#define TVM_TIR_UF_EQUALITY_H_

#include <tvm/arith/int_set.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/container.h>
#include <tvm/te/cache_info.h>
#include <tvm/te/dimension.h>
#include <tvm/tir/expr.h>

#include <vector>

namespace tvm {
namespace tir {
Map<te::Dimension, arith::IntSet> ProjectInverse(
    arith::IntSet range_set, UninterpFun fun,
    const Map<FunctionRef, te::CacheInfo> cacheTensorInfos);

}  // namespace tir
}  // namespace tvm

#endif
