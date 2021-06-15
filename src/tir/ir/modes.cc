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
Modes ModesNode::make(Array<tvm::te::Dimension> dimensions, Array<UninterpFun> dim_widths) {
  CHECK_EQ(dim_widths.size(), dimensions.size());
  ObjectPtr<ModesNode> n = make_object<ModesNode>();
  n->dimensions = dimensions;
  n->dim_widths = dim_widths;
  return Modes(n);
}

Modes ModesNode::make(Array<tvm::te::Dimension> dimensions, Array<PrimExpr> dense_shape) {
  CHECK_EQ(dense_shape.size(), dimensions.size());
  Array<UninterpFun> dim_widths;
  for (size_t i = 0; i < dense_shape.size(); ++i) {
    dim_widths.push_back(
        UninterpFunNode::from_constant(dimensions[i]->name + "_w", dense_shape[i]));
  }
  return ModesNode::make(dimensions, dim_widths);
}

Modes ModesNode::make(std::string name, Array<PrimExpr> dense_shape) {
  std::cerr << "[MODES] Modes object created without dimensions for " << name << std::endl;
  Array<UninterpFun> dim_widths;
  Array<Dimension> dimensions;
  for (size_t i = 0; i < dense_shape.size(); ++i) {
    dimensions.push_back(te::DimensionNode::make("mode_dim_" + std::to_string(i),
                                                 te::DimensionNode::DimensionType::kRangeDim));
    dim_widths.push_back(
        UninterpFunNode::from_constant("mode_fun_" + std::to_string(i), dense_shape[i]));
  }
  return ModesNode::make(dimensions, dim_widths);
}

const Array<PrimExpr> ModesNode::get_dense_shape() const {
  Array<PrimExpr> dense_shape;
  for (auto fun : dim_widths) {
    dense_shape.push_back(fun->range->min + fun->range->extent);
  }
  return dense_shape;
}

const bool ModesNode::is_ragged() const {
  for (auto fun : dim_widths) {
    if (fun->arity() > 0) return true;
  }
  return false;
}

TVM_REGISTER_NODE_TYPE(ModesNode);
TVM_REGISTER_GLOBAL("tir.Modes")
    .set_body_typed([](Array<tvm::te::Dimension> dimensions, Array<UninterpFun> dim_widths) {
      return ModesNode::make(dimensions, dim_widths);
    });
}  // namespace tir
}  // namespace tvm
