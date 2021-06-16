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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ModesNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ModesNode*>(node.get());
      p->stream << "Modes(" << op << ")";
    });

Modes ModesNode::make(Array<tvm::te::Dimension> dimensions, Array<PrimExpr> dim_widths,
                      Array<UninterpFun> dim_width_ufs, Array<UninterpFun> dim_position_ufs) {
  int ndim = dimensions.size();
  CHECK(ndim > 0);

  CHECK(dim_widths.size() > 0 || dim_width_ufs.size() > 0 || dim_position_ufs.size() > 0);

  if (dim_widths.size() > 0 && dim_width_ufs.size() == 0 && dim_position_ufs.size() == 0) {
    CHECK(dim_widths.size() == ndim);
    for (size_t i = 0; i < dim_widths.size(); ++i) {
      dim_width_ufs.push_back(
          UninterpFunNode::from_constant(dimensions[i]->name + "_w", dim_widths[i]));
    }
  } else if (dim_widths.size() > 0 && dim_width_ufs.size() == 0 && dim_position_ufs.size() > 0) {
    CHECK(dim_position_ufs.size() == ndim);
    for (size_t i = 0; i < dim_widths.size(); ++i) {
      auto pos_uf = dim_position_ufs[i];
      Var innermost_dim_var = NullValue<Var>();
      int dim_min_pos = 1000;
      for (size_t j = 0; j < pos_uf->arity(); ++j) {
        auto dim = pos_uf->dimensions[j];
        int current_pos = dimensions.GetIdx(dim);
        if (current_pos < dim_min_pos) {
          dim_min_pos = current_pos;
          innermost_dim_var = pos_uf->parameters[j];
        }
      }

      PrimExpr body = NullValue<PrimExpr>();
      {
        std::unordered_map<const VarNode*, PrimExpr> vsub;
        vsub[innermost_dim_var.as<VarNode>()] = innermost_dim_var + 1;
        PrimExpr bodyp1 = VarReplacer(vsub)(pos_uf->body);
        body = bodyp1 - pos_uf->body;
      }

      dim_width_ufs.push_back(UninterpFunNode::make(dimensions[i]->name + "_w",
                                                    Range::make_by_min_extent(0, dim_widths[i]),
                                                    pos_uf->dimensions, pos_uf->parameters, body));
    }
  }

  ObjectPtr<ModesNode> n = make_object<ModesNode>();
  n->dimensions = dimensions;
  n->dim_widths = dim_width_ufs;
  n->dim_positions = dim_position_ufs;
  return Modes(n);
}

Modes ModesNode::make(std::string name, Array<PrimExpr> dense_shape) {
  std::cerr << "[MODES] Modes object created without dimensions for " << name << std::endl;
  Array<Dimension> dimensions;
  for (size_t i = 0; i < dense_shape.size(); ++i) {
    dimensions.push_back(te::DimensionNode::make("mode_dim_" + std::to_string(i),
                                                 te::DimensionNode::DimensionType::kRangeDim));
  }
  return ModesNode::make(dimensions, dense_shape, {}, {});
}

const Array<PrimExpr> ModesNode::get_dense_shape() const {
  Array<PrimExpr> dense_shape;
  for (auto fun : dim_widths) {
    dense_shape.push_back(fun->range->min + fun->range->extent - 1);
  }
  return dense_shape;
}

const bool ModesNode::is_ragged() const {
  for (auto fun : dim_widths) {
    if (fun->arity() > 0) return true;
  }
  return false;
}

const bool ModesNode::is_ragged(int i) const { return (dim_widths[i]->arity() > 0); }

const PrimExpr ModesNode::ComputePosition(Array<PrimExpr> coords) const {
  CHECK_EQ(coords.size(), ndim());

  // Map from an outer dimension Do to the outermost inner dimension
  // Di such that Di depends on Do and Do is outer to Di
  std::unordered_map<int, int> outer_to_inner_deps;

  for (size_t i = 0; i < ndim(); ++i) {
    Dimension dimo = dimensions[i];
    size_t j = 0;
    for (j = i + 1; j < ndim(); ++j) {
      UninterpFun pos_uf = dim_positions[j];
      if (pos_uf->dimensions.Contains(dimo)) {
        break;
      }
    }
    outer_to_inner_deps[i] = j;
  }

  PrimExpr offset = 0;
  Array<PrimExpr> args(coords);
  for (int i = ndim() - 1; i >= 0; --i) {
    std::cout << "[CP]  " << i << std::endl;
    Dimension dim = dimensions[i];
    int dim_inner_idx = outer_to_inner_deps.at(i);
    PrimExpr this_offset = 1;
    if (dim_inner_idx < ndim()) {
      auto inner_uf = dim_positions[dim_inner_idx];
      this_offset = CallNode::make(inner_uf->body.dtype(), inner_uf->fname, args,
                                   CallNode::CallType::UninterpFunCall, dimensions, inner_uf, 0);
      for (size_t j = i + 1; j < ndim(); ++j) {
        if (is_ragged(j)) {
          continue;
        } else {
          this_offset = this_offset * dim_widths[i]->range->extent;
        }
      }
    } else {
      this_offset = coords[i];
      for (size_t j = i + 1; j < ndim(); ++j) {
        this_offset = this_offset * dim_widths[i]->range->extent;
      }
    }

    args[i] = offset = offset + this_offset;
  }
  // return UninterpFun::InlineUninterpFunCalls(offset);
  return offset;
}

TVM_REGISTER_NODE_TYPE(ModesNode);
TVM_REGISTER_GLOBAL("tir.Modes")
    .set_body_typed([](Array<tvm::te::Dimension> dimensions, Array<PrimExpr> dim_widths,
                       Array<UninterpFun> dim_width_ufs, Array<UninterpFun> dim_position_ufs) {
      return ModesNode::make(dimensions, dim_widths, dim_width_ufs, dim_position_ufs);
    });
}  // namespace tir
}  // namespace tvm
