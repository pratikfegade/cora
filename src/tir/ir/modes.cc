#include <tvm/arith/int_set.h>
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/cache_info.h>
#include <tvm/te/dimension.h>
#include <tvm/tir/expr_equality.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/uf_equality.h>
#include <tvm/tir/uninterp_fun.h>

#include <algorithm>
#include <iterator>
#include <set>
#include <vector>

#include "../../arith/interval_set.h"
#include "../../arith/projection_set.h"
#include "var_replacer.h"

namespace tvm {
namespace tir {

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ModesNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ModesNode*>(node.get());
      p->stream << "Modes(" << op->str() << ", " << op << ")";
    });

Modes ModesNode::make(Array<tvm::te::Dimension> dimensions, Array<PrimExpr> dim_widths,
                      Array<UninterpFun> dim_width_ufs, Array<UninterpFun> dim_aggregate_ufs,
                      bool loop_layout) {
  // std::cout << "[MN] Creating modes " << dimensions.size() << " " << dim_widths.size() << " "
  // << dim_width_ufs.size() << " " << dim_aggregate_ufs.size() << std::endl;
  size_t ndim = dimensions.size();
  CHECK(ndim > 0);

  if (!loop_layout) {
    CHECK_EQ(dim_width_ufs.size(), dim_aggregate_ufs.size()) << loop_layout;
  }
  CHECK(dim_width_ufs.size() == 0 || dim_widths.size() == dim_width_ufs.size());

  if (dim_widths.size() > 0 && dim_width_ufs.size() == 0 && dim_aggregate_ufs.size() == 0) {
    CHECK(dim_widths.size() == ndim);
    for (size_t i = 0; i < dim_widths.size(); ++i) {
      dim_width_ufs.push_back(
          UninterpFunNode::from_constant(dimensions[i]->name + "_w", dim_widths[i]));
    }
  }

  // if (dim_width_ufs.size() != dim_aggregate_ufs.size()) {
  //   std::cout << " " << std::endl;
  // }

  ObjectPtr<ModesNode> n = make_object<ModesNode>();
  n->dimensions = dimensions;
  n->dim_widths = dim_width_ufs;
  n->dim_aggregates = dim_aggregate_ufs;
  n->loop_layout = loop_layout;

  bool dense = true;
  CHECK_GT(n->dim_widths.size(), 0);
  for (auto fun : dim_width_ufs) {
    if (fun->arity() > 0) {
      dense = false;
      break;
    }
  }

  CHECK(loop_layout || dense || (n->dim_widths.size(), n->dim_aggregates.size()));

  auto ret = Modes(n);
  // std::cout << "[MN]  Created " << ret << std::endl;
  return ret;
}

Modes ModesNode::make(std::string name, Array<PrimExpr> dense_shape) {
  // std::cerr << "[MODES] Modes object created without dimensions for " << name << std::endl;
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

const Array<Dimension> ModesNode::get_dependent_dimensions(Dimension dim) const {
  return (dim_widths[dimensions.GetIdx(dim)]->dimensions);
}

const std::string ModesNode::str() const {
  std::string str = "";
  for (size_t i = 0; i < ndim(); ++i) {
    str += is_ragged(i) ? "R" : "D";
  }
  return str;
}

const PrimExpr ModesNode::ComputePositionTaco(std::string name, Array<Dimension> relevant_dims,
                                              Array<PrimExpr> coords) const {
  CHECK_LE(relevant_dims.size(), ndim());
  CHECK_EQ(relevant_dims.size(), coords.size());

  int start = 0;
  int end = ndim();
  bool started = false;
  for (size_t i = 0; i < ndim(); ++i) {
    Dimension dim = dimensions[i];
    if (!started && dim == relevant_dims[0]) {
      started = true;
      start = i;
    } else if (!started && dim != relevant_dims[0]) {
      continue;
    } else if (relevant_dims.size() == i - start) {
      end = i;
      break;
    } else {
      CHECK(dim == relevant_dims[i - start]);
    }
  }

  CHECK_EQ(end - start, relevant_dims.size());

  auto get_width = [&](int i) {
    CHECK(!is_ragged(i));
    return dim_widths[i]->range->min + dim_widths[i]->range->extent - 1;
  };

  PrimExpr offset = 0;
  for (size_t i = 0; i < relevant_dims.size(); ++i) {
    Dimension dim = relevant_dims[i];
    PrimExpr coord = coords[i];
    int idx = i + start;

    if (is_ragged(i)) {
      Array<PrimExpr> args;
      args.push_back(offset);
      args.push_back_all(coords);
      offset = UninterpFun::MakeCallTo(dim_aggregates[i], args, {});
    } else {
      offset = offset * get_width(i) + coord;
    }
  }

  return offset;
}

const PrimExpr ModesNode::ComputePositionTaco(std::string name, Array<PrimExpr> coords) const {
  return ComputePositionTaco(name, dimensions, coords);
}

const PrimExpr ModesNode::ComputePosition(std::string name, Array<PrimExpr> coords) const {
  return ComputePosition(name, coords, dimensions);
}

const PrimExpr ModesNode::ComputePosition(std::string name, Array<PrimExpr> coords,
                                          Array<Dimension> relevant_dims) const {
  // if (!loop_layout) {
  // std::cout << " " << std::endl;
  // }

  // bool print = false;
  bool print = (name == "mummy");
  if (print) std::cout << "[CP] For " << name << " " << coords.size() << std::endl;

  // Map from an outer dimension Do to the outermost inner dimension
  // Di such that Di depends on Do and Do is outer to Di
  std::unordered_map<int, std::vector<int>> outer_to_inner_deps;
  std::unordered_map<int, std::vector<int>> inner_to_outer_deps;
  for (size_t i = 0; i < ndim(); ++i) {
    Dimension dimo = dimensions[i];
    {
      std::vector<int> dependent_dims;
      for (size_t j = i + 1; j < ndim(); ++j) {
        // std::cout << "[CP]  " << i << " " << j << std::endl;
        UninterpFun l_uf = dim_widths[j];
        if (l_uf.defined() && l_uf->dimensions.Contains(dimo)) {
          dependent_dims.push_back(j);
        }
      }
      outer_to_inner_deps[i] = dependent_dims;
    }
    {
      std::vector<int> inv_dependent_dims;
      UninterpFun l_uf = dim_widths[i];
      if (l_uf.defined()) {
        for (Dimension dim : l_uf->dimensions) {
          inv_dependent_dims.push_back(dimensions.GetIdx(dim));
        }
      }
      inner_to_outer_deps[i] = inv_dependent_dims;
    }
  }

  auto get_outermost_dependent_dimension = [&](int i) {
    CHECK(outer_to_inner_deps.count(i));
    auto dependent = outer_to_inner_deps[i];
    if (dependent.size() > 0)
      return dependent[0];
    else
      return static_cast<int>(ndim());
  };

  auto get_width = [&](int i) {
    CHECK(!is_ragged(i));
    return dim_widths[i]->range->min + dim_widths[i]->range->extent - 1;
  };

  auto get_ragged_contribution = [&](int i, std::set<int> processed, int processing) {
    CHECK(is_ragged(i));
    std::vector<int> outer_dependent_dims_vec = inner_to_outer_deps[i];
    std::set<int> outer_dependent_dims(outer_dependent_dims_vec.begin(),
                                       outer_dependent_dims_vec.end());
    std::set<int> processed_dependent_dims;
    std::set_intersection(
        outer_dependent_dims.begin(), outer_dependent_dims.end(), processed.begin(),
        processed.end(), std::inserter(processed_dependent_dims, processed_dependent_dims.begin()));

    if (processed_dependent_dims.size() == 0 && !outer_dependent_dims.count(processing)) {
      return CallNode::make(DataType::Int(32), dim_widths[i]->fname, coords,
                            CallNode::CallType::UninterpFunCall, relevant_dims, dim_widths[i], 0);
    } else if (processed_dependent_dims.size() == 0 && outer_dependent_dims.count(processing)) {
      return CallNode::make(DataType::Int(32), dim_aggregates[i]->fname, coords,
                            CallNode::CallType::UninterpFunCall, relevant_dims, dim_aggregates[i],
                            0);
    } else {
      Array<PrimExpr> args;
      CHECK(dim_aggregates[i].defined());

      // std::cout << "[CP] FUN " << dim_aggregates[i]->fname << std::endl;

      for (Dimension in_dim : dim_aggregates[i]->dimensions) {
        int dim_idx = dimensions.GetIdx(in_dim);
        if (processed_dependent_dims.count(dim_idx)) {
          CHECK(!is_ragged(dim_idx));
          args.push_back(get_width(dim_idx));
        } else {
          args.push_back(coords[dim_idx]);
        }
      }

      // for (size_t i = 0; i < dim_aggregates[i]->arity(); ++i) {
      //   if (processed_dependent_dims.count(i)) {
      //     CHECK(!is_ragged(i));
      //     args.push_back(get_width(i));
      //   } else {
      //     args.push_back(coords[i]);
      //   }
      // }
      return CallNode::make(DataType::Int(32), dim_aggregates[i]->fname, args,
                            CallNode::CallType::UninterpFunCall, dim_aggregates[i]->dimensions,
                            dim_aggregates[i], 0);
    }
  };

  if (coords.size() == 0) {
    CHECK_EQ(relevant_dims.size(), 0);
    std::set<int> processed;
    for (int i = 0; i < ndim(); ++i) {
      processed.insert(i);
    }

    std::vector<bool> handled_already;
    for (int i = 0; i < ndim(); ++i) {
      handled_already.push_back(false);
    }

    PrimExpr size = 1;
    for (int i = 0; i < ndim(); ++i) {
      int outermost_dependent_dimension = get_outermost_dependent_dimension(i);
      if (outermost_dependent_dimension == ndim()) {
        if (is_ragged(i)) {
          CHECK(handled_already[i]);
        } else {
          size = size * get_width(i);
        }
      } else {
        size = size * get_ragged_contribution(outermost_dependent_dimension, processed, i);
        for (auto dependent_dimension : outer_to_inner_deps[i]) {
          CHECK(is_ragged(dependent_dimension));
          handled_already[dependent_dimension] = true;
        }
      }
    }
    return UninterpFun::InlineUninterpFunCalls(size);
    // return size;
  } else {
    int num_dims = relevant_dims.size();
    PrimExpr offset = 0;
    std::set<int> processed;
    for (int i = num_dims - 1; i >= 0; --i) {
      Dimension i_dim = relevant_dims[i];
      int i_idx = dimensions.GetIdx(i_dim);
      // if (print) std::cout << "[CP]  Dim " << i << std::endl;
      PrimExpr this_offset = NullValue<PrimExpr>();

      int outermost_dependent_dimension = get_outermost_dependent_dimension(i_idx);

      std::vector<bool> handled_already;
      for (int j = 0; j < ndim(); ++j) {
        handled_already.push_back(false);
      }

      if (outermost_dependent_dimension == ndim()) {
        this_offset = coords[i];
      } else {
        this_offset = get_ragged_contribution(outermost_dependent_dimension, processed, i_idx);
        for (auto dependent_dimension : outer_to_inner_deps[i_idx]) {
          CHECK(is_ragged(dependent_dimension));
          handled_already[dependent_dimension] = true;
        }
      }
      for (int j = i + 1; j < num_dims; ++j) {
        Dimension j_dim = relevant_dims[j];
        int j_idx = dimensions.GetIdx(j_dim);
        if (is_ragged(j_idx)) {
          if (handled_already[j_idx]) {
            continue;
          } else {
            this_offset = this_offset * get_ragged_contribution(j_idx, processed, i_idx);
          }
        } else {
          int outermost_dependent_dimension = get_outermost_dependent_dimension(j_idx);
          if (outermost_dependent_dimension == ndim()) {
            this_offset = this_offset * get_width(j_idx);
          } else {
            this_offset = this_offset *
                          get_ragged_contribution(outermost_dependent_dimension, processed, i_idx);
            for (auto dependent_dimension : outer_to_inner_deps[j_idx]) {
              CHECK(is_ragged(dependent_dimension));
              handled_already[dependent_dimension] = true;
            }
          }
        }
      }

      if (print) std::cout << "[CP]   " << offset << " " << this_offset << std::endl;
      offset = offset + this_offset;
      processed.insert(i_idx);
    }
    return UninterpFun::InlineUninterpFunCalls(offset);
    // return offset;
  }
}

const PrimExpr ModesNode::GetAllocationSize() const {
  PrimExpr size = ComputePosition("Alloc", {}, {});
  if (is_ragged()) {
    std::cout << "[GAS] " << GetRef<Modes>(this) << " " << size << std::endl;
  }
  return size;
}

TVM_REGISTER_NODE_TYPE(ModesNode);
TVM_REGISTER_GLOBAL("tir.Modes")
    .set_body_typed([](Array<tvm::te::Dimension> dimensions, Array<PrimExpr> dim_widths,
                       Array<UninterpFun> dim_width_ufs, Array<UninterpFun> dim_aggregate_ufs,
                       bool loop_layout) {
      return ModesNode::make(dimensions, dim_widths, dim_width_ufs, dim_aggregate_ufs, loop_layout);
    });
}  // namespace tir
}  // namespace tvm
