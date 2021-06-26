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
#include <unordered_set>
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

Modes ModesNode::make(Array<tvm::te::Dimension> dimensions, Array<PrimExpr> l_maxes,
                      Array<UninterpFun> l_funs, Array<UninterpFun> a_funs, bool loop_layout) {
  // std::cout << "[MN] Creating modes " << dimensions.size() << " " << l_maxes.size() << " "
  // << l_funs.size() << " " << a_funs.size() << std::endl;
  size_t ndim = dimensions.size();
  CHECK(ndim > 0);

  if (!loop_layout) {
    CHECK_EQ(l_funs.size(), a_funs.size()) << loop_layout;
  }
  CHECK(l_funs.size() == 0 || l_maxes.size() == l_funs.size());

  if (l_maxes.size() > 0 && l_funs.size() == 0 && a_funs.size() == 0) {
    CHECK(l_maxes.size() == ndim);
    for (size_t i = 0; i < l_maxes.size(); ++i) {
      l_funs.push_back(UninterpFunNode::from_constant(dimensions[i]->name + "_w", l_maxes[i]));
    }
  }

  // if (l_funs.size() != a_funs.size()) {
  //   std::cout << " " << std::endl;
  // }

  ObjectPtr<ModesNode> n = make_object<ModesNode>();
  n->dimensions = dimensions;
  n->l_funs = l_funs;
  n->a_funs = a_funs;
  n->loop_layout = loop_layout;

  bool dense = true;
  CHECK_GT(n->l_funs.size(), 0);
  for (auto fun : l_funs) {
    CHECK_LE(fun->arity(), 1) << "We only support ragged dimensions dependent on one other "
                                 "dimension.";
    if (fun->arity() > 0) {
      dense = false;
      break;
    }
  }

  CHECK(loop_layout || dense || (n->l_funs.size(), n->a_funs.size()));

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
  for (auto fun : l_funs) {
    dense_shape.push_back(fun->range->min + fun->range->extent - 1);
  }
  return dense_shape;
}

const bool ModesNode::is_ragged() const {
  for (auto fun : l_funs) {
    if (fun->arity() > 0) return true;
  }
  return false;
}

const bool ModesNode::is_ragged(int i) const { return (l_funs[i]->arity() > 0); }

const Array<Dimension> ModesNode::get_dependent_dimensions(Dimension dim) const {
  return (l_funs[dimensions.GetIdx(dim)]->dimensions);
}

const std::string ModesNode::str() const {
  std::string str = "";
  for (size_t i = 0; i < ndim(); ++i) {
    str += is_ragged(i) ? "R" : "D";
  }
  return str;
}

const void ModesNode::SetupTransitiveDependences() const {
  if (transitive_dependent_dims.size() > 0) return;

  std::unordered_map<int, std::vector<int>> temp_map;
  for (size_t i = 0; i < ndim(); ++i) {
    Dimension dim = dimensions[i];
    Array<Dimension> needed_dims = l_funs[i]->dimensions;
    if (needed_dims.size() == 0) {
      continue;
    }
    CHECK_EQ(needed_dims.size(), 1)
        << "We only support ragged dimensions dependent on one other dimension.";
    int needed_dim_idx = dimensions.GetIdx(needed_dims[0]);
    if (temp_map.count(needed_dim_idx)) {
      temp_map[needed_dim_idx].push_back(i);
    } else {
      temp_map[needed_dim_idx] = {i};
    }
  }

  for (auto it : temp_map) {
    Array<Dimension> dependent_dims;
    for (auto idx : it.second) {
      dependent_dims.push_back(dimensions[idx]);
    }
    transitive_dependent_dims.Set(dimensions[it.first], dependent_dims);
  }
}

const PrimExpr ModesNode::ComputePositionOld(std::string name, Array<PrimExpr> coords) const {
  return ComputePosition(name, coords, dimensions);
}

const PrimExpr ComputeTExpr(const ModesNode* self, int dim_idx, Array<PrimExpr> relaxed_coords,
                            bool print) {
  auto has_dependent_dims = [&](int idx) {
    self->SetupTransitiveDependences();
    return self->transitive_dependent_dims.count(self->dimensions[idx]) &&
           self->transitive_dependent_dims.at(self->dimensions[idx]).size() > 0;
  };

  auto get_transitive_dependent_dims = [&](int idx) {
    self->SetupTransitiveDependences();
    CHECK(self->transitive_dependent_dims.count(self->dimensions[idx]));
    return self->transitive_dependent_dims.at(self->dimensions[idx]);
  };

  Dimension dim = self->dimensions[dim_idx];
  if (print) std::cout << "[CP]  iDim " << dim << std::endl;

  PrimExpr t_expr = 1;
  std::unordered_set<const Object*> handled_already;
  if (has_dependent_dims(dim_idx)) {
    CHECK(self->a_funs[dim_idx].defined()) << dim_idx << " " << self->dimensions[dim_idx];
    t_expr = UninterpFun::MakeCallTo(self->a_funs[dim_idx], Array<PrimExpr>(relaxed_coords),
                                     self->dimensions);
    for (auto dim : get_transitive_dependent_dims(dim_idx)) {
      handled_already.insert(dim.get());
    }
  } else {
    t_expr = relaxed_coords[dim_idx];
  }
  if (print) std::cout << "[CP]     t_expr update " << t_expr << std::endl;

  for (int j = dim_idx + 1; j < self->ndim(); ++j) {
    if (print) std::cout << "[CP]   jDim " << self->dimensions[j] << std::endl;
    if (handled_already.count(self->dimensions[j].get())) {
      continue;
    }

    if (has_dependent_dims(j)) {
      CHECK(self->a_funs[j].defined());
      t_expr = t_expr * UninterpFun::MakeCallTo(self->a_funs[j], Array<PrimExpr>(relaxed_coords),
                                                self->dimensions);
      for (auto dim : get_transitive_dependent_dims(j)) {
        handled_already.insert(dim.get());
      }
    } else {
      CHECK(self->l_funs[j].defined());
      t_expr = t_expr * UninterpFun::MakeCallTo(self->l_funs[j], Array<PrimExpr>(relaxed_coords),
                                                self->dimensions);
    }
    if (print) std::cout << "[CP]     t_expr update " << t_expr << std::endl;
  }

  return t_expr;
}

const PrimExpr ModesNode::ComputePosition(std::string name, Array<PrimExpr> coords) const {
  bool print = false;  //(name == "A");

  std::cout << "[CP] For " << name << std::endl;
  PrimExpr lowered_offset = 0;

  std::vector<PrimExpr> relaxed_coords;
  for (auto coord : coords) {
    relaxed_coords.push_back(coord);
  }
  for (int i = ndim() - 1; i >= 0; --i) {
    PrimExpr t_expr = ComputeTExpr(this, i, relaxed_coords, print);
    lowered_offset = lowered_offset + t_expr;
    if (print) std::cout << "[CP]   loffset update " << lowered_offset << std::endl;
    relaxed_coords[i] =
        UninterpFun::MakeCallTo(l_funs[i], Array<PrimExpr>(relaxed_coords), dimensions);
  }

  return UninterpFun::InlineUninterpFunCalls(lowered_offset);
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
        UninterpFun l_uf = l_funs[j];
        if (l_uf.defined() && l_uf->dimensions.Contains(dimo)) {
          dependent_dims.push_back(j);
        }
      }
      outer_to_inner_deps[i] = dependent_dims;
    }
    {
      std::vector<int> inv_dependent_dims;
      UninterpFun l_uf = l_funs[i];
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
    return l_funs[i]->range->min + l_funs[i]->range->extent - 1;
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
      return CallNode::make(DataType::Int(32), l_funs[i]->fname, coords,
                            CallNode::CallType::UninterpFunCall, relevant_dims, l_funs[i], 0);
    } else if (processed_dependent_dims.size() == 0 && outer_dependent_dims.count(processing)) {
      return CallNode::make(DataType::Int(32), a_funs[i]->fname, coords,
                            CallNode::CallType::UninterpFunCall, relevant_dims, a_funs[i], 0);
    } else {
      Array<PrimExpr> args;
      CHECK(a_funs[i].defined());

      // std::cout << "[CP] FUN " << a_funs[i]->fname << std::endl;

      for (Dimension in_dim : a_funs[i]->dimensions) {
        int dim_idx = dimensions.GetIdx(in_dim);
        if (processed_dependent_dims.count(dim_idx)) {
          CHECK(!is_ragged(dim_idx));
          args.push_back(get_width(dim_idx));
        } else {
          args.push_back(coords[dim_idx]);
        }
      }

      // for (size_t i = 0; i < a_funs[i]->arity(); ++i) {
      //   if (processed_dependent_dims.count(i)) {
      //     CHECK(!is_ragged(i));
      //     args.push_back(get_width(i));
      //   } else {
      //     args.push_back(coords[i]);
      //   }
      // }
      return CallNode::make(DataType::Int(32), a_funs[i]->fname, args,
                            CallNode::CallType::UninterpFunCall, a_funs[i]->dimensions, a_funs[i],
                            0);
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
  Array<PrimExpr> l_maxes;
  Array<Dimension> dims;
  for (size_t i = 0; i < ndim(); ++i) {
    UninterpFun l_fun = l_funs[i];
    PrimExpr l_max = UninterpFun::MakeCallTo(l_fun, l_maxes, dims);
    l_maxes.push_back(l_max);
    dims.push_back(dimensions[i]);
  }

  return UninterpFun::InlineUninterpFunCalls(ComputeTExpr(this, 0, l_maxes, false));
}

TVM_REGISTER_NODE_TYPE(ModesNode);
TVM_REGISTER_GLOBAL("tir.Modes")
    .set_body_typed([](Array<tvm::te::Dimension> dimensions, Array<PrimExpr> l_maxes,
                       Array<UninterpFun> l_funs, Array<UninterpFun> a_funs, bool loop_layout) {
      return ModesNode::make(dimensions, l_maxes, l_funs, a_funs, loop_layout);
    });
}  // namespace tir
}  // namespace tvm
