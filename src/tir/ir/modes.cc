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
                      Array<UninterpFun> l_fun_mins, Array<UninterpFun> l_funs,
                      Array<UninterpFun> a_funs, bool is_loop_layout) {
  if ((l_fun_mins.size() != l_funs.size()) && is_loop_layout) {
    std::cout << " " << std::endl;
  }
  ObjectPtr<ModesNode> n = make_object<ModesNode>();
  n->dimensions = dimensions;
  n->l_maxes = l_maxes;
  n->l_funs = l_funs;
  n->l_fun_mins = l_fun_mins;
  n->a_funs = a_funs;
  n->loop_layout = is_loop_layout;
  auto ret = Modes(n);
  ret->setup_transitive_dependences();
  return ret;
}

Modes ModesNode::make(Array<tvm::te::Dimension> dimensions, Array<PrimExpr> l_maxes,
                      Array<UninterpFun> l_fun_mins_, Array<UninterpFun> l_funs_,
                      Map<Dimension, UninterpFun> user_a_funs, bool is_loop_layout) {
  size_t ndim = dimensions.size();
  CHECK(ndim > 0);

  CHECK(l_funs_.size() == 0 || l_maxes.size() == l_funs_.size());

  Array<UninterpFun> l_funs = l_funs_;
  if (l_maxes.size() > 0 && l_funs.size() == 0) {
    CHECK(l_maxes.size() == ndim);
    for (size_t i = 0; i < l_maxes.size(); ++i) {
      l_funs.push_back(UninterpFunNode::from_constant(dimensions[i]->name + "_w", l_maxes[i]));
    }
  }

  Array<UninterpFun> l_fun_mins;
  l_fun_mins = l_fun_mins_;
  if (l_maxes.size() > 0 && l_fun_mins.size() == 0) {
    CHECK(l_maxes.size() == ndim);
    for (size_t i = 0; i < l_maxes.size(); ++i) {
      l_fun_mins.push_back(UninterpFunNode::from_constant("z", 0));
    }
  }

  ObjectPtr<ModesNode> n = make_object<ModesNode>();
  n->dimensions = dimensions;
  n->l_funs = l_funs;
  n->l_fun_mins = l_fun_mins;
  n->l_maxes = l_maxes;
  n->loop_layout = is_loop_layout;

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

  auto ret = Modes(n);
  ret->setup_transitive_dependences();

  Array<UninterpFun> a_funs;
  for (size_t i = 0; i < dimensions.size(); ++i) {
    Dimension dim = dimensions[i];
    if (user_a_funs.count(dim)) {
      a_funs.push_back(user_a_funs.at(dim));
    } else if (ret->has_dependent_dims(i)) {
      PrimExpr a_fun_max_extent = 1;
      for (auto dependent_dim : ret->get_transitive_dependent_dims(i)) {
        int dependent_dim_idx = dimensions.GetIdx(dependent_dim);
        a_fun_max_extent = a_fun_max_extent * l_funs[dependent_dim_idx]->range->extent;
      }

      a_funs.push_back(UninterpFunNode::make(
          dim->name + "_afun", Range::make_by_min_extent(0, a_fun_max_extent), {dim},
          {Var("param", DataType::Int(32))}, NullValue<PrimExpr>(), UninterpFunNode::kAFun));
    } else {
      a_funs.push_back(NullValue<UninterpFun>());
    }
  }
  n->a_funs = a_funs;

  if ((ret->l_fun_mins.size() != ret->l_funs.size()) && ret->loop_layout) {
    std::cout << " " << std::endl;
  }

  return ret;
}

Modes ModesNode::make_loop_layout(Array<tvm::te::Dimension> dimensions, Array<PrimExpr> l_maxes,
                                  Array<UninterpFun> l_fun_mins, Array<UninterpFun> l_funs) {
  return ModesNode::make(dimensions, l_maxes, l_fun_mins, l_funs, Map<Dimension, UninterpFun>(),
                         true);
}

Modes ModesNode::make_storage_layout(Array<tvm::te::Dimension> dimensions, Array<PrimExpr> l_maxes,
                                     Array<UninterpFun> l_funs, Array<UninterpFun> a_funs) {
  Map<Dimension, UninterpFun> af_map;
  for (size_t i = 0; i < dimensions.size(); ++i) {
    if (a_funs[i].defined()) {
      af_map.Set(dimensions[i], a_funs[i]);
    }
  }
  return ModesNode::make(dimensions, l_maxes, {}, l_funs, af_map, false);
}

Modes ModesNode::make_storage_layout(Array<tvm::te::Dimension> dimensions, Array<PrimExpr> l_maxes,
                                     Array<UninterpFun> l_funs,
                                     Map<Dimension, UninterpFun> user_a_funs) {
  return ModesNode::make(dimensions, l_maxes, {}, l_funs, user_a_funs, false);
}

Modes ModesNode::make(std::string name, Array<PrimExpr> dense_shape, bool is_loop_layout) {
  Array<Dimension> dimensions;
  for (size_t i = 0; i < dense_shape.size(); ++i) {
    dimensions.push_back(te::DimensionNode::make("mode_dim_" + std::to_string(i),
                                                 te::DimensionNode::DimensionType::kRangeDim));
  }
  return ModesNode::make(dimensions, dense_shape, Array<UninterpFun>(), Array<UninterpFun>(),
                         Map<Dimension, UninterpFun>(), is_loop_layout);
}

const Array<PrimExpr> ModesNode::get_dense_shape() const {
  Array<PrimExpr> dense_shape;
  for (auto fun : l_funs) {
    dense_shape.push_back(fun->range->max_inclusive());
  }
  return dense_shape;
}

TVM_REGISTER_GLOBAL("tir.ModesDenseShape").set_body_typed([](Modes modes) {
  return modes->get_dense_shape();
});

const bool ModesNode::is_ragged() const {
  for (auto fun : l_funs) {
    if (fun->arity() > 0) return true;
  }
  return false;
}

TVM_REGISTER_GLOBAL("tir.ModesIsRagged").set_body_typed([](Modes modes) {
  return modes->is_ragged();
});

const bool ModesNode::is_ragged(int i) const { return (l_funs[i]->arity() > 0); }

const std::string ModesNode::str() const {
  std::string str = "";
  for (size_t i = 0; i < ndim(); ++i) {
    str += is_ragged(i) ? "R" : "D";
  }
  return str;
}

const bool ModesNode::has_dependent_dims(int idx) const {
  setup_transitive_dependences();
  return transitive_dependent_dims.count(dimensions[idx]) &&
         transitive_dependent_dims.at(dimensions[idx]).size() > 0;
}

const void ModesNode::setup_transitive_dependences() const {
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

  auto get_transitive_deps = [&](int idx) {
    Array<Dimension> immediate_deps;

    auto it = temp_map.find(idx);
    if (it != temp_map.end()) {
      for (auto dep : it->second) {
        immediate_deps.push_back(dimensions[dep]);
      }
    }

    std::vector<int> queue = {idx};
    Array<Dimension> transitive_deps;
    while (queue.size() > 0) {
      int current = queue.back();
      queue.pop_back();
      if (current != idx) {
        transitive_deps.push_back(dimensions[current]);
      }
      auto it = temp_map.find(current);
      if (it != temp_map.end()) {
        for (auto dep : it->second) {
          queue.push_back(dep);
        }
      }
    }
    return std::make_pair(immediate_deps, transitive_deps);
  };

  for (auto it : temp_map) {
    auto deps = get_transitive_deps(it.first);
    immediate_dependent_dims.Set(dimensions[it.first], deps.first);
    transitive_dependent_dims.Set(dimensions[it.first], deps.second);
  }
}

const Array<Dimension> ModesNode::get_transitive_dependent_dims(int idx) const {
  setup_transitive_dependences();
  CHECK(transitive_dependent_dims.count(dimensions[idx]));
  return transitive_dependent_dims.at(dimensions[idx]);
}

const Array<Dimension> ModesNode::get_immediate_dependent_dims(int idx) const {
  setup_transitive_dependences();
  CHECK(immediate_dependent_dims.count(dimensions[idx]));
  return immediate_dependent_dims.at(dimensions[idx]);
}

const PrimExpr ComputeTExpr(const ModesNode* self, int dim_idx, Array<PrimExpr> relaxed_coords,
                            bool print) {
  Dimension dim = self->dimensions[dim_idx];
  bool print2 = print && (dim_idx == 0);
  if (print2) std::cout << "[CP]  iDim " << dim << std::endl;

  PrimExpr t_expr = 1;
  std::unordered_set<const Object*> handled_already;
  if (self->has_dependent_dims(dim_idx)) {
    CHECK(self->a_funs[dim_idx].defined()) << dim_idx << " " << self->dimensions[dim_idx];
    t_expr = self->a_funs[dim_idx].MakeCallTo(Array<PrimExpr>(relaxed_coords), self->dimensions);
    if (print2) std::cout << "[CP]      Transitive dependent dims" << std::endl;
    for (auto dim : self->get_transitive_dependent_dims(dim_idx)) {
      if (print2) std::cout << "[CP]         " << dim << std::endl;
      handled_already.insert(dim.get());
    }
  } else {
    t_expr = relaxed_coords[dim_idx];
  }
  if (print2) std::cout << "[CP]     t_expr update " << t_expr << std::endl;

  for (int j = dim_idx + 1; j < self->ndim(); ++j) {
    if (print2) std::cout << "[CP]      jDim " << self->dimensions[j] << std::endl;
    if (handled_already.count(self->dimensions[j].get())) {
      if (print2) std::cout << "[CP]       Handled" << std::endl;
      continue;
    }

    if (self->has_dependent_dims(j)) {
      CHECK(self->a_funs[j].defined());
      t_expr =
          t_expr * self->a_funs[j].MakeCallTo(Array<PrimExpr>(relaxed_coords), self->dimensions);
      if (print2) std::cout << "[CP]      Transitive dependent dims" << std::endl;
      for (auto dim : self->get_transitive_dependent_dims(j)) {
        if (print2) std::cout << "[CP]         " << dim << std::endl;
        handled_already.insert(dim.get());
      }
    } else {
      CHECK(self->l_funs[j].defined());
      t_expr =
          t_expr * self->l_funs[j].MakeCallTo(Array<PrimExpr>(relaxed_coords), self->dimensions);
    }
    if (print2) std::cout << "[CP]     t_expr update " << t_expr << std::endl;
  }

  return t_expr;
}

const PrimExpr ModesNode::ComputePosition(std::string name, Array<PrimExpr> coords) const {
  bool print = false;  //(name == "O");

  if (print) {
    for (size_t i = 0; i < dimensions.size(); ++i) {
      std::cout << "[CP] " << dimensions[i] << " " << is_ragged(i) << " " << a_funs[i] << std::endl;
    }
  }

  // std::cout << "[CP] For " << name << std::endl;
  PrimExpr lowered_offset = 0;

  std::vector<PrimExpr> relaxed_coords;
  for (auto coord : coords) {
    relaxed_coords.push_back(coord);
  }
  for (int i = ndim() - 1; i >= 0; --i) {
    PrimExpr t_expr = ComputeTExpr(this, i, relaxed_coords, print);
    lowered_offset = lowered_offset + t_expr;
    if (print) std::cout << "[CP]   loffset update " << lowered_offset << std::endl;
    relaxed_coords[i] = l_funs[i].MakeCallTo(Array<PrimExpr>(relaxed_coords), dimensions);
  }

  return UninterpFun::InlineUninterpFunCalls(lowered_offset);
}

const PrimExpr ModesNode::ComputePosition(std::string name, Array<PrimExpr> coords,
                                          Array<Dimension> relevant_dims) const {
  bool print = false;
  // bool print = (name == "mummy");
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
    return l_funs[i]->range->max_inclusive();
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
      return l_funs[i].MakeCallTo(coords, relevant_dims);
    } else if (processed_dependent_dims.size() == 0 && outer_dependent_dims.count(processing)) {
      return a_funs[i].MakeCallTo(coords, relevant_dims);
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

      return a_funs[i].MakeCallTo(args, a_funs[i]->dimensions);
    }
  };

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
    for (size_t j = 0; j < ndim(); ++j) {
      handled_already.push_back(false);
    }

    if (outermost_dependent_dimension == static_cast<int>(ndim())) {
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
        if (outermost_dependent_dimension == static_cast<int>(ndim())) {
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
}

const PrimExpr ModesNode::GetAllocationSize() const {
  Array<PrimExpr> l_maxes;
  Array<Dimension> dims;
  for (size_t i = 0; i < ndim(); ++i) {
    UninterpFun l_fun = l_funs[i];
    PrimExpr l_max = l_fun.MakeCallTo(l_maxes, dims);
    l_maxes.push_back(l_max);
    dims.push_back(dimensions[i]);
  }

  return UninterpFun::InlineUninterpFunCalls(ComputeTExpr(this, 0, l_maxes, false));
}

TVM_REGISTER_NODE_TYPE(ModesNode);
TVM_REGISTER_GLOBAL("tir.StorageModes")
    .set_body_typed([](Array<tvm::te::Dimension> dimensions, Array<PrimExpr> l_maxes,
                       Array<UninterpFun> l_funs, Map<Dimension, UninterpFun> user_a_funs) {
      return ModesNode::make_storage_layout(dimensions, l_maxes, l_funs, user_a_funs);
    });

TVM_REGISTER_GLOBAL("tir.LoopModes")
    .set_body_typed([](Array<tvm::te::Dimension> dimensions, Array<PrimExpr> l_maxes,
                       Array<UninterpFun> l_fun_mins, Array<UninterpFun> l_funs) {
      return ModesNode::make_loop_layout(dimensions, l_maxes, l_fun_mins, l_funs);
    });
}  // namespace tir
}  // namespace tvm
