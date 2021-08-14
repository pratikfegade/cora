#include "ragged_utils.h"

#include <tvm/ir/attrs.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_equality.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/modes.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_set>

namespace tvm {
namespace te {

bool verify_itervar_order(const Stage& stage, const Array<IterVar>& order) {
  bool print = false;  // stage->op->name == "O.local";
  if (print) std::cout << "[VIO] For stage " << stage << std::endl;
  Map<IterVar, Array<IterVar>> root_var_deps;

  Array<IterVar> root_vars = stage->op->root_iter_vars();

  std::unordered_map<IterVar, Range> range_state;
  for (auto iv : root_vars) {
    if (print) std::cout << "[VIO]   Inserting RV " << iv << std::endl;
    range_state[iv] = iv->dom;
  }

  arith::Analyzer analyzer;
  PassDownDomain(stage, &range_state, &analyzer, true);

  std::unordered_map<IterVar, int> bit_state;
  for (size_t i = 0; i < order.size(); ++i) {
    bit_state[order[i]] = 1 << i;
  }
  PassUpBitMaskOr(stage, &bit_state, true);

  VarCollector var_collector;
  for (size_t i = 0; i < order.size(); ++i) {
    auto iv = order[i];
    if (print)
      std::cout << "[VIO]   Inferred range " << iv << " " << range_state.at(iv) << std::endl;
    CHECK(range_state.count(iv)) << iv;
    std::unordered_set<const VarNode*> vars_needed =
        var_collector.collect(UninterpFun::InlineUninterpFunCalls(range_state.at(iv)));

    for (size_t j = i + 1; j < order.size(); ++j) {
      for (auto root_iv : root_vars) {
        if (bit_state.count(root_iv) && (bit_state.at(root_iv) & (1 << j)) != 0) {
          if (vars_needed.count(root_iv->var.as<VarNode>())) return false;
        }
      }
    }
  }

  return true;
}

bool verify_dimension_order(const Stage& stage, const Array<Dimension>& order) {
  Map<Dimension, Array<Dimension>> root_dim_deps;

  const BaseVarDimOpNode* op_node = stage->op.as<BaseVarDimOpNode>();
  CHECK(op_node);
  Array<Dimension> root_dims = stage->dim_relation_graph->root_dimensions;

  std::unordered_map<const DimensionNode*, Range> range_state;
  for (auto dim : root_dims) {
    range_state[dim.as<DimensionNode>()] = op_node->GetIterVarFromDim(0, dim)->dom;
  }

  arith::Analyzer analyzer;
  DimensionPassDownDomain(stage, op_node, &range_state, false);

  std::unordered_map<const DimensionNode*, int> bit_state;
  for (size_t i = 0; i < order.size(); ++i) {
    bit_state[order[i].as<DimensionNode>()] = 1 << i;
  }
  DimensionPassUpBitMaskOr(stage, &bit_state, true);

  VarCollector var_collector;
  for (size_t i = 0; i < order.size(); ++i) {
    auto dim = order[i];
    CHECK(range_state.count(dim.operator->()));
    std::unordered_set<const VarNode*> vars_needed = var_collector.collect(
        UninterpFun::InlineUninterpFunCalls(range_state.at(dim.operator->())));

    for (size_t j = i + 1; j < order.size(); ++j) {
      for (auto root_dim : root_dims) {
        auto root_iv = op_node->GetIterVarFromDim(0, root_dim);
        if (bit_state.count(root_dim.operator->()) &&
            (bit_state.at(root_dim.operator->()) & (1 << j)) != 0) {
          if (vars_needed.count(root_iv->var.as<VarNode>())) return false;
        }
      }
    }
  }

  return true;
}

PrimExpr zero_if_args_zero_ufun_call(DataType dtype, Array<PrimExpr> args, Array<Dimension> dims,
                                     UninterpFun func) {
  Array<PrimExpr> compressed_args;
  Array<Dimension> compressed_dims;

  for (size_t i = 0; i < dims.size(); ++i) {
    if (func->dimensions.Contains(dims[i])) {
      compressed_args.push_back(args[i]);
      compressed_dims.push_back(dims[i]);
    }
  }

  bool args_zero = true;
  for (auto arg : compressed_args) {
    if (!is_zero(arg)) {
      args_zero = false;
      break;
    }
  }

  if (args_zero) {
    return IntImm(dtype, 0);
  } else {
    return func.MakeCallTo(compressed_args, compressed_dims, dtype);
  }
}

IntSet RelaxFusionFunction(IntSet fused, UninterpFun func, const Map<Var, IntSet>& dom_map,
                           const std::unordered_map<IterVar, Range>* bound_dom_map) {
  bool print = false;
  UninterpFun fo_fun;
  UninterpFun fi_fun;
  if (func->type == UninterpFunNode::kFOFun) {
    fo_fun = Downcast<UninterpFun>(func);
    fi_fun = Downcast<UninterpFun>(fo_fun->fusion_info->fused_to_inner_uf);
  } else {
    fi_fun = Downcast<UninterpFun>(func);
    fo_fun = Downcast<UninterpFun>(fi_fun->fusion_info->fused_to_outer_uf);
  }

  PrimExpr fused_min = fused.min();
  PrimExpr fused_max_inclusive = fused.max();

  if (print) {
    std::cout << "[RFF] Evaling " << func << std::endl;
    std::cout << "[RFF]  Fused Min " << fused_min << std::endl;
    std::cout << "[RFF]  Fused Max " << fused_max_inclusive << std::endl;
    // std::cout << "[RFF]  Outer " << fused << std::endl;
    // std::cout << "[RFF]  Inner " << fused << std::endl;
  }
  auto dtype = DataType::Int(32);
  PrimExpr outer_min = zero_if_args_zero_ufun_call(dtype, {fused_min}, fo_fun->dimensions, fo_fun);
  PrimExpr outer_max_inclusive =
      zero_if_args_zero_ufun_call(dtype, {fused_max_inclusive}, fo_fun->dimensions, fo_fun);

  if (func->type == UninterpFunNode::kFOFun) {
    auto ret = IntSet::interval(outer_min, outer_max_inclusive);
    if (print) {
      // std::cout << "[RFF]   Retting " << ret << std::endl;
    }
    return ret;
  } else {
    PrimExpr inner_min_boundary =
        zero_if_args_zero_ufun_call(dtype, {fused_min}, fi_fun->dimensions, fi_fun);

    PrimExpr inner_max_inclusive_boundary =
        zero_if_args_zero_ufun_call(dtype, {fused_max_inclusive}, fi_fun->dimensions, fi_fun);

    if (bound_dom_map) {
      CHECK(bound_dom_map->count(fo_fun->fusion_info->inner));
      Range inner_range = bound_dom_map->at(fo_fun->fusion_info->inner);
      IntSet inner_range_relaxed = EvalSet(inner_range, dom_map, bound_dom_map);

      auto ret = IntSet::interval(
          SelectNode::make(EQNode::make(fo_fun->fusion_info->outer, outer_min), inner_min_boundary,
                           inner_range_relaxed.min()),
          SelectNode::make(EQNode::make(fo_fun->fusion_info->outer, outer_max_inclusive),
                           inner_max_inclusive_boundary, inner_range_relaxed.max()));
      if (print) {
        std::cout << "[RFF]  Inner Range " << inner_range << std::endl;
        std::cout << "[RFF]   Retting " << ret << std::endl;
      }
      return ret;
    } else {
      if (print) {
        std::cout << "[RFF]   Retting Everything" << std::endl;
      }
      return arith::IntSet::everything();
    }
  }
}

}  // namespace te
}  // namespace tvm
