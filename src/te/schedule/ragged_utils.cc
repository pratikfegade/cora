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
  std::cout << "[VIO] For stage " << stage << std::endl;
  Map<IterVar, Array<IterVar>> root_var_deps;

  Array<IterVar> root_vars = stage->op->root_iter_vars();

  std::unordered_map<IterVar, Range> range_state;
  for (auto iv : root_vars) {
    std::cout << "[VIO]   Inserting RV " << iv << std::endl;
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
    CHECK(range_state.count(iv)) << iv;
    std::unordered_set<const VarNode*> vars_needed =
        var_collector.collect(UninterpFun::InlineUninterpFunCalls(range_state.at(iv)));

    for (size_t j = i + 1; j < order.size(); ++j) {
      auto leaf_iv = order[j];
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
      auto leaf_iv = order[j];
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

}  // namespace te
}  // namespace tvm
