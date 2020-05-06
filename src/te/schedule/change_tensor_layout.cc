#include <tvm/te/schedule.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/te/operation.h>
#include "graph.h"
#include "message_passing.h"
#include <algorithm>

namespace tvm {
namespace te {
class TensorLayoutFreezer: public StmtExprMutator {
  PrimExpr VisitExpr_(const CallNode* call) override {
    if (call->func.defined()) {
      Tensor tensor = Downcast<Operation>(call->func).output(call->value_index);
      auto compute_op = tensor->op.as<ComputeOpNode>();
      if (!compute_op) {
	// We only change layouts for results of ComputeOps
	return StmtExprMutator::VisitExpr_(call);
      }
      auto relations_graph = compute_op->dim_relation_graph;
      std::unordered_map<const DimensionNode*, PrimExpr> root_values;
      for (size_t i = 0; i < compute_op->root_index_dimensions.size(); ++i) {
	root_values[compute_op->root_index_dimensions[i].operator->()] = call->args[i];
      }

      DimensionPassDownValues(compute_op, dim_dom_map[compute_op], &root_values, true);

      Array<PrimExpr> args;
      for (auto dim: relations_graph->leaf_dimensions) {
	args.push_back(root_values[dim.operator->()]);
      }

      return CallNode::make(call->dtype, call->name, args, call->call_type,
			    call->argument_dimensions, call->func, call->value_index);
    }
    return StmtExprMutator::VisitExpr_(call);
  }

  ComputeOpNode* reader;
  std::unordered_map<const ComputeOpNode*, std::unordered_map<const DimensionNode*, Range>>& dim_dom_map;

public:
  TensorLayoutFreezer(ComputeOpNode* reader_,
		      std::unordered_map<const ComputeOpNode*, std::unordered_map<const DimensionNode*, Range>>& dim_dom_map_) :
    reader(reader_), dim_dom_map(dim_dom_map_) {}

  void replace() {
    // TODO(ppf): Handle reductions here
    Array<PrimExpr> body;
    for (auto e: reader->body) {
      PrimExpr replaced = this->VisitExpr(e);
      // std::cout << "[FTD] Body: " << e << std::endl;
      // std::cout << "[FTD] Replaced: " << replaced << std::endl;
      body.push_back(replaced);
    }

    reader->body = body;
  }
};

Map<Dimension, Range> GetIndexDimRangeFromLoopDimRange(const ComputeOpNode* compute_op, Map<IterVar, Range> dom_map) {
  Map<Dimension, Range> ret;
  for (const auto& root_dim: compute_op->root_index_dimensions) {
    if (root_dim->type <= DimensionNode::kRangeDim) {
      const auto& iv = compute_op->GetIterVarFromDim(0, root_dim);
      ret.Set(root_dim, dom_map.count(iv) ? dom_map.at(iv) : iv->dom);
    }
    else {
      UninterpFun ufun = compute_op->GetDimVarEntry(0, root_dim).value_expr;
      Array<PrimExpr> args;
      bool non_constant = false;
      CHECK(ufun->dimensions.defined());
      for (auto arg_dim: ufun->dimensions) {
	Range r = dom_map.at(compute_op->GetIterVarFromDim(0, arg_dim));
	if (!tir::is_one(r->extent)) {
	  non_constant = true;
	}
      }

      if (non_constant) {
	ret.Set(root_dim, ufun->range);
      }
      else {
	ret.Set(root_dim,
          Range::make_by_min_extent(
	    UninterpFun::MakeCallTo(ufun, args, compute_op->root_index_dimensions), 1));
      }
    }
  }

  return ret;
}

void Schedule::freeze_tensor_dimensions(Map<IterVar, Range> dom_map) {
  std::unordered_map<const ComputeOpNode*, std::unordered_map<const DimensionNode*, Range>> dim_doms;
  for (auto stage: (*this)->stages) {
    auto compute_op = stage->op.as<ComputeOpNode>();
    if (!compute_op) continue;
    std::unordered_map<const DimensionNode*, Range> state;

    for (const auto& dim: compute_op->loop_dimensions) {
      const auto& iv = compute_op->GetIterVarFromDim(0, dim);
      state[dim.operator->()] = dom_map.count(iv) ? dom_map.at(iv) : iv->dom;
    }

    // for (const auto& dim: compute_op->root_index_dimensions) {
    //   const auto& iv = compute_op->GetIterVarFromDim(0, dim);
    //   state[dim.operator->()] = dom_map.count(iv) ?
    // 	dom_map.at(compute_op->GetIterVarFromDim(0, dim)) :
    // 	iv->dom;
    // }

    for (const auto& it: GetIndexDimRangeFromLoopDimRange(compute_op, dom_map)) {
      state[it.first.operator->()] = it.second;
    }

    DimensionPassDownDomain(compute_op, &state, true);

    Array<Range> new_shape;
    for (auto dim: compute_op->dim_relation_graph->leaf_dimensions) {
      // std::cout << "[FTD]  After Dim: " << dim->name << std::endl;
      // std::cout << "[FTD]   After Range : " << state[dim.operator->()] << std::endl;
      // CHECK(state.count(dim.operator->())) << dim->name;
      // CHECK(is_zero(state[dim.operator->()]->min));
      new_shape.push_back(state[dim.operator->()]);
    }
    const_cast<ComputeOpNode*>(compute_op)->set_realize_bounds(new_shape);
    dim_doms[compute_op] = state;
  }

  for (auto stage: (*this)->stages) {
    auto compute_op = stage->op.as<ComputeOpNode>();
    if (!compute_op) continue;
    TensorLayoutFreezer(const_cast<ComputeOpNode*>(compute_op), dim_doms).replace();
  }
}

Tensor Schedule::split_tensor_dimension(const Tensor& tensor,
					const size_t dim_idx,
					const int factor) {
  auto compute_op = const_cast<ComputeOpNode*>(tensor->op.as<ComputeOpNode>());
  CHECK(compute_op) <<
    "Layout changes allowed only for ComputeOp";
  CHECK(dim_idx < compute_op->dim_relation_graph->leaf_dimensions.size());
  Dimension parent = compute_op->dim_relation_graph->leaf_dimensions[dim_idx];
  Dimension inner = DimensionNode::make(parent->name + ".inner", parent->type);
  Dimension outer = DimensionNode::make(parent->name + ".outer", parent->type);

  Array<DimensionRelation>& relations = compute_op->dim_relation_graph->relations;
  relations.push_back(DimensionSplitNode::make(parent, outer, inner, factor, PrimExpr()));

  auto leaf_dims = compute_op->dim_relation_graph->leaf_dimensions.CopyOnWrite();
  size_t pos = std::distance(leaf_dims->data.begin(), std::find(leaf_dims->data.begin(), leaf_dims->data.end(), parent));
  leaf_dims->data.erase(leaf_dims->data.begin() + pos);
  leaf_dims->data.insert(leaf_dims->data.begin() + pos, inner);
  leaf_dims->data.insert(leaf_dims->data.begin() + pos, outer);

  // std::cout << "[STD] After splitting leaf dimensions" << std::endl;
  // for (auto dim: compute_op->dim_relation_graph->leaf_dimensions) {
    // std::cout << "[STD]  " << dim->name << std::endl;
  // }

  return tensor;
}

Tensor Schedule::fuse_tensor_dimensions(const Tensor& tensor,
					const size_t dim_idx1,
					const size_t dim_idx2) {
  auto compute_op = const_cast<ComputeOpNode*>(tensor->op.as<ComputeOpNode>());
  CHECK(compute_op) <<
    "Layout changes allowed only for ComputeOp";
  CHECK(dim_idx1 < compute_op->dim_relation_graph->leaf_dimensions.size());
  CHECK(dim_idx2 < compute_op->dim_relation_graph->leaf_dimensions.size());
  CHECK(dim_idx1 == dim_idx2 - 1);

  Dimension inner = compute_op->dim_relation_graph->leaf_dimensions[dim_idx2];
  Dimension outer = compute_op->dim_relation_graph->leaf_dimensions[dim_idx1];
  auto fused_type = (inner->type == DimensionNode::kFunDim) || (outer->type == DimensionNode::kFunDim) ?
    DimensionNode::kFunDim : DimensionNode::kRangeDim;
  Dimension fused = DimensionNode::make(outer->name + "." + inner->name + ".fused", fused_type);

  Array<DimensionRelation>& relations = compute_op->dim_relation_graph->relations;
  relations.push_back(DimensionFuseNode::make(outer, inner, fused));

  auto leaf_dims = compute_op->dim_relation_graph->leaf_dimensions.CopyOnWrite();
  size_t pos1 = std::distance(leaf_dims->data.begin(), std::find(leaf_dims->data.begin(), leaf_dims->data.end(), inner));
  leaf_dims->data.erase(leaf_dims->data.begin() + pos1);
  size_t pos2 = std::distance(leaf_dims->data.begin(), std::find(leaf_dims->data.begin(), leaf_dims->data.end(), outer));
  leaf_dims->data.erase(leaf_dims->data.begin() + pos2);
  leaf_dims->data.insert(leaf_dims->data.begin() + pos2, fused);

  return tensor;
}

Tensor Schedule::reorder_tensor_dimensions(const Tensor& tensor,
					   const size_t dim_idx1,
					   const size_t dim_idx2) {
  auto compute_op = const_cast<ComputeOpNode*>(tensor->op.as<ComputeOpNode>());
  CHECK(compute_op) <<
    "Layout changes allowed only for ComputeOp";
  CHECK(dim_idx1 < compute_op->dim_relation_graph->leaf_dimensions.size());
  CHECK(dim_idx2 < compute_op->dim_relation_graph->leaf_dimensions.size());
  CHECK(dim_idx1 == dim_idx2 - 1);

  Dimension dim1 = compute_op->dim_relation_graph->leaf_dimensions[dim_idx1];
  Dimension dim2 = compute_op->dim_relation_graph->leaf_dimensions[dim_idx2];

  auto leaf_dims = compute_op->dim_relation_graph->leaf_dimensions.CopyOnWrite();
  // auto iter1 = std::find(leaf_dims->data.begin(), leaf_dims->data.end(), dim1);
  // auto iter2 = std::find(leaf_dims->data.begin(), leaf_dims->data.end(), dim2);
  // std::iter_swap(iter1, iter2);

  size_t pos1 = std::distance(leaf_dims->data.begin(), std::find(leaf_dims->data.begin(), leaf_dims->data.end(), dim1));
  size_t pos2 = std::distance(leaf_dims->data.begin(), std::find(leaf_dims->data.begin(), leaf_dims->data.end(), dim2));
  leaf_dims->data[pos2] = dim1;
  leaf_dims->data[pos1] = dim2;

  // std::cout << "[RTD] After reordering leaf dimensions" << std::endl;
  // for (auto dim: compute_op->dim_relation_graph->leaf_dimensions) {
    // std::cout << "[RTD]  " << dim->name << std::endl;
  // }

  return tensor;
}

Tensor Schedule::index_by_dense_dimensions(const Tensor& tensor) {
  auto compute_op = const_cast<ComputeOpNode*>(tensor->op.as<ComputeOpNode>());
  CHECK(compute_op) <<
    "Layout changes allowed only for ComputeOp";

  Array<DimensionRelation>& relations = compute_op->dim_relation_graph->relations;
  relations.push_back(DimensionChangeNode::make(Array<Dimension>(compute_op->dim_relation_graph->leaf_dimensions),
						Array<Dimension>(compute_op->loop_dimensions)));

  auto leaf_dims = compute_op->dim_relation_graph->leaf_dimensions.CopyOnWrite();
  leaf_dims->data.resize(0);

  for (auto dim: compute_op->axis) {
    leaf_dims->data.push_back(dim);
  }

  return tensor;
}
}
}
