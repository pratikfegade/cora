#include <tvm/te/schedule.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/te/operation.h>
#include "graph.h"
#include "message_passing.h"

namespace tvm {
namespace te {
class SplitTensorReplacer: public StmtExprMutator {
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

      DimensionPassDownValues(relations_graph, &root_values, true);

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

public:
  SplitTensorReplacer(ComputeOpNode* reader_) :
    reader(reader_) {}

  void replace() {
    // TODO(ppf): Handle reductions here
    Array<PrimExpr> body;
    for (auto e: reader->body) {
      PrimExpr replaced = this->VisitExpr(e);
      std::cout << "[FTD] Body: " << e << std::endl;
      std::cout << "[FTD] Replaced: " << replaced << std::endl;
      body.push_back(replaced);
    }

    reader->body = body;
  }
};

void Schedule::freeze_tensor_dimensions(Map<IterVar, Range> dom_map) {
  for (auto stage: (*this)->stages) {
    auto compute_op = stage->op.as<ComputeOpNode>();
    if (!compute_op) continue;

    std::cout << "[FTD] Freezing shape for : " << compute_op->name << std::endl;
    std::unordered_map<const DimensionNode*, Range> state;
    for (auto dim: compute_op->root_index_dimensions) {
      auto iv = compute_op->GetIterVarFromDim(dim);
      if (dom_map.count(iv)) {
	state[dim.operator->()] = dom_map.at(compute_op->GetIterVarFromDim(dim));
      }
      else {
	state[dim.operator->()] = iv->dom;
      }
    }

    DimensionPassDownDomain(compute_op->dim_relation_graph, &state, true);


    Array<PrimExpr> new_shape;
    for (auto dim: compute_op->dim_relation_graph->leaf_dimensions) {
      CHECK(state.count(dim.operator->())) << dim->name;
      std::cout << "[FTD]   DIM: " << dim->name << " " << state[dim.operator->()] << std::endl;
      CHECK(is_zero(state[dim.operator->()]->min));
      new_shape.push_back(state[dim.operator->()]->extent);
    }
    const_cast<ComputeOpNode*>(compute_op)->update_shape(new_shape);
  }

  for (auto stage: (*this)->stages) {
    auto compute_op = stage->op.as<ComputeOpNode>();
    if (!compute_op) continue;
    SplitTensorReplacer(const_cast<ComputeOpNode*>(compute_op)).replace();
  }
}

Tensor Schedule::split_tensor_dimension(const Tensor& tensor,
					const size_t dim_idx,
					const int factor) {
  auto compute_op = const_cast<ComputeOpNode*>(tensor->op.as<ComputeOpNode>());
  CHECK(compute_op) <<
    "Layout changes allowed only for ComputeOp";
  CHECK(dim_idx < compute_op->output_shape_storage.size());
  Dimension parent = compute_op->root_index_dimensions[dim_idx];
  Dimension inner = DimensionNode::make(parent->name + ".inner", parent->type);
  Dimension outer = DimensionNode::make(parent->name + ".outer", parent->type);

  Array<DimensionRelation>& relations = compute_op->dim_relation_graph->relations;
  relations.push_back(DimensionSplitNode::make(parent, outer, inner, factor, PrimExpr()));

  auto leaf_dims = compute_op->dim_relation_graph->leaf_dimensions.CopyOnWrite();
  size_t pos = std::distance(leaf_dims->data.begin(), std::find(leaf_dims->data.begin(), leaf_dims->data.end(), parent));
  leaf_dims->data.erase(leaf_dims->data.begin() + pos);
  leaf_dims->data.insert(leaf_dims->data.begin() + pos, inner);
  leaf_dims->data.insert(leaf_dims->data.begin() + pos, outer);

  return tensor;
}
}
}
