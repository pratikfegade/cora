#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>

#include "graph.h"
#include "message_passing.h"
#include "ragged_utils.h"
#include "tensor_layout_utils.h"

#define COUT std::cout << "[CTL] "

namespace tvm {
namespace te {

Map<Dimension, Range> GetIndexDimRangeFromLoopDimRange(const ComputeOpNode* compute_op,
                                                       const Map<IterVar, Range>& dom_map) {
  Map<Dimension, Range> ret;
  for (const auto& root_dim : compute_op->root_index_dimensions) {
    CHECK(root_dim->isLoopDim());
    const auto& iv = compute_op->GetIterVarFromDim(0, root_dim);
    ret.Set(root_dim, dom_map.count(iv) ? dom_map.at(iv) : iv->dom);
  }

  return ret;
}

Array<Range> ComputeRealizeBounds(const Stage& stage, const ComputeOpNode* compute_op,
                                  const Map<IterVar, Range>& dom_map) {
  std::unordered_map<const DimensionNode*, Range> state;
  for (const auto& root_dim : compute_op->root_index_dimensions) {
    CHECK(root_dim->isLoopDim());
    const auto& iv = compute_op->GetIterVarFromDim(0, root_dim);
    state[root_dim.operator->()] = dom_map.count(iv) ? dom_map.at(iv) : iv->dom;
  }

  DimensionPassDownDomain(stage, compute_op, &state, true);

  Array<Range> new_shape;
  for (auto dim : stage->dim_relation_graph->leaf_dimensions) {
    new_shape.push_back(state[dim.operator->()]);
  }
  CHECK(new_shape.size() > 0) << stage;
  return new_shape;
}

void Schedule::freeze_tensor_dimensions(const Map<IterVar, Range>& dom_map) {
  Schedule& sch = *this;
  auto feed_graph = GetFeedGraph(sch, true);

  sch->InvalidateCache();
  sch->InitCache();

  for (auto s : sch->stages) {
    if (s->op.as<ComputeOpNode>()) {
      if (s->attach_type == kInlinedAlready) continue;
      auto compute_op = s->op.as<ComputeOpNode>();
      CHECK(compute_op);
      // std::cout << "[COMP_STAGE] " << s->op << " " << std::endl;

      Operation old_op = s->op;

      // std::cout << "[CTD] Op " << compute_op->name << std::endl;
      ComputeOpNode* mutable_compute_op = const_cast<ComputeOpNode*>(compute_op);
      mutable_compute_op->set_realize_bounds(ComputeRealizeBounds(s, compute_op, dom_map),
                                             "change_tensor_layout.cc:185");

      auto root_layouts = compute_op->storage_layouts;
      for (size_t i = 0; i < compute_op->num_outputs(); ++i) {
        if (root_layouts.size() > 0) {
          Modes leaf_layout = DimensionPassDownModes(s, compute_op, root_layouts[i]);
          if (leaf_layout.defined()) {
            // std::cout << "[CTL] Setting storage layout for " << old_op << " " << leaf_layout
            // << std::endl;
            mutable_compute_op->set_storage_layout(i, leaf_layout);
          }
        }
      }

      if (s->is_output) continue;
      feed_graph = GetFeedGraph(sch, true);
      // CheckSchedule(sch, "change_tensor_layout.cc:269", false);

      if (!feed_graph.count(old_op.output(0))) {
        for (auto it : feed_graph) {
          std::cout << "[FG] " << it.first->op << " " << old_op << std::endl;
        }
      }
      CHECK(feed_graph.count(old_op.output(0))) << old_op;
      auto readers = Array<Operation>(feed_graph.at(old_op.output(0)));

      std::unordered_map<Tensor, Tensor> vmap;
      std::unordered_map<Tensor, Tensor> rvmap;
      sch->InvalidateCache();
      sch->InitCache();
      auto& op2stage_ = sch->op2stage_cache_;
      for (Operation op : readers) {
        // std::cout << "[CTD]   Reader " << op << std::endl;
        Stage op_stage = op2stage_.at(op.get());
        Operation repl_op = ReplaceInputsGeneral(s, old_op, s->op, op, dom_map, root_layouts);
        // CHECK(!repl_op.same_as(op_stage->op))
        // << "Cannot find tensor " << s->op << " in the inputs to " << repl_op;
        if (!repl_op.same_as(op_stage->op)) {
          for (size_t i = 0; i < op_stage->op->num_outputs(); ++i) {
            vmap[op_stage->op.output(i)] = repl_op.output(i);
            rvmap[repl_op.output(i)] = op_stage->op.output(i);
          }
          op_stage->op = repl_op;
        }
      }
      ReplaceDataFlow(sch->stages, sch->cacheTensorInfos, &vmap, &rvmap);

      // Refresh the feed graph
      feed_graph = GetFeedGraph(sch, true);
      continue;
    }
  }
}

Tensor Schedule::split_tensor_dimension(const Tensor& tensor, const size_t dim_idx,
                                        const int factor) {
  auto compute_op = const_cast<ComputeOpNode*>(tensor->op.as<ComputeOpNode>());
  Stage s = this->operator[](tensor->op);
  CHECK(compute_op) << "Layout changes allowed only for ComputeOp";
  CHECK(dim_idx < s->dim_relation_graph->leaf_dimensions.size());
  Dimension parent = s->dim_relation_graph->leaf_dimensions[dim_idx];
  Dimension inner = DimensionNode::make(parent->name + ".inner", parent->type);
  Dimension outer = DimensionNode::make(parent->name + ".outer", parent->type);

  Array<DimensionRelation>& relations = s->dim_relation_graph->relations;
  relations.push_back(DimensionSplitNode::make(parent, outer, inner, factor, PrimExpr()));

  auto leaf_dims = s->dim_relation_graph->leaf_dimensions.CopyOnWrite();
  size_t pos = std::distance(leaf_dims->data.begin(),
                             std::find(leaf_dims->data.begin(), leaf_dims->data.end(), parent));
  leaf_dims->data.erase(leaf_dims->data.begin() + pos);
  leaf_dims->data.insert(leaf_dims->data.begin() + pos, inner);
  leaf_dims->data.insert(leaf_dims->data.begin() + pos, outer);

  // std::cout << "[STD] Splitting " << tensor->op << " "
  // << s->dim_relation_graph->leaf_dimensions.size() << std::endl;
  return tensor;
}

Tensor Schedule::fuse_tensor_dimensions(const Tensor& tensor, const size_t dim_idx1,
                                        const size_t dim_idx2, const int factor) {
  auto compute_op = const_cast<ComputeOpNode*>(tensor->op.as<ComputeOpNode>());
  Stage s = this->operator[](tensor->op);
  CHECK(compute_op) << "Layout changes allowed only for ComputeOp";
  CHECK(dim_idx1 < s->dim_relation_graph->leaf_dimensions.size());
  CHECK(dim_idx2 < s->dim_relation_graph->leaf_dimensions.size());
  CHECK(dim_idx1 == dim_idx2 - 1);

  Dimension inner = s->dim_relation_graph->leaf_dimensions[dim_idx2];
  Dimension outer = s->dim_relation_graph->leaf_dimensions[dim_idx1];

  bool dependent_ragged_dims = verify_dimension_order(s, {inner, outer});
  CHECK(outer->type != DimensionNode::kFunDim && inner->type != DimensionNode::kFunDim);
  Dimension fused =
      DimensionNode::make(outer->name + "." + inner->name + ".fused", DimensionNode::kRangeDim);

  Array<DimensionRelation>& relations = s->dim_relation_graph->relations;
  relations.push_back(DimensionFuseNode::make(outer, inner, fused, dependent_ragged_dims, factor));

  auto leaf_dims = s->dim_relation_graph->leaf_dimensions.CopyOnWrite();
  size_t pos1 = std::distance(leaf_dims->data.begin(),
                              std::find(leaf_dims->data.begin(), leaf_dims->data.end(), inner));
  leaf_dims->data.erase(leaf_dims->data.begin() + pos1);
  size_t pos2 = std::distance(leaf_dims->data.begin(),
                              std::find(leaf_dims->data.begin(), leaf_dims->data.end(), outer));
  leaf_dims->data.erase(leaf_dims->data.begin() + pos2);
  leaf_dims->data.insert(leaf_dims->data.begin() + pos2, fused);

  return tensor;
}

Tensor Schedule::reorder_tensor_dimensions(const Tensor& tensor, const size_t dim_idx1,
                                           const size_t dim_idx2) {
  auto compute_op = const_cast<ComputeOpNode*>(tensor->op.as<ComputeOpNode>());
  Stage s = this->operator[](tensor->op);
  CHECK(compute_op) << "Layout changes allowed only for ComputeOp";
  CHECK(dim_idx1 < s->dim_relation_graph->leaf_dimensions.size());
  CHECK(dim_idx2 < s->dim_relation_graph->leaf_dimensions.size());
  CHECK(dim_idx1 == dim_idx2 - 1);

  Dimension dim1 = s->dim_relation_graph->leaf_dimensions[dim_idx1];
  Dimension dim2 = s->dim_relation_graph->leaf_dimensions[dim_idx2];

  CHECK(verify_dimension_order(s, {dim2, dim1}));

  auto leaf_dims = s->dim_relation_graph->leaf_dimensions.CopyOnWrite();
  size_t pos1 = std::distance(leaf_dims->data.begin(),
                              std::find(leaf_dims->data.begin(), leaf_dims->data.end(), dim1));
  size_t pos2 = std::distance(leaf_dims->data.begin(),
                              std::find(leaf_dims->data.begin(), leaf_dims->data.end(), dim2));
  leaf_dims->data[pos2] = dim1;
  leaf_dims->data[pos1] = dim2;

  return tensor;
}

}  // namespace te
}  // namespace tvm

#undef COUT
