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
    auto r = state[dim.operator->()];
    new_shape.push_back(Range::make_by_min_extent(Simplify(r->min), Simplify(r->extent)));
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
    if (auto compute_op = s->op.as<ComputeOpNode>()) {
      if (s->attach_type == kInlinedAlready) continue;
      // std::cout << "[COMP_STAGE] " << s->op << " " << std::endl;

      Operation old_op = s->op;

      // std::cout << "[CTD] Op " << compute_op->name << std::endl;
      ComputeOpNode* mutable_compute_op = const_cast<ComputeOpNode*>(compute_op);
      // std::cout << "[CTL] Setting realize bounds " << old_op << std::endl;
      mutable_compute_op->set_realize_bounds(ComputeRealizeBounds(s, compute_op, dom_map),
                                             "change_tensor_layout.cc:185");

      auto root_layouts = compute_op->storage_layouts;
      for (size_t i = 0; i < static_cast<size_t>(compute_op->num_outputs()); ++i) {
        if (root_layouts.size() > 0) {
          Modes leaf_layout = DimensionPassDownModes(s, compute_op, root_layouts[i]);
          if (leaf_layout.defined()) {
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
          for (size_t i = 0; i < static_cast<size_t>(op_stage->op->num_outputs()); ++i) {
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
    } else if (auto placeholder_op = s->op.as<PlaceholderOpNode>()) {
      Operation old_op = s->op;

      if (placeholder_op->self_index_dimensions.size() == 0) {
        continue;
      }

      std::cout << "[CTD] Op " << placeholder_op->name << std::endl;
      PlaceholderOpNode* mutable_placeholder_op = const_cast<PlaceholderOpNode*>(placeholder_op);

      auto root_layout = placeholder_op->layout;
      if (root_layout.defined()) {
        Modes leaf_layout = DimensionPassDownModes(s, placeholder_op, root_layout);
        if (leaf_layout.defined()) {
          mutable_placeholder_op->set_storage_layout(leaf_layout);
        }
      }

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
        std::cout << "[CTD]   Reader " << op << std::endl;
        Stage op_stage = op2stage_.at(op.get());
        Operation repl_op = ReplaceInputsGeneral(s, old_op, s->op, op, dom_map, {root_layout});
        // CHECK(!repl_op.same_as(op_stage->op))
        // << "Cannot find tensor " << s->op << " in the inputs to " << repl_op;
        if (!repl_op.same_as(op_stage->op)) {
          for (size_t i = 0; i < static_cast<size_t>(op_stage->op->num_outputs()); ++i) {
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
  auto bvd_op = tensor->op.as<BaseVarDimOpNode>();
  Stage s = this->operator[](tensor->op);
  CHECK(bvd_op) << "Layout changes allowed only for ComputeOp";
  CHECK(dim_idx < s->dim_relation_graph->leaf_dimensions.size());
  Dimension parent = s->dim_relation_graph->leaf_dimensions[dim_idx];
  Dimension inner =
      Dimension::get_or_create_dimension({DimKey::kSplitInner, parent.operator->(), nullptr});
  Dimension outer =
      Dimension::get_or_create_dimension({DimKey::kSplitOuter, parent.operator->(), nullptr});

  Array<DimensionRelation>& relations = s->dim_relation_graph->relations;
  relations.push_back(DimensionSplitNode::make(parent, outer, inner, factor, PrimExpr()));

  auto leaf_dims = s->dim_relation_graph->leaf_dimensions.CopyOnWrite();
  size_t pos = std::distance(leaf_dims->data.begin(),
                             std::find(leaf_dims->data.begin(), leaf_dims->data.end(), parent));
  leaf_dims->data.erase(leaf_dims->data.begin() + pos);
  leaf_dims->data.insert(leaf_dims->data.begin() + pos, inner);
  leaf_dims->data.insert(leaf_dims->data.begin() + pos, outer);

  return tensor;
}

Tensor Schedule::fuse_tensor_dimensions(const Tensor& tensor, const size_t dim_idx1,
                                        const size_t dim_idx2, const int factor) {
  std::cout << "[FTD] Fusing dimensions " << tensor << " " << dim_idx1 << " " << dim_idx2
            << std::endl;
  auto bvd_op = tensor->op.as<BaseVarDimOpNode>();
  Stage s = this->operator[](tensor->op);
  CHECK(bvd_op) << "Layout changes allowed only for ComputeOp or PlaceholderOp";
  CHECK(dim_idx1 < s->dim_relation_graph->leaf_dimensions.size());
  CHECK(dim_idx2 < s->dim_relation_graph->leaf_dimensions.size());
  CHECK(dim_idx1 == dim_idx2 - 1);

  Dimension inner = s->dim_relation_graph->leaf_dimensions[dim_idx2];
  Dimension outer = s->dim_relation_graph->leaf_dimensions[dim_idx1];
  Dimension fused =
      Dimension::get_or_create_dimension({DimKey::kFuse, outer.operator->(), inner.operator->()});

  bool dependent_ragged_dims = !verify_dimension_order(s, {inner, outer});

  DimensionRelation fuse_relation;
  if (dependent_ragged_dims) {
    std::unordered_map<const DimensionNode*, Range> state;

    auto shape = tensor->op->output_shape(tensor->value_index);
    auto root_dims = bvd_op->GetRootIndexDimensions(tensor->value_index);
    for (size_t i = 0; i < root_dims.size(); ++i) {
      state[root_dims[i].operator->()] = Range::make_by_min_max_exclusive(0, shape[i]);
    }

    DimensionPassDownDomain(s, bvd_op, &state, true);
    PrimExpr outer_max = state.count(outer.operator->())
                             ? state.at(outer.operator->())->max_inclusive()
                             : NullValue<PrimExpr>();
    PrimExpr inner_max = state.count(inner.operator->())
                             ? state.at(inner.operator->())->max_inclusive()
                             : NullValue<PrimExpr>();

    UninterpFun fused_to_outer_uf;
    UninterpFun fused_to_inner_uf;
    UninterpFun outer_inner_to_fused_uf;
    auto prefix = inner->name + "_" + outer->name;
    auto var1 = Var("arg0", DataType::Int(32));
    auto var2 = Var("arg1", DataType::Int(32));
    fused_to_outer_uf = UninterpFunNode::make(
        prefix + "_dfo",
        outer_max.defined() ? Range::make_by_min_max_inclusive(0, outer_max) : NullValue<Range>(),
        {fused}, {var1}, NullValue<PrimExpr>(), UninterpFunNode::kFOFun);
    fused_to_inner_uf = UninterpFunNode::make(
        prefix + "_dfi",
        inner_max.defined() ? Range::make_by_min_max_inclusive(0, inner_max) : NullValue<Range>(),
        {fused}, {var1}, NullValue<PrimExpr>(), UninterpFunNode::kFIFun);
    Range fused_range = NullValue<Range>();
    if (outer_max.defined() && inner_max.defined()) {
      PrimExpr max = outer_max * inner_max + outer_max + inner_max;
      fused_range = Range::make_by_min_max_inclusive(0, max);
    }
    outer_inner_to_fused_uf =
        UninterpFunNode::make(prefix + "_doif", fused_range, {outer, inner}, {var1, var2},
                              NullValue<PrimExpr>(), UninterpFunNode::kOIFFun);

    auto fusion_info = RaggedFusionInfoNode::make({}, {}, {}, fused_to_outer_uf, fused_to_inner_uf,
                                                  outer_inner_to_fused_uf);
    const_cast<UninterpFunNode*>(fused_to_inner_uf.operator->())->fusion_info = fusion_info;
    const_cast<UninterpFunNode*>(fused_to_outer_uf.operator->())->fusion_info = fusion_info;
    const_cast<UninterpFunNode*>(outer_inner_to_fused_uf.operator->())->fusion_info = fusion_info;

    fuse_relation = RaggedDimensionFuseNode::make(outer, inner, fused, fused_to_outer_uf,
                                                  fused_to_inner_uf, outer_inner_to_fused_uf);
  } else {
    CHECK(outer->type != DimensionNode::kFunDim && inner->type != DimensionNode::kFunDim);
    fuse_relation = DimensionFuseNode::make(outer, inner, fused, factor);
  }

  Array<DimensionRelation>& relations = s->dim_relation_graph->relations;
  relations.push_back(fuse_relation);

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
  auto bvd_op = tensor->op.as<BaseVarDimOpNode>();
  Stage s = this->operator[](tensor->op);
  CHECK(bvd_op) << "Layout changes allowed only for ComputeOp";
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
