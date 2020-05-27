#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>

#include "graph.h"
#include "message_passing.h"
#include "tensor_layout_utils.h"

#define COUT std::cout << "[CTL] "

namespace tvm {
namespace te {

FeedGraph GetFeedGraph(Schedule& sch) {
  static Array<Operation> roots;
  roots.resize(0);
  for (Operation op : sch->outputs) {
    roots.push_back(sch->stage_map[op]->op);
  }
  return CreateFeedGraph(CreateReadGraph(roots));
}

Map<Dimension, Range> GetIndexDimRangeFromLoopDimRange(const ComputeOpNode* compute_op,
                                                       const Map<IterVar, Range>& dom_map) {
  Map<Dimension, Range> ret;
  for (const auto& root_dim : compute_op->root_index_dimensions) {
    if (root_dim->type <= DimensionNode::kRangeDim) {
      const auto& iv = compute_op->GetIterVarFromDim(0, root_dim);
      ret.Set(root_dim, dom_map.count(iv) ? dom_map.at(iv) : iv->dom);
    } else {
      UninterpFun ufun = compute_op->GetDimVarEntry(0, root_dim).value_expr;
      bool non_constant = false;
      CHECK(ufun->dimensions.defined());
      for (auto arg_dim : ufun->dimensions) {
        Range r = dom_map.at(compute_op->GetIterVarFromDim(0, arg_dim));
        if (!tir::is_one(r->extent)) {
          non_constant = true;
        }
      }

      if (non_constant) {
        ret.Set(root_dim, ufun->range);
      } else {
        Array<PrimExpr> args;
        for (auto arg_dim : ufun->dimensions) {
          Range r = dom_map.at(compute_op->GetIterVarFromDim(0, arg_dim));
          args.push_back(r->min);
        }
        ret.Set(root_dim, Range::make_by_min_extent(
                              UninterpFun::MakeCallTo(ufun, args, ufun->dimensions), 1));
      }
    }
  }

  return ret;
}

Array<Range> ComputeRealizeBounds(const Stage& stage, const ComputeOpNode* compute_op,
                                  const Map<IterVar, Range>& dom_map) {
  // std::cout << "[FTD] Op " << compute_op->name << " " << compute_op << std::endl;

  std::unordered_map<const DimensionNode*, Range> state;

  for (const auto& dim : compute_op->loop_dimensions) {
    const auto& iv = compute_op->GetIterVarFromDim(0, dim);
    state[dim.operator->()] = dom_map.count(iv) ? dom_map.at(iv) : iv->dom;
    // std::cout << "[FTD]  Before Dim: " << dim->name << " " << state[dim.operator->()] << " "
    // << dom_map.count(iv) << std::endl;
  }

  for (const auto& it : GetIndexDimRangeFromLoopDimRange(compute_op, dom_map)) {
    state[it.first.operator->()] = it.second;
  }

  DimensionPassDownDomain(stage, compute_op, &state, true);

  Array<Range> new_shape;
  for (auto dim : stage->dim_relation_graph->leaf_dimensions) {
    // std::cout << "[FTD]  After Dim: " << dim->name << " " << state[dim.operator->()] <<
    // std::endl;
    new_shape.push_back(state[dim.operator->()]);
  }
  CHECK(new_shape.size() > 0) << stage;
  return new_shape;
}

void ReplaceIndexTensorByDenseTensor(Schedule& sch, Stage s, Tensor old_tensor, Tensor new_tensor,
                                     Array<Dimension> old_dims, Array<Dimension> new_dims) {
  s->op = new_tensor->op;
  // Refresh the feed graph
  auto feed_graph = GetFeedGraph(sch);

  auto readers = Array<Operation>(feed_graph.at(old_tensor));
  AccessPatternCollector collector(old_tensor, old_dims, readers);
  collector.collect();
  PatternsSet patterns = collector.access_patterns;
  AccessToPatternMap access_to_pattern_map = collector.access_to_pattern_map;

  CHECK_EQ(patterns.size(), 1)
      << "Tensor dense indexing suported for single access pattern tensors";

  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  sch->InvalidateCache();
  sch->InitCache();
  auto& op2stage_ = sch->op2stage_cache_;
  for (Operation op : readers) {
    Stage op_stage = op2stage_.at(op.get());
    Operation repl_op =
        ReplaceInputs(op, &access_to_pattern_map, new_tensor, new_dims, old_dims, false);
    CHECK(!repl_op.same_as(op_stage->op))
        << "Cannot find tensor " << old_tensor << " in the inputs to " << repl_op;
    vmap[op_stage->op.output(0)] = repl_op.output(0);
    rvmap[repl_op.output(0)] = op_stage->op.output(0);
    op_stage->op = repl_op;
  }
  ReplaceDataFlow(sch->stages, &vmap, &rvmap);
}

Operation CreateDenselyIndexedComputeOpCopy(Stage s, const ComputeOpNode* old_op,
                                            const Map<IterVar, Range>& dom_map) {
  auto n = make_object<ComputeOpNode>();

  n->realize_bounds = ComputeRealizeBounds(s, old_op, dom_map);
  n->who_set_realize_bounds = "change_tensor_layout.cc:127";

  // OperationNode fields
  n->name = std::move(old_op->name);
  n->tag = std::move(old_op->tag);
  n->attrs = std::move(old_op->attrs);

  // BaseVarDimOpNode fields
  n->dim2var_maps = std::move(old_op->dim2var_maps);
  n->var2dim_map = std::move(old_op->var2dim_map);

  // BaseComputeOpNode fields
  n->axis = std::move(old_op->axis);
  n->reduce_axis = std::move(old_op)->reduce_axis;
  for (const auto& r : n->realize_bounds) {
    n->output_shape_storage.push_back(r->extent);
  }
  n->index_variables = std::move(old_op->index_variables);
  n->index_expressions = std::move(old_op->index_expressions);
  n->loop_dimensions = std::move(old_op->loop_dimensions);
  n->index_dimensions = std::move(old_op->index_dimensions);
  n->root_index_dimensions = s->dim_relation_graph->leaf_dimensions;

  // ComputeOpNode fields
  n->body = std::move(old_op->body);

  Operation new_op = std::move(Operation(n));
  return new_op;
}

const DimensionChangeNode* GetChangeRel(Stage s) {
  const DimensionChangeNode* change_rel = nullptr;
  CHECK(s->dim_relation_graph->relations.defined());
  for (const auto& rel : s->dim_relation_graph->relations) {
    if ((change_rel = rel.as<DimensionChangeNode>())) {
      break;
    }
  }
  return change_rel;
}

void IndexByDenseLayoutChange(Schedule& sch, const Map<IterVar, Range>& dom_map) {
  auto feed_graph = GetFeedGraph(sch);

  std::unordered_set<const Object*> scan_updates;
  Array<Stage> scan_stages;
  Array<Stage> compute_op_stages;
  for (auto s : sch->stages) {
    if (const ScanOpNode* scan = s->op.as<ScanOpNode>()) {
      auto change_rel = GetChangeRel(s);
      if (change_rel) {
        for (Tensor t : scan->update) {
          scan_updates.insert(t->op.get());
        }
        scan_stages.push_back(s);
      }
    } else if (s->op.as<ComputeOpNode>()) {
      compute_op_stages.push_back(s);
    }
  }

  for (auto s : compute_op_stages) {
    if (s->attach_type == kInlinedAlready) continue;
    auto compute_op = s->op.as<ComputeOpNode>();
    CHECK(compute_op);
    if (scan_updates.count(s->op.get())) {
      continue;
    }

    const DimensionChangeNode* change_rel = GetChangeRel(s);
    if (change_rel) {
      CHECK(compute_op) << "Only compute ops supported for dense tensor indexing";
      CHECK_EQ(compute_op->num_outputs(), 1)
          << "Only single output ops supported for dense indexing";
      Tensor tensor = s->op.output(0);
      CHECK(feed_graph.count(tensor)) << "Tensor cannot be found in feed graph";

      std::cout << "[IBD] ComputeOp " << GetRef<Operation>(compute_op) << std::endl;
      Operation new_op = CreateDenselyIndexedComputeOpCopy(s, compute_op, dom_map);
      ReplaceIndexTensorByDenseTensor(sch, s, tensor, new_op.output(0),
                                      compute_op->root_index_dimensions,
                                      s->dim_relation_graph->leaf_dimensions);

      // Refresh the feed graph
      feed_graph = GetFeedGraph(sch);
    } else {
      const_cast<ComputeOpNode*>(compute_op)
          ->set_realize_bounds(ComputeRealizeBounds(s, compute_op, dom_map),
                               "change_tensor_layout.cc:185");
      continue;
    }
  }

  // Now process the scans
  for (auto stage : scan_stages) {
    sch->InvalidateCache();
    sch->InitCache();
    auto scan_op = stage->op.as<ScanOpNode>();

    // Replace state placeholders inside the scan
    Array<Array<Dimension>> all_old_dims;
    Array<Array<Dimension>> all_new_dims;
    int num_outputs = scan_op->num_outputs();
    Array<Tensor> new_states;
    for (Tensor old_state : scan_op->state_placeholder) {
      auto old_state_op = old_state->op.as<PlaceholderOpNode>();
      auto state_stage = sch->op2stage_cache_[old_state_op];

      // Create new state placeholder
      Array<PrimExpr> new_shape;
      for (auto iv : old_state_op->axis) {
        new_shape.push_back(iv->dom->extent);
      }

      Array<Dimension> old_dims = old_state_op->self_index_dimensions;
      Array<Dimension> new_dims = old_state_op->loop_dimensions;
      all_old_dims.push_back(old_dims);
      all_new_dims.push_back(new_dims);

      Operation new_state_op = PlaceholderOpNode::make(
          old_state_op->name + ".d", new_shape, old_state_op->dtype, old_state_op->axis, new_dims,
          {}, old_state_op->loop_dimensions, {});

      Tensor new_state = new_state_op.output(old_state->value_index);
      new_states.push_back(new_state);

      // Replace the state placeholder
      ReplaceIndexTensorByDenseTensor(sch, state_stage, old_state, new_state, old_dims, new_dims);
    }

    scan_op = stage->op.as<ScanOpNode>();

    // New updates
    sch->InvalidateCache();
    sch->InitCache();
    Array<Tensor> new_updates;
    for (int i = 0; i < num_outputs; ++i) {
      Tensor old_update = scan_op->update[i];
      auto update_op = old_update->op.as<ComputeOpNode>();
      Stage update_stage = sch->op2stage_cache_[update_op];
      Operation new_update_op = CreateDenselyIndexedComputeOpCopy(update_stage, update_op, dom_map);
      new_updates.push_back(new_update_op.output(0));
      update_stage->op = new_update_op;
    }

    // Create a new scan op:
    Operation new_scan_op;
    {
      auto n = make_object<ScanOpNode>();
      n->name = scan_op->name + ".d";
      n->tag = scan_op->tag;
      n->attrs = scan_op->attrs;

      n->dim2var_maps = scan_op->dim2var_maps;
      n->var2dim_map = scan_op->var2dim_map;

      n->scan_axis = scan_op->scan_axis;
      n->init = scan_op->init;
      n->update = new_updates;
      n->state_placeholder = new_states;
      n->inputs = scan_op->inputs;
      n->scan_dim = scan_op->scan_dim;

      for (int i = 0; i < num_outputs; ++i) {
        auto new_update_op = new_updates[i]->op.as<ComputeOpNode>();
        for (size_t k = 0; k < new_update_op->root_index_dimensions.size(); ++k) {
          auto dim = new_update_op->root_index_dimensions[k];
          n->spatial_dimensions_.push_back(dim);
          n->spatial_axis_.push_back(n->dim2var_maps[i].at(dim.as<DimensionNode>()).iv);
        }
      }

      new_scan_op = Operation(n);
    }

    // Replace the scan
    for (int i = 0; i < num_outputs; ++i) {
      ReplaceIndexTensorByDenseTensor(sch, stage, GetRef<Operation>(scan_op).output(i),
                                      new_scan_op.output(i), all_old_dims[i], all_new_dims[i]);
    }
  }
}

void Schedule::freeze_tensor_dimensions(const Map<IterVar, Range>& dom_map) {
  IndexByDenseLayoutChange(*this, dom_map);
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

  return tensor;
}

Tensor Schedule::fuse_tensor_dimensions(const Tensor& tensor, const size_t dim_idx1,
                                        const size_t dim_idx2) {
  auto compute_op = const_cast<ComputeOpNode*>(tensor->op.as<ComputeOpNode>());
  Stage s = this->operator[](tensor->op);
  CHECK(compute_op) << "Layout changes allowed only for ComputeOp";
  CHECK(dim_idx1 < s->dim_relation_graph->leaf_dimensions.size());
  CHECK(dim_idx2 < s->dim_relation_graph->leaf_dimensions.size());
  CHECK(dim_idx1 == dim_idx2 - 1);

  Dimension inner = s->dim_relation_graph->leaf_dimensions[dim_idx2];
  Dimension outer = s->dim_relation_graph->leaf_dimensions[dim_idx1];
  auto fused_type =
      (inner->type == DimensionNode::kFunDim) || (outer->type == DimensionNode::kFunDim)
          ? DimensionNode::kFunDim
          : DimensionNode::kRangeDim;
  Dimension fused = DimensionNode::make(outer->name + "." + inner->name + ".fused", fused_type);

  Array<DimensionRelation>& relations = s->dim_relation_graph->relations;
  relations.push_back(DimensionFuseNode::make(outer, inner, fused));

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

  auto leaf_dims = s->dim_relation_graph->leaf_dimensions.CopyOnWrite();
  size_t pos1 = std::distance(leaf_dims->data.begin(),
                              std::find(leaf_dims->data.begin(), leaf_dims->data.end(), dim1));
  size_t pos2 = std::distance(leaf_dims->data.begin(),
                              std::find(leaf_dims->data.begin(), leaf_dims->data.end(), dim2));
  leaf_dims->data[pos2] = dim1;
  leaf_dims->data[pos1] = dim2;

  return tensor;
}

Tensor Schedule::index_by_dense_dimensions(const Tensor& tensor) {
  Stage s = this->operator[](tensor->op);
  // std::cout << "[IDD] Op " << tensor->op << " " << s << std::endl;
  Array<Dimension> dense_dims;
  if (auto compute_op = const_cast<ComputeOpNode*>(tensor->op.as<ComputeOpNode>())) {
    for (const auto& dim : compute_op->loop_dimensions) {
      dense_dims.push_back(dim);
    }
  } else if (auto scan_op = const_cast<ScanOpNode*>(tensor->op.as<ScanOpNode>())) {
    for (int i = 0; i < scan_op->num_outputs(); ++i) {
      const ComputeOpNode* update_op = scan_op->update[i]->op.as<ComputeOpNode>();
      CHECK(update_op) << "Don't support update ops that aren't computeops yet.";
      for (const auto& dim : update_op->loop_dimensions) {
        dense_dims.push_back(dim);
      }
    }

    // Also change dimensions for update ops
    for (const auto& update : scan_op->update) {
      this->index_by_dense_dimensions(update);
    }
  } else {
    CHECK(false) << "Layout changes allowed only for ComputeOp and ScanOp";
  }

  s->dim_relation_graph->relations.push_back(DimensionChangeNode::make(
      Array<Dimension>(s->dim_relation_graph->leaf_dimensions), dense_dims));

  auto leaf_dims = s->dim_relation_graph->leaf_dimensions.CopyOnWrite();
  leaf_dims->data.resize(0);

  for (auto dim : dense_dims) {
    leaf_dims->data.push_back(dim);
  }

  return tensor;
}
}  // namespace te
}  // namespace tvm

#undef COUT
