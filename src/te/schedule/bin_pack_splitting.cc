#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>

#include "../operation/op_util.h"
#include "graph.h"
#include "message_passing.h"
#include "ragged_utils.h"
#include "tensor_layout_utils.h"

namespace tvm {
namespace te {

Array<Operation> SplitALoop(Schedule& sch, Operation op, size_t to_split_index,
                            UninterpFun split_point, std::unordered_map<Tensor, Tensor> graph1_vsub,
                            std::unordered_map<Tensor, Tensor> graph2_vsub) {
  sch->InvalidateCache();
  Stage s = sch.operator[](op);
  // CHECK(s->is_output)
  // << "Loop splitting for bin packing is currently allowed only for output tensors";

  auto compute_op = op.as<ComputeOpNode>();
  CHECK(compute_op) << "Loop splitting for bin packing is currently allowed only for ComputeOp";

  std::unordered_map<const VarNode*, PrimExpr> op1_vsub;
  std::unordered_map<const VarNode*, PrimExpr> op2_vsub;

  Modes loop_layout = compute_op->loop_layout();
  Array<IterVar> op1_axis;
  Array<IterVar> op2_axis;
  Array<Var> op1_vars;
  Array<Var> op2_vars;
  Array<Dimension> op1_dims;
  Array<Dimension> op2_dims;
  Array<UninterpFun> op1_min_ufs;
  Array<UninterpFun> op2_min_ufs;
  Array<UninterpFun> op1_ext_ufs;
  Array<UninterpFun> op2_ext_ufs;
  Array<PrimExpr> op1_maxes;
  Array<PrimExpr> op2_maxes;
  for (size_t i = 0; i < compute_op->axis.size(); ++i) {
    auto orig_iv = compute_op->axis[i];
    auto orig_dim = compute_op->root_index_dimensions[i];
    IterVar iv1;
    IterVar iv2;
    auto var1 = Var("iv1" + std::to_string(i), orig_iv->var.dtype());
    auto var2 = Var("iv2" + std::to_string(i), orig_iv->var.dtype());
    VarReplacer op1_replacer(op1_vsub);
    VarReplacer op2_replacer(op2_vsub);
    if (i == to_split_index) {
      iv1 = IterVarNode::make(
          Range::make_by_min_max_exclusive(op1_replacer(orig_iv->dom->min),
                                           split_point.MakeCallTo(op1_vars, op1_dims)),
          var1, orig_iv->iter_type);
      iv2 = IterVarNode::make(
          Range::make_by_min_max_exclusive(split_point.MakeCallTo(op2_vars, op2_dims),
                                           op2_replacer(orig_iv->dom->max_exclusive())),
          var2, orig_iv->iter_type);
      op1_min_ufs.push_back(loop_layout->l_fun_mins[i]);
      op1_ext_ufs.push_back(split_point);
      op2_min_ufs.push_back(split_point);
      op2_ext_ufs.push_back(loop_layout->l_funs[i]);
    } else {
      iv1 = IterVarNode::make(op1_replacer.replace(orig_iv->dom), var1, orig_iv->iter_type);
      iv2 = IterVarNode::make(op2_replacer.replace(orig_iv->dom), var2, orig_iv->iter_type);
      op1_min_ufs.push_back(loop_layout->l_fun_mins[i]);
      op1_ext_ufs.push_back(loop_layout->l_funs[i]);
      op2_min_ufs.push_back(loop_layout->l_fun_mins[i]);
      op2_ext_ufs.push_back(loop_layout->l_funs[i]);
    }
    op1_axis.push_back(iv1);
    op2_axis.push_back(iv2);
    op1_vars.push_back(iv1->var);
    op2_vars.push_back(iv2->var);
    op1_dims.push_back(orig_dim);
    op2_dims.push_back(orig_dim);
    op1_vsub[orig_iv->var.get()] = var1;
    op2_vsub[orig_iv->var.get()] = var2;
    op1_maxes.push_back(loop_layout->l_maxes[i]);
    op2_maxes.push_back(loop_layout->l_maxes[i]);
  }

  Array<IterVar> op1_red_axis;
  Array<IterVar> op2_red_axis;
  for (size_t i = 0; i < compute_op->reduce_axis.size(); ++i) {
    auto orig_iv = compute_op->reduce_axis[i];
    auto orig_dim = compute_op->reduction_dimensions[i];
    IterVar iv1 =
        IterVarNode::make(VarReplacer(op1_vsub).replace(orig_iv->dom),
                          Var("k1" + std::to_string(i), orig_iv->var.dtype()), orig_iv->iter_type);
    IterVar iv2 =
        IterVarNode::make(VarReplacer(op2_vsub).replace(orig_iv->dom),
                          Var("k2" + std::to_string(i), orig_iv->var.dtype()), orig_iv->iter_type);
    op1_red_axis.push_back(iv1);
    op2_red_axis.push_back(iv2);
    op1_vsub[orig_iv->var.get()] = iv1->var;
    op2_vsub[orig_iv->var.get()] = iv2->var;
  }

  Array<PrimExpr> op1_body;
  Array<PrimExpr> op2_body;
  Array<PrimExpr> op1_pred;
  Array<PrimExpr> op2_pred;
  VarReplacer op1_replacer(op1_vsub);
  VarReplacer op2_replacer(op2_vsub);

  auto replace = [&](PrimExpr e, VarReplacer& replacer, std::unordered_map<Tensor, Tensor>& vsub) {
    return te::ReplaceTensor(replacer(e), vsub);
  };

  for (size_t i = 0; i < compute_op->num_outputs(); ++i) {
    if (auto reduce = compute_op->body[i].as<tir::ReduceNode>()) {
      CHECK_EQ(compute_op->num_outputs(), 1)
          << "Splitting reduction ops with multiple outputs is not yet supported";

      Array<PrimExpr> op1_source;
      Array<PrimExpr> op2_source;
      for (auto s : reduce->source) {
        op1_source.push_back(op1_replacer(s));
        op2_source.push_back(op2_replacer(s));
      }

      op1_body.push_back(ReduceNode::make(reduce->combiner, op1_source, op1_red_axis,
                                          replace(reduce->condition, op1_replacer, graph1_vsub),
                                          reduce->value_index, reduce->dimensions));
      op2_body.push_back(ReduceNode::make(reduce->combiner, op2_source, op2_red_axis,
                                          replace(reduce->condition, op2_replacer, graph2_vsub),
                                          reduce->value_index, reduce->dimensions));
    } else {
      op1_body.push_back(replace(compute_op->body[i], op1_replacer, graph1_vsub));
      op2_body.push_back(replace(compute_op->body[i], op2_replacer, graph2_vsub));
    }
    op1_pred.push_back(replace(compute_op->pred[i], op1_replacer, graph1_vsub));
    op2_pred.push_back(replace(compute_op->pred[i], op2_replacer, graph2_vsub));
  }

  Modes op1_loop_layout =
      ModesNode::make_loop_layout(op1_dims, op1_maxes, op1_min_ufs, op1_ext_ufs);
  Modes op2_loop_layout =
      ModesNode::make_loop_layout(op2_dims, op2_maxes, op2_min_ufs, op2_ext_ufs);

  Operation op1 = ComputeOpNode::make(compute_op->name + "1", "", {}, op1_axis, op1_dims,
                                      compute_op->output_shape_storage, compute_op->storage_layouts,
                                      op1_loop_layout, op1_body, op1_pred);

  Operation op2 = ComputeOpNode::make(compute_op->name + "2", "", {}, op2_axis, op2_dims,
                                      compute_op->output_shape_storage, compute_op->storage_layouts,
                                      op2_loop_layout, op2_body, op2_pred);

  Stage s1(op1);
  Stage s2(op2);

  ArrayNode* stages = sch->stages.CopyOnWrite();
  size_t pos = FindNodeRef(stages, s);
  CHECK_LT(pos, stages->data.size());
  stages->data.erase(stages->data.begin() + pos);
  stages->data.insert(stages->data.begin() + pos, s1);
  stages->data.insert(stages->data.begin() + pos + 1, s2);

  MapNode* stage_map = sch->stage_map.CopyOnWrite();
  stage_map->data.erase(op);
  sch->stage_map.Set(op1, s1);
  sch->stage_map.Set(op2, s2);

  if (s->is_output) {
    s1->is_output = true;
    s2->is_output = true;
    ArrayNode* sch_outputs = sch->outputs.CopyOnWrite();
    pos = FindNodeRef(sch_outputs, s->origin_op);
    CHECK(pos < sch->outputs.size());
    sch_outputs->data.erase(sch_outputs->data.begin() + pos);
    sch_outputs->data.insert(sch_outputs->data.end(), op1);
    sch_outputs->data.insert(sch_outputs->data.end(), op2);
  }
  return Array<Operation>({op1, op2});
}

Array<Array<Operation>> SplitAGraph(Schedule& sch, Array<Operation> graph_ops,
                                    size_t to_split_index, UninterpFun split_point) {
  std::unordered_map<Tensor, Tensor> graph1_vsub;
  std::unordered_map<Tensor, Tensor> graph2_vsub;
  Array<Operation> graph1_ops;
  Array<Operation> graph2_ops;
  for (auto op : graph_ops) {
    auto new_ops = SplitALoop(sch, op, to_split_index, split_point, graph1_vsub, graph2_vsub);
    auto graph1_op = new_ops[0];
    auto graph2_op = new_ops[1];
    for (size_t i = 0; i < op->num_outputs(); ++i) {
      graph1_vsub[op.output(i)] = graph1_op.output(i);
      graph2_vsub[op.output(i)] = graph2_op.output(i);
    }
    graph1_ops.push_back(graph1_op);
    graph2_ops.push_back(graph2_op);
  }
  return Array<Array<Operation>>({graph1_ops, graph2_ops});
}

Array<Array<Operation>> Schedule::split_for_bin_packing(Array<Tensor> input_tensors,
                                                        Tensor output_tensor,
                                                        Map<IterVar, UninterpFun> to_split,
                                                        bool include_inputs) {
  std::unordered_map<size_t, UninterpFun> to_split_indices;
  auto compute_op = output_tensor->op.as<ComputeOpNode>();
  CHECK(compute_op) << "Loop splitting for bin packing is currently allowed only for ComputeOp";
  for (size_t i = 0; i < compute_op->axis.size(); ++i) {
    auto orig_iv = compute_op->axis[i];
    if (to_split.count(orig_iv)) {
      to_split_indices[i] = to_split.at(orig_iv);
    }
  }

  Array<Operation> graph_ops = GetSubGraph({output_tensor}, input_tensors, include_inputs);

  Array<Array<Operation>> ops = {graph_ops};
  for (auto it : to_split_indices) {
    Array<Array<Operation>> new_ops;
    for (auto op : ops) {
      new_ops.push_back_all(SplitAGraph(*this, op, it.first, it.second));
    }
    ops = new_ops;
  }
  return ops;
}

}  // namespace te
}  // namespace tvm

#undef COUT
