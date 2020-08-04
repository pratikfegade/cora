#include "rec_lowering.h"

#include <tvm/runtime/registry.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

#include "graph.h"
#include "schedule_utils.h"

namespace tvm {
namespace te {
class DynamicBatchingState {
 public:
  Var num_nodes;
  Var num_batches;
  Var max_batch_len;
  Var max_child_num;
  Var max_int_idx;
  Dimension batch_dim;
  Dimension node_in_batch_dim;
  Dimension child_pos_dim;
  Dimension node_dim;
  std::unordered_map<int, Dimension> child_dims;
  Tensor batch_lens;
  Tensor batch_starts;
  Tensor child_num;
  Tensor child_data;
  UninterpFun batch_uf;
  UninterpFun node_in_batch_uf;
  UninterpFun node_uf;
  UninterpFun child_pos_uf;

  DynamicBatchingState(Var num_nodes_, Var num_batches_, Var max_batch_len_, Var max_child_num_,
                       Var max_int_idx_)
      : num_nodes(num_nodes_),
        num_batches(num_batches_),
        max_batch_len(max_batch_len_),
        max_child_num(max_child_num_),
        max_int_idx(max_int_idx_) {
    batch_dim = DimensionNode::make("batch", DimensionNode::kRangeDim);
    node_in_batch_dim = DimensionNode::make("node_in_batch", DimensionNode::kRangeDim);
    node_dim = DimensionNode::make("node", DimensionNode::kFunDim);
    child_pos_dim = DimensionNode::make("child_pos", DimensionNode::kRangeDim);

    batch_uf = UninterpFunNode::from_constant("nbs", num_batches);

    batch_lens = PlaceholderOpNode::make(
                     "batch_lens", {num_batches}, DataType::Int(32), {batch_dim}, {batch_dim},
                     {IterVarNode::make(Range(0, num_batches), Var("batch", DataType::Int(32)),
                                        kDataPar, "")},
                     {batch_uf})
                     .output(0);

    batch_starts = PlaceholderOpNode::make(
                       "batch_starts", {num_batches}, DataType::Int(32), {batch_dim}, {batch_dim},
                       {IterVarNode::make(Range(0, num_batches), Var("batch", DataType::Int(32)),
                                          kDataPar, "")},
                       {UninterpFunNode::from_constant("nbs", num_batches)})
                       .output(0);

    auto batch_var = Var("bth", DataType::Int(32));
    node_in_batch_uf = UninterpFunNode::make("nidx", Range(0, max_batch_len), {batch_dim},
                                             {batch_var}, batch_lens[num_batches - 1 - batch_var]);

    auto batch_var2 = Var("bth", DataType::Int(32));
    auto node_pos = Var("nidx", DataType::Int(32));
    node_uf = UninterpFunNode::make("ldnd", Range(0, num_nodes), {batch_dim, node_in_batch_dim},
                                    {batch_var2, node_pos},
                                    batch_starts[num_batches - 1 - batch_var2] + node_pos);

    child_num =
        PlaceholderOpNode::make(
            "child_num", {num_nodes}, DataType::Int(32), {node_dim},
            {batch_dim, node_in_batch_dim, node_dim},
            {IterVarNode::make(Range(0, num_batches), Var("batch", DataType::Int(32)), kDataPar,
                               ""),
             IterVarNode::make(Range(0, max_batch_len), Var("nidx", DataType::Int(32)), kDataPar,
                               ""),
             IterVarNode::make(Range(0, num_nodes), Var("node", DataType::Int(32)), kDataPar, "")},
            {batch_uf, node_in_batch_uf, node_uf})
            .output(0);

    auto node_var = Var("node", DataType::Int(32));
    child_pos_uf = UninterpFunNode::make("nchild", Range(0, max_child_num), {node_dim}, {node_var},
                                         child_num[node_var]);

    child_data =
        PlaceholderOpNode::make(
            "child_data", {num_nodes, max_child_num}, DataType::Int(32), {node_dim, child_pos_dim},
            Array<Dimension>({batch_dim, node_in_batch_dim, node_dim, child_pos_dim}),
            {IterVarNode::make(Range(0, num_batches), Var("batch", DataType::Int(32)), kDataPar,
                               ""),
             IterVarNode::make(Range(0, max_batch_len), Var("nidx", DataType::Int(32)), kDataPar,
                               ""),
             IterVarNode::make(Range(0, num_nodes), Var("node", DataType::Int(32)), kDataPar, ""),
             IterVarNode::make(Range(0, max_child_num), Var("nchild", DataType::Int(32)), kDataPar,
                               "")},
            {batch_uf, node_in_batch_uf, node_uf, child_pos_uf})
            .output(0);
  }
};

class LowerDSIntrinsics : public ExprMutator {
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->call_type == CallNode::PureIntrinsic) {
      if (op->name == tvm::tir::intrinsic::tvm_get_child) {
        PrimExpr node = ExprMutator::VisitExpr(op->args[0]);
        PrimExpr idx = ExprMutator::VisitExpr(op->args[1]);
        auto imm = idx.as<IntImmNode>();
        CHECK(imm);

        Dimension dim;
        if (!dbs->child_dims.count(imm->value)) {
          dbs->child_dims[imm->value] =
              DimensionNode::make("child_dim" + std::to_string(imm->value), DimensionNode::kFunDim);
        }
        dim = dbs->child_dims.at(imm->value);

        if (extra_dims.count(dim)) {
          return extra_dims.at(dim)->iv->var;
        } else {
          auto name = "child" + std::to_string(imm->value);
          IterVar iv = IterVarNode::make(Range(0, dbs->num_nodes), Var(name, DataType::Int(32)),
                                         kDataPar, "");
          auto n_var = Var("node", DataType::Int(32));
          auto i_var = Var("idx", DataType::Int(32));
          auto uf = UninterpFunNode::make(
              name, Range(0, dbs->num_nodes), {dbs->node_dim, dbs->child_pos_dim}, {n_var, i_var},
              CallNode::make(DataType::Int(32), dbs->child_data->op->name, {n_var, i_var},
                             CallNode::Halide, {dbs->node_dim, dbs->child_pos_dim},
                             dbs->child_data->op, dbs->child_data->value_index));
          DimInfo di = DimInfoNode::make(dim, iv, uf);
          extra_dims.Set(dim, di);
          return iv->var;
        }
      } else if (op->name == tvm::tir::intrinsic::tvm_num_child) {
        PrimExpr node = ExprMutator::VisitExpr(op->args[0]);
        return CallNode::make(DataType::Int(32), dbs->child_num->op->name, {node}, CallNode::Halide,
                              {dbs->node_dim}, dbs->child_num->op, dbs->child_num->value_index);
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

 public:
  LowerDSIntrinsics(DynamicBatchingState* dbs_) : dbs(dbs_) {}

  DynamicBatchingState* dbs;
  Map<Dimension, DimInfo> extra_dims;
};

void ReplaceRecDataFlow(const Array<Operation>& ops, std::unordered_map<Tensor, Tensor>* vmap,
                        std::unordered_map<Tensor, Tensor>* rmap) {
  for (Operation old_op : ops) {
    Operation new_op = old_op->ReplaceInputs(old_op, *vmap);
    if (!new_op.same_as(old_op)) {
      std::cout << "[REPL] Replacing " << old_op << " with " << new_op << std::endl;
      for (int i = 0; i < new_op->num_outputs(); ++i) {
        auto it = rmap->find(old_op.output(i));
        if (it != rmap->end()) {
          (*vmap)[it->second] = new_op.output(i);
        } else {
          (*vmap)[old_op.output(i)] = new_op.output(i);
          (*rmap)[new_op.output(i)] = old_op.output(i);
        }
      }
    }
  }
}

Array<Array<Operation>, Array<Operation>> LowerDynamicBatching(Array<Operation> outputs,
                                                               Var num_nodes, Var num_batches,
                                                               Var max_batch_len, Var max_child_num,
                                                               Var max_int_idx) {
  CHECK_EQ(outputs.size(), 1) << "Only 1 output supported now";
  auto scan = outputs[0].as<ScanOpNode>();
  CHECK(scan) << "Only scan op output suported now";
  CHECK(scan->is_rec_op);

  DynamicBatchingState dbs(num_nodes, num_batches, max_batch_len, max_child_num, max_int_idx);

  Array<Tensor> inputs;
  for (Tensor t : scan->state_placeholder) {
    inputs.push_back(t);
  }
  for (Tensor t : scan->inputs) {
    inputs.push_back(t);
  }
  Array<Operation> ops = te::GetSubGraph(scan->update, inputs, false);

  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rmap;
  Array<Dimension> int_dims;
  for (auto ra_op : ops) {
    auto compute_op = ra_op.as<ComputeOpNode>();
    CHECK(compute_op) << "Only compute ops supported now";

    Array<IterVar> new_axis;
    IterVar batch_iv;
    IterVar node_in_batch_iv;
    IterVar node_iv;
    {
      batch_iv = IterVarNode::make(Range(0, dbs.num_batches), Var("batch", DataType::Int(32)),
                                   kDataPar, "");
      node_in_batch_iv = IterVarNode::make(
          Range(0, UninterpFun::MakeCallTo(dbs.node_in_batch_uf, {batch_iv->var}, {dbs.batch_dim})),
          Var("nidx", DataType::Int(32)), kDataPar, "");
      new_axis.push_back(batch_iv);
      new_axis.push_back(node_in_batch_iv);
      node_iv = compute_op->axis[0];
      for (size_t i = 1; i < compute_op->axis.size(); ++i) {
        new_axis.push_back(compute_op->axis[i]);
      }
    }

    Array<Dimension> new_root_index_dimensions;
    {
      size_t have_dim_num = int_dims.size();
      size_t need_dim_num = compute_op->axis.size() - 1;
      if (need_dim_num > have_dim_num) {
        for (size_t i = have_dim_num; i < need_dim_num; ++i) {
          int_dims.push_back(
              DimensionNode::make("h_dim" + std::to_string(i), DimensionNode::kRangeDim));
        }
      }

      new_root_index_dimensions.push_back(dbs.node_dim);
      for (size_t i = 0; i < need_dim_num; ++i) {
        new_root_index_dimensions.push_back(int_dims[i]);
      }
    }

    Array<DimInfo> new_dim_infos;
    {
      new_dim_infos.push_back(DimInfoNode::make(dbs.batch_dim, batch_iv, {}));
      new_dim_infos.push_back(DimInfoNode::make(dbs.node_in_batch_dim, node_in_batch_iv, {}));
      new_dim_infos.push_back(DimInfoNode::make(dbs.node_dim, node_iv, dbs.node_uf));

      for (size_t i = 1; i < compute_op->axis.size(); ++i) {
        new_dim_infos.push_back(DimInfoNode::make(int_dims[i - 1], new_axis[i + 1], {}));
      }
    }

    Array<PrimExpr> new_body;
    {
      LowerDSIntrinsics lower(&dbs);
      if (compute_op->body[0]->IsInstance<tir::ReduceNode>()) {
        // Specially handle reduce so the replaced op
        // still share all the components
        PrimExpr new_reduce = lower(compute_op->body[0]);
        if (!new_reduce.same_as(compute_op->body[0])) {
          const tir::ReduceNode* r = new_reduce.as<tir::ReduceNode>();
          for (size_t k = 0; k < compute_op->body.size(); ++k) {
            auto n = make_object<tir::ReduceNode>(*r);
            n->value_index = static_cast<int>(k);
            n->dtype = r->source[k].dtype();
            new_body.push_back(PrimExpr(n));
          }
        } else {
          new_body = compute_op->body;
        }
      } else {
        for (auto e : compute_op->body) {
          PrimExpr new_expr = lower(e);
          new_body.push_back(new_expr);
        }
      }

      for (auto it : lower.extra_dims) {
        new_dim_infos.push_back(it.second);
      }
    }

    Operation ila_op = ComputeOpNode::make(
        compute_op->name + ".ila", compute_op->tag, compute_op->attrs, new_axis,
        new_root_index_dimensions, compute_op->output_shape_storage, new_dim_infos, new_body, {});

    vmap[ra_op.output(0)] = ila_op.output(0);
    rmap[ila_op.output(0)] = ra_op.output(0);

    std::cout << "[RAILA] ILA Op " << ila_op << " " << ra_op << std::endl;
  }

  auto g = te::CreateReadGraph(outputs, true, false);
  Array<Operation> post_order = te::PostDFSOrder(outputs, g);
  // CHECK_EQ(ops.size(), post_order.size());
  Array<Operation> new_post_order;
  for (auto op : post_order) {
    if (vmap.count(op.output(0)))
      new_post_order.push_back(vmap.at(op.output(0))->op);
    else
      new_post_order.push_back(op);
  }
  ReplaceRecDataFlow(new_post_order, &vmap, &rmap);

  Array<Tensor> new_updates;
  for (auto t : scan->update) {
    new_updates.push_back(vmap[t]);
  }
  Operation ila_scan = ScanOpNode::make(
      scan->name + ".ila", scan->tag, scan->attrs, UninterpFunNode::from_constant("z", 0),
      UninterpFunNode::from_constant("nbtchs", dbs.num_batches), dbs.batch_dim, false, scan->init,
      new_updates, scan->state_placeholder, scan->inputs, {}, {}, {});
  return {{
              dbs.batch_lens->op,
              dbs.batch_starts->op,
              dbs.child_num->op,
              dbs.child_data->op,
          },
          {ila_scan}};
}

TVM_REGISTER_GLOBAL("te.LowerDynamicBatching")
    .set_body_typed([](Array<Operation> outputs, Var num_nodes, Var num_batches, Var max_batch_len,
                       Var max_child_num, Var max_int_idx) {
      return LowerDynamicBatching(outputs, num_nodes, num_batches, max_batch_len, max_child_num,
                                  max_int_idx);
    });

}  // namespace te
}  // namespace tvm
