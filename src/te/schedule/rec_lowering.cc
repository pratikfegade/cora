#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/rec_lowering.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

#include <map>

#include "../../tir/ir/var_replacer.h"
#include <tvm/tir/ir_pass.h>
#include "graph.h"
#include "schedule_utils.h"

namespace tvm {
namespace te {
ILAOps ILAOpsNode::make(Array<Tensor> ds_tensors, Array<Operation> outputs,
                        Map<Tensor, Array<Tensor>> ra_ila_mapping,
                        Map<std::string, Dimension> ds_dimensions) {
  ObjectPtr<ILAOpsNode> n = make_object<ILAOpsNode>();
  n->ds_tensors = ds_tensors;
  n->outputs = outputs;
  n->ra_ila_mapping = ra_ila_mapping;
  n->ds_dimensions = ds_dimensions;
  return ILAOps(n);
}

TVM_REGISTER_NODE_TYPE(ILAOpsNode);

class DSInfo {
 public:
  bool leaf_specialization;
  bool is_list;
  bool homogenous_batch;
  int batch_size;
  int length;

  DSInfo(bool leaf_specialization_, bool is_list_, bool homogenous_batch_, int batch_size_,
         int length_)
      : leaf_specialization(leaf_specialization_),
        is_list(is_list_),
        homogenous_batch(homogenous_batch_),
        batch_size(batch_size_),
        length(length_) {}
};

class DynamicBatchingState {
 public:
  Var num_nodes;
  // Var num_batches;
  PrimExpr num_batches;
  Var max_batch_len;
  Var max_child_num;
  Var max_int_idx;
  Dimension batch_dim;
  Dimension node_in_batch_dim;
  Dimension child_pos_dim;
  Dimension node_dim;
  std::map<int, Dimension> child_dims;
  std::map<int, Tensor> child_tensors;
  Tensor batch_lens;
  Tensor batch_starts;
  Tensor child_num;
  Tensor child_data;
  UninterpFun batch_uf;
  UninterpFun node_in_batch_uf;
  UninterpFun node_uf;
  UninterpFun child_pos_uf;

  DynamicBatchingState(Var num_nodes_, Var num_batches_, Var max_batch_len_, Var max_child_num_,
                       Var max_int_idx_, const DSInfo* p_dsInfo)
      : num_nodes(num_nodes_),
        num_batches(num_batches_),
        max_batch_len(max_batch_len_),
        max_child_num(max_child_num_),
        max_int_idx(max_int_idx_) {
    const DSInfo& dsInfo = *p_dsInfo;
    bool rectangle = dsInfo.homogenous_batch && dsInfo.is_list;

    if (rectangle) {
      num_batches = dsInfo.length;
    }

    batch_dim = DimensionNode::make("batch", DimensionNode::kRangeDim);
    node_in_batch_dim = DimensionNode::make("node_in_batch", DimensionNode::kRangeDim);
    node_dim = DimensionNode::make("node", DimensionNode::kFunDim);
    child_pos_dim = DimensionNode::make("child_pos", DimensionNode::kRangeDim);

    if (rectangle)
      batch_uf = UninterpFunNode::from_constant("nbs", dsInfo.length);
    else
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
    if (rectangle)
      node_in_batch_uf =
          UninterpFunNode::make("nidx", Range(0, dsInfo.batch_size), {}, {}, dsInfo.batch_size);
    else
      node_in_batch_uf =
          UninterpFunNode::make("nidx", Range(0, max_batch_len), {batch_dim}, {batch_var},
                                batch_lens[num_batches - 1 - batch_var]);

    auto batch_var2 = Var("bth", DataType::Int(32));
    auto node_pos = Var("nidx", DataType::Int(32));
    if (rectangle)
      node_uf = UninterpFunNode::make(
          "ldnd", Range(0, num_nodes), {batch_dim, node_in_batch_dim}, {batch_var2, node_pos},
          (num_batches - 1 - batch_var2) * dsInfo.batch_size + node_pos);
    else
      node_uf = UninterpFunNode::make("ldnd", Range(0, num_nodes), {batch_dim, node_in_batch_dim},
                                      {batch_var2, node_pos},
                                      batch_starts[num_batches - 1 - batch_var2] + node_pos);

    auto batch_iv =
        IterVarNode::make(Range(0, num_batches), Var("batch", DataType::Int(32)), kDataPar, "");
    auto node_in_batch_iv = IterVarNode::make(
        Range(0, UninterpFun::MakeCallTo(node_in_batch_uf, {batch_iv->var}, {batch_dim})),
        Var("nidx", DataType::Int(32)), kDataPar, "");
    auto node_iv =
        IterVarNode::make(Range(0, num_nodes), Var("node", DataType::Int(32)), kDataPar, "");
    child_num = PlaceholderOpNode::make("child_num", {num_nodes}, DataType::Int(32), {node_dim},
                                        {batch_dim, node_in_batch_dim, node_dim},
                                        {batch_iv, node_in_batch_iv, node_iv},
                                        {batch_uf, node_in_batch_uf, node_uf})
                    .output(0);

    auto node_var = Var("node", DataType::Int(32));
    child_pos_uf = UninterpFunNode::make("nchild", Range(0, max_child_num), {node_dim}, {node_var},
                                         child_num[node_var]);

    auto batch_iv1 =
        IterVarNode::make(Range(0, num_batches), Var("batch", DataType::Int(32)), kDataPar, "");
    auto node_in_batch_iv1 = IterVarNode::make(
        Range(0, UninterpFun::MakeCallTo(node_in_batch_uf, {batch_iv1->var}, {batch_dim})),
        Var("nidx", DataType::Int(32)), kDataPar, "");
    auto node_iv1 =
        IterVarNode::make(Range(0, num_nodes), Var("node", DataType::Int(32)), kDataPar, "");
    auto child_iv =
        IterVarNode::make(Range(0, max_child_num), Var("nchild", DataType::Int(32)), kDataPar, "");
    child_data =
        PlaceholderOpNode::make(
            "child_data", {num_nodes, max_child_num}, DataType::Int(32), {node_dim, child_pos_dim},
            Array<Dimension>({batch_dim, node_in_batch_dim, node_dim, child_pos_dim}),
            {batch_iv1, node_in_batch_iv1, node_iv1, child_iv},
            {batch_uf, node_in_batch_uf, node_uf, child_pos_uf})
            .output(0);
  }

  std::pair<Dimension, Tensor> getChildTensorAndDim(int idx) {
    if (child_dims.count(idx)) {
      CHECK(child_tensors.count(idx));
    } else {
      child_dims[idx] =
          DimensionNode::make("child_dim" + std::to_string(idx), DimensionNode::kFunDim);
      auto batch_iv =
          IterVarNode::make(Range(0, num_batches), Var("batch", DataType::Int(32)), kDataPar, "");
      auto node_in_batch_iv = IterVarNode::make(
          Range(0, UninterpFun::MakeCallTo(node_in_batch_uf, {batch_iv->var}, {batch_dim})),
          Var("nidx", DataType::Int(32)), kDataPar, "");
      auto node_iv =
          IterVarNode::make(Range(0, num_nodes), Var("node", DataType::Int(32)), kDataPar, "");
      child_tensors[idx] =
          PlaceholderOpNode::make(
              "child_data" + std::to_string(idx), {num_nodes}, DataType::Int(32), {node_dim},
              Array<Dimension>({batch_dim, node_in_batch_dim, node_dim}),
              {batch_iv, node_in_batch_iv, node_iv}, {batch_uf, node_in_batch_uf, node_uf})
              .output(0);
    }
    return std::make_pair(child_dims.at(idx), child_tensors.at(idx));
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

        const DSInfo& dsInfo = *p_dsInfo;
        if (dsInfo.homogenous_batch && dsInfo.is_list) {
          return node + dsInfo.batch_size;
        } else {
          auto pair = dbs->getChildTensorAndDim(imm->value);
          Dimension child_dim = pair.first;
          Tensor child_tensor = pair.second;

          if (extra_dims.count(child_dim)) {
            return extra_dims.at(child_dim)->iv->var;
          } else {
            auto name = "child" + std::to_string(imm->value);
            IterVar iv = IterVarNode::make(Range(0, dbs->num_nodes), Var(name, DataType::Int(32)),
                                           kDataPar, "");
            auto n_var = Var("node", DataType::Int(32));
            auto uf = UninterpFunNode::make(
                name, Range(0, dbs->num_nodes), {dbs->node_dim}, {n_var},
                CallNode::make(DataType::Int(32), child_tensor->op->name, {n_var}, CallNode::Halide,
                               {dbs->node_dim}, child_tensor->op, child_tensor->value_index));
            extra_dims.Set(child_dim, DimInfoNode::make(child_dim, iv, uf));
            return iv->var;
          }
        }
      } else if (op->name == tvm::tir::intrinsic::tvm_num_child) {
        PrimExpr node = ExprMutator::VisitExpr(op->args[0]);
        return CallNode::make(DataType::Int(32), dbs->child_num->op->name, {node}, CallNode::Halide,
                              {dbs->node_dim}, dbs->child_num->op, dbs->child_num->value_index);
      } else if (op->name == tvm::tir::intrinsic::tvm_is_leaf) {
        PrimExpr node = ExprMutator::VisitExpr(op->args[0]);

        if (node.same_as(op_current_node_var) && scan_range == kLeavesOnly)
          return IntImm(DataType::Bool(), 1);
        else if (node.same_as(op_current_node_var) && scan_range == kNoLeaves)
          return IntImm(DataType::Bool(), 0);
        else
          return node > dbs->max_int_idx;
      } else if (op->name == tvm::tir::intrinsic::tvm_if_then_else) {
        PrimExpr condition = this->VisitExpr(op->args[0]);
        PrimExpr b1 = this->VisitExpr(op->args[1]);
        PrimExpr b2 = this->VisitExpr(op->args[2]);
	// std::cout << "[LOWERING_IF] "  << GetRef<PrimExpr>(op)<< std::endl;
        if (ana.CanProve(condition == 1))
          // return this->VisitExpr(op->args[1]);
          return this->VisitExpr(b1);
        else if (ana.CanProve(condition == 0))
          // return this->VisitExpr(op->args[2]);
          return this->VisitExpr(b2);
        else if (ana.CanProve(b1 == b2))
	  // return op->args[1];
	  return b1;
	else
          return ExprMutator::VisitExpr_(op);
      }
    } else if (op->call_type == CallNode::Halide) {
      auto callee = op->func.as<OperationNode>();
      if (node_independent_ops.count(callee)) {
        Array<PrimExpr> new_args;
        Array<Dimension> new_dims;
        for (size_t i = 1; i < op->args.size(); ++i) {
          new_args.push_back(this->VisitExpr(op->args[i]));
          // new_dims.push_back(op->argument_dimensions[i]);
        }
        auto ret = CallNode::make(op->dtype, op->name, new_args, op->call_type, new_dims, op->func,
                                  op->value_index);
        // std::cout << "[RAL] Calling node independent op " << GetRef<PrimExpr>(op) << " " << ret
        // << std::endl;
        return ret;
      }
      // else if (zero_ops.count(callee)) {
      // 	return 0;
      // }
    }
    return ExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) {
    auto it = vsub.find(op);
    if (it != vsub.end()) return it->second;
    return GetRef<PrimExpr>(op);
  }

 public:
  LowerDSIntrinsics(DynamicBatchingState* dbs_, ScanRange scan_range_, Var op_current_node_var_,
                    std::unordered_map<const VarNode*, PrimExpr> vsub_, const DSInfo* p_dsInfo_,
                    const std::unordered_set<const Object*>& node_independent_ops_,
                    const std::unordered_set<const Object*>& zero_ops_)
      : dbs(dbs_),
        scan_range(scan_range_),
        op_current_node_var(op_current_node_var_),
        vsub(vsub_),
        p_dsInfo(p_dsInfo_),
        node_independent_ops(node_independent_ops_),
        zero_ops(zero_ops_) {}

  arith::Analyzer ana;
  DynamicBatchingState* dbs;
  ScanRange scan_range;
  Map<Dimension, DimInfo> extra_dims;
  Var op_current_node_var;
  std::unordered_map<const VarNode*, PrimExpr> vsub;
  const DSInfo* p_dsInfo;
  const std::unordered_set<const Object*>& node_independent_ops;
  const std::unordered_set<const Object*>& zero_ops;
};

std::pair<PrimExpr, PrimExpr> getBatchRange(const DynamicBatchingState* dbs, ScanRange scan_range) {
  switch (scan_range) {
    case kAll:
      return std::make_pair(0, dbs->num_batches);
    case kLeavesOnly:
      return std::make_pair(0, 1);
    case kRootsOnly:
      return std::make_pair(dbs->num_batches - 1, dbs->num_batches);
    case kNoLeaves:
      return std::make_pair(1, dbs->num_batches);
    default:
      CHECK(false);
      return {};
  }
}

std::pair<UninterpFun, UninterpFun> getScanRangeUFs(const DynamicBatchingState* dbs,
                                                    ScanRange scan_range) {
  auto p = getBatchRange(dbs, scan_range);
  return std::make_pair(UninterpFunNode::from_constant("min", p.first),
                        UninterpFunNode::from_constant("max", p.second));
}

void ReplaceRecDataFlow(const Array<Operation>& ops, std::unordered_map<Tensor, Tensor>* vmap,
                        std::unordered_map<Tensor, Tensor>* rmap) {
  for (Operation old_op : ops) {
    Operation new_op = old_op->ReplaceInputs(old_op, *vmap);
    if (!new_op.same_as(old_op)) {
      // std::cout << "[REPL] Replacing " << old_op << " with " << new_op << std::endl;
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

size_t num_outputs(Operation op) { return static_cast<size_t>(op->num_outputs()); }

Map<Operation, Operation> LowerDynBatchInternal(Array<Operation> outputs,
                                                DynamicBatchingState* p_dbs, ScanRange scan_range,
                                                bool scan_init_separate, const DSInfo* p_dsInfo,
                                                std::string prefix = "") {
  const DSInfo& dsInfo = *p_dsInfo;
  CHECK_EQ(outputs.size(), 1) << "Only 1 output supported now";
  auto scan = outputs[0].as<ScanOpNode>();
  CHECK(scan) << "Only scan op output suported now";
  CHECK(scan->is_rec_op);

  DynamicBatchingState& dbs = *p_dbs;

  Array<Tensor> inputs;
  for (Tensor t : scan->state_placeholder) {
    inputs.push_back(t);
  }
  for (Tensor t : scan->inputs) {
    inputs.push_back(t);
  }

  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rmap;

  Array<Dimension> int_dims;
  Array<Tensor> new_state;
  {
    for (size_t i = 0; i < num_outputs(outputs[0]); ++i) {
      Tensor old_state = scan->state_placeholder[i];

      Array<Dimension> dims;
      Array<Dimension> root_dims;
      Array<IterVar> ivs;
      Array<UninterpFun> ufs;
      Array<PrimExpr> shape;

      auto batch_iv = IterVarNode::make(Range(0, dbs.num_batches), Var("batch", DataType::Int(32)),
                                        kDataPar, "");
      auto node_in_batch_iv = IterVarNode::make(
          Range(0, UninterpFun::MakeCallTo(dbs.node_in_batch_uf, {batch_iv->var}, {dbs.batch_dim})),
          Var("nidx", DataType::Int(32)), kDataPar, "");
      auto node_iv =
          IterVarNode::make(Range(0, dbs.num_nodes), Var("node", DataType::Int(32)), kDataPar, "");

      dims.push_back(dbs.batch_dim);
      dims.push_back(dbs.node_in_batch_dim);
      dims.push_back(dbs.node_dim);
      root_dims.push_back(dbs.node_dim);

      ivs.push_back(batch_iv);
      ivs.push_back(node_in_batch_iv);
      ivs.push_back(node_iv);

      ufs.push_back(dbs.batch_uf);
      ufs.push_back(dbs.node_in_batch_uf);
      ufs.push_back(dbs.node_uf);

      size_t have_dim_num = int_dims.size();
      size_t need_dim_num = old_state.ndim() - 1;
      if (need_dim_num > have_dim_num) {
        for (size_t i = have_dim_num; i < need_dim_num; ++i) {
          int_dims.push_back(
              DimensionNode::make("h_dim" + std::to_string(i), DimensionNode::kRangeDim));
        }
      }

      shape.push_back(dbs.num_nodes);
      for (size_t j = 1; j < old_state.ndim(); ++j) {
        PrimExpr this_shape = old_state->op->output_shape(0)[j];
        dims.push_back(int_dims[j - 1]);
        root_dims.push_back(int_dims[j - 1]);
        auto new_iv =
            IterVarNode::make(Range(0, this_shape), Var("lv", DataType::Int(32)), kDataPar, "");
        auto uf = UninterpFunNode::from_constant("c", this_shape);
        ivs.push_back(new_iv);
        ufs.push_back(uf);
        shape.push_back(this_shape);
      }

      auto this_state =
          PlaceholderOpNode::make(prefix + old_state->op->name + ".ila", shape,
                                  old_state->op->output_dtype(0), root_dims, dims, ivs, ufs)
              .output(0);
      new_state.push_back(this_state);
      vmap[old_state] = this_state;
      rmap[this_state] = old_state;
    }
  }

  Array<Operation> ops = te::GetSubGraph(scan->update, inputs, false);
  std::unordered_set<const Object*> node_independent_ops;
  std::unordered_set<const Object*> zero_ops;
  for (auto ra_op : ops) {
    auto compute_op = ra_op.as<ComputeOpNode>();
    CHECK(compute_op) << "Only compute ops supported now";

    IterVar node_iv = compute_op->axis[0];

    Array<DimInfo> extra_dims;
    std::unordered_map<const VarNode*, PrimExpr> iv_vsub;

    Array<IterVar> new_reduce_axis;
    {
      for (size_t i = 0; i < compute_op->reduce_axis.size(); ++i) {
        IterVar old_iv = compute_op->reduce_axis[i];
        IterVar new_iv = IterVarNode::make(old_iv->dom, old_iv->var.copy_with_suffix(".ila"),
                                           old_iv->iter_type, old_iv->thread_tag);
        new_reduce_axis.push_back(new_iv);
        iv_vsub[old_iv->var.as<VarNode>()] = new_iv->var;
      }
    }

    Array<IterVar> new_hidden_axis;
    {
      for (size_t i = 1; i < compute_op->axis.size(); ++i) {
        IterVar old_iv = compute_op->axis[i];
        IterVar new_iv = IterVarNode::make(old_iv->dom, old_iv->var.copy_with_suffix(".ila"),
                                           old_iv->iter_type, old_iv->thread_tag);
        new_hidden_axis.push_back(new_iv);
        iv_vsub[old_iv->var.as<VarNode>()] = new_iv->var;
      }
    }

    Array<PrimExpr> new_body;
    Array<PrimExpr> new_pred;
    {
      if (dsInfo.homogenous_batch && dsInfo.is_list) {
        iv_vsub[dbs.num_batches.as<VarNode>()] = dsInfo.length;
      }
      LowerDSIntrinsics lower(&dbs, scan_range, node_iv->var, iv_vsub, p_dsInfo,
                              node_independent_ops, zero_ops);
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
            n->axis = new_reduce_axis;
            new_body.push_back(PrimExpr(n));
            // std::cout << "NEWVODY " << ra_op->name << " " << PrimExpr(n) << std::endl;
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

      if (compute_op->pred.defined()) {
        for (auto e : compute_op->pred) {
          CHECK(e.dtype().is_bool()) << e << " " << e.dtype() << " " << compute_op->name;
          PrimExpr new_expr = lower(e);
          new_pred.push_back(new_expr);
        }
      }

      for (auto it : lower.extra_dims) {
        extra_dims.push_back(it.second);
      }
    }

    bool body_zero = false;
    {
      arith::Analyzer ana;
      Array<PrimExpr> body_rhs = new_body;
      if (auto r = new_body[0].as<ReduceNode>()) {
	body_rhs = r->source;
      }

      bool non_zero = false;;
      for (auto e: body_rhs) {
	if (!ana.CanProve(e == 0)) {
	  non_zero = true;
	}
	// std::cout << "[RAL] Act body " << ra_op->name << " " << arith::Simplify(e) << std::endl;
      }
      if (!non_zero) {
	body_zero = true;
      }
      if (body_zero) {
        zero_ops.insert(compute_op);
        // node_independent_ops.insert(compute_op);
      }
      if (body_zero) {
	Array<PrimExpr> zero_body;
	for (auto e: body_rhs) {
	  zero_body.push_back(0.0f);
	}
	new_body = zero_body;
      }
    }

    // std::cout << "[RAL] Body zero " << ra_op->name << " " << body_zero << std::endl;


    // Check if the body of the operation depends on nodes. If not, we
    // can hoist it out
    bool node_independent = false;
    if (!body_zero) {
      arith::Analyzer ana;
      tir::VarCollector collector;
      for (auto e : new_body) {
        collector(UninterpFun::InlineUninterpFunCalls(e));
      }
      bool all_pred_false = true;
      for (auto e : new_pred) {
        if (!ana.CanProve(!e)) all_pred_false = false;
      }
      for (auto e : new_pred) {
        collector(UninterpFun::InlineUninterpFunCalls(e));
      }
      if (!all_pred_false) {
        node_independent = true;
        if (collector.getCollected().count(node_iv->var.as<VarNode>())) node_independent = false;
        for (auto di : extra_dims) {
          if (collector.getCollected().count(di->iv->var.as<VarNode>())) node_independent = false;
        }
      }
      if (node_independent) {
        node_independent_ops.insert(compute_op);
      }
    }

    Array<IterVar> new_axis;
    IterVar batch_iv;
    IterVar node_in_batch_iv;
    {
      if (!node_independent) {
        auto p = getBatchRange(p_dbs, scan_range);
        batch_iv = IterVarNode::make(Range(p.first, p.second), Var("batch", DataType::Int(32)),
                                     kDataPar, "");
        node_in_batch_iv = IterVarNode::make(
            Range(0,
                  UninterpFun::MakeCallTo(dbs.node_in_batch_uf, {batch_iv->var}, {dbs.batch_dim})),
            Var("nidx", DataType::Int(32)), kDataPar, "");
        new_axis.push_back(batch_iv);
        new_axis.push_back(node_in_batch_iv);
      }
      for (auto iv : new_hidden_axis) {
        new_axis.push_back(iv);
      }
    }

    Array<DimInfo> new_dim_infos;
    {
      size_t have_dim_num = int_dims.size();
      size_t need_dim_num = compute_op->axis.size() - 1;
      if (need_dim_num > have_dim_num) {
        for (size_t i = have_dim_num; i < need_dim_num; ++i) {
          int_dims.push_back(
              DimensionNode::make("h_dim" + std::to_string(i), DimensionNode::kRangeDim));
        }
      }

      CHECK(int_dims.size() >= compute_op->axis.size() - 1)
          << int_dims.size() << " " << compute_op->axis.size();
      if (!node_independent) {
        new_dim_infos.push_back(DimInfoNode::make(dbs.batch_dim, batch_iv, {}));
        new_dim_infos.push_back(DimInfoNode::make(dbs.node_in_batch_dim, node_in_batch_iv, {}));
        new_dim_infos.push_back(DimInfoNode::make(dbs.node_dim, node_iv, dbs.node_uf));
        for (size_t i = 1; i < compute_op->axis.size(); ++i) {
          new_dim_infos.push_back(DimInfoNode::make(int_dims[i - 1], new_axis[i + 1], {}));
          // if (!int_dims[i - 1].defined()) {
          // std::cout << "[RAL] Empty dim1 for " << ra_op << std::endl;
          // }
        }
      } else {
        for (size_t i = 1; i < compute_op->axis.size(); ++i) {
          new_dim_infos.push_back(DimInfoNode::make(int_dims[i - 1], new_axis[i - 1], {}));
          // if (!int_dims[i - 1].defined()) {
          // std::cout << "[RAL] Empty dim2 for " << ra_op << std::endl;
          // }
        }
      }
      for (auto di : extra_dims) {
        new_dim_infos.push_back(di);
        // if (!di.defined()) {
        // std::cout << "[RAL] Empty dim3 for " << ra_op << std::endl;
        // }
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

      if (!node_independent) {
        new_root_index_dimensions.push_back(dbs.node_dim);
      }
      for (size_t i = 0; i < need_dim_num; ++i) {
        new_root_index_dimensions.push_back(int_dims[i]);
      }
    }

    Array<PrimExpr> new_shape;
    {
      if (node_independent) {
        for (size_t i = 1; i < compute_op->output_shape_storage.size(); ++i) {
          new_shape.push_back(compute_op->output_shape_storage[i]);
        }
      } else {
        new_shape = compute_op->output_shape_storage;
      }
    }

    CHECK_EQ(new_root_index_dimensions.size(), new_shape.size());

    // for (auto di : new_dim_infos) {
    //   std::cout << "[DIM]   dim for " << ra_op->name << " " << di->dim  << std::endl;
    // }

    Operation ila_op = ComputeOpNode::make(prefix + compute_op->name + ".ila", compute_op->tag,
                                           compute_op->attrs, new_axis, new_root_index_dimensions,
                                           new_shape, new_dim_infos, new_body, {new_pred});
    // std::cout << "[RAL] Created " << ila_op << " " << new_shape.size() << " " << new_axis
    // << std::endl;
    vmap[ra_op.output(0)] = ila_op.output(0);
    rmap[ila_op.output(0)] = ra_op.output(0);
  }

  auto scan_ufs = getScanRangeUFs(&dbs, scan_range);

  Operation ila_scan;
  {
    auto n = make_object<ScanOpNode>();
    n->scan_dim = dbs.batch_dim;
    n->name = scan->name + ".ila";
    n->tag = scan->tag;
    n->attrs = scan->attrs;
    n->init_separate = scan_init_separate;
    n->init = scan->init;
    n->update = scan->update;
    n->state_placeholder = new_state;
    n->inputs = scan->inputs;

    ila_scan = Operation(n);
  }

  vmap[outputs[0].output(0)] = ila_scan.output(0);
  // vmap[ila_scan.output(0)] = outputs[0].output(0);

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

  ScanOpNode* mut_new_scan = const_cast<ScanOpNode*>(vmap[ila_scan.output(0)]->op.as<ScanOpNode>());
  mut_new_scan->scan_axis =
      mut_new_scan->RefreshDimVarMappings(scan_ufs.first, scan_ufs.second, {}, {}, {});

  Map<Operation, Operation> ra_ila_mapping;
  for (auto it : vmap) {
    ra_ila_mapping.Set(it.first->op, it.second->op);
  }
  ra_ila_mapping.Set(outputs[0], vmap[ila_scan.output(0)]->op);

  return ra_ila_mapping;
}

ILAOps LowerDynamicBatching(Array<Operation> outputs, Var num_nodes, Var num_batches,
                            Var max_batch_len, Var max_child_num, Var max_int_idx,
                            bool leaf_specialization, bool is_list, bool homogenous_batch,
                            int batch_size, int length, ScanRange scan_range) {
  DSInfo* dsInfo = new DSInfo(leaf_specialization, is_list, homogenous_batch, batch_size, length);
  static DynamicBatchingState dbs(num_nodes, num_batches, max_batch_len, max_child_num, max_int_idx,
                                  dsInfo);

  Map<Tensor, Array<Tensor>> ret_mapping;
  Array<Operation> new_outputs;
  if (leaf_specialization) {
    CHECK(scan_range == kAll);
    auto leaf_mapping = LowerDynBatchInternal(outputs, &dbs, kLeavesOnly, false, dsInfo, "l");
    auto int_mapping = LowerDynBatchInternal(outputs, &dbs, kNoLeaves, true, dsInfo, "i");

    Array<Operation> all_ops;
    for (auto it : leaf_mapping) {
      if (!all_ops.Contains(it.first) && it.first != outputs[0]) {
        all_ops.push_back(it.first);
      }
    }
    for (auto it : int_mapping) {
      if (!all_ops.Contains(it.first) && it.first != outputs[0]) {
        all_ops.push_back(it.first);
      }
    }

    Map<Tensor, Array<Tensor>> full_ra_ila_mapping;
    for (auto op : all_ops) {
      Array<Operation> new_ops;
      if (leaf_mapping.count(op)) {
        new_ops.push_back(leaf_mapping.at(op));
      }

      if (int_mapping.count(op)) {
        new_ops.push_back(int_mapping.at(op));
      }

      for (size_t i = 0; i < num_outputs(op); ++i) {
        Array<Tensor> new_ts;
        for (auto new_op : new_ops) {
          new_ts.push_back(new_op.output(i));
        }
        full_ra_ila_mapping.Set(op.output(i), new_ts);
      }
    }

    const ScanOpNode* leaf_scan = leaf_mapping.at(outputs[0]).as<ScanOpNode>();
    // std::cout << "[LEAFSCAB] " << leaf_mapping.at(outputs[0]) << std::endl;
    CHECK(leaf_scan) << leaf_mapping.at(outputs[0]);
    const ScanOpNode* int_scan = int_mapping.at(outputs[0]).as<ScanOpNode>();
    CHECK(int_scan);
    Operation new_scan_op;
    {
      Array<Tensor> new_inits;
      // for (size_t i = 0; i < num_outputs(leaf_scan); ++i) {
      for (size_t i = 0; i < num_outputs(outputs[0]); ++i) {
        new_inits.push_back(leaf_mapping.at(outputs[0]).output(i));
      }

      auto n = make_object<ScanOpNode>();
      n->name = int_scan->name + ".cum";
      n->tag = int_scan->tag;
      n->attrs = int_scan->attrs;

      n->dim2var_maps = int_scan->dim2var_maps;
      n->var2dim_map = int_scan->var2dim_map;

      n->scan_axis = int_scan->scan_axis;
      n->explicit_dims = int_scan->explicit_dims;
      n->explicit_loop_ivs = int_scan->explicit_loop_ivs;
      // n->init = new_inits;
      n->init = leaf_scan->update;
      n->update = int_scan->update;
      n->state_placeholder = int_scan->state_placeholder;
      n->inputs = int_scan->inputs;
      n->scan_dim = int_scan->scan_dim;
      n->init_separate = int_scan->init_separate;

      n->spatial_dimensions_ = int_scan->spatial_dimensions_;
      n->spatial_axis_ = int_scan->spatial_axis_;

      new_scan_op = Operation(n);
    }
    for (size_t i = 0; i < num_outputs(outputs[0]); ++i) {
      auto t = new_scan_op.output(i);
      full_ra_ila_mapping.Set(outputs[0].output(i), {t});
    }
    new_outputs.push_back(new_scan_op);
    ret_mapping = full_ra_ila_mapping;
  } else {
    auto op_mapping = LowerDynBatchInternal(outputs, &dbs, scan_range, false, dsInfo);
    new_outputs.push_back(op_mapping.at(outputs[0]));

    for (auto it : op_mapping) {
      for (size_t i = 0; i < num_outputs(it.first); ++i) {
        ret_mapping.Set(it.first.output(i), {it.second.output(i)});
      }
    }
  }
  Array<Tensor> ds_ops;
  ds_ops.push_back(dbs.batch_lens);
  ds_ops.push_back(dbs.batch_starts);
  ds_ops.push_back(dbs.child_num);

  for (auto it : dbs.child_tensors) {
    ds_ops.push_back(it.second);
  }

  // for (auto it : ret_mapping) {
  //   std::cout << "[ReT] " << it.first->op << " " << it.first->value_index << std::endl;
  //   for (auto nt : it.second) {
  //     std::cout << "[ReT]   " << nt->op << " " << nt->value_index << std::endl;
  //   }
  // }

  Map<std::string, Dimension> ds_dims;
  ds_dims.Set(dbs.batch_dim->name, dbs.batch_dim);
  ds_dims.Set(dbs.node_in_batch_dim->name, dbs.node_in_batch_dim);
  ds_dims.Set(dbs.node_dim->name, dbs.node_dim);

  return ILAOpsNode::make(ds_ops, new_outputs, ret_mapping, ds_dims);
}

TVM_REGISTER_GLOBAL("te.LowerDynamicBatching")
    .set_body_typed([](Array<Operation> outputs, Var num_nodes, Var num_batches, Var max_batch_len,
                       Var max_child_num, Var max_int_idx, bool leaf_specialization, bool is_list,
                       bool homogenous_batch, int batch_size, int length, int scan_range) {
      return LowerDynamicBatching(outputs, num_nodes, num_batches, max_batch_len, max_child_num,
                                  max_int_idx, leaf_specialization, is_list, homogenous_batch,
                                  batch_size, length, static_cast<ScanRange>(scan_range));
    });

class StaticBatchingState {
 public:
  Var num_nodes;
  PrimExpr num_trees;
  Var max_tree_len;
  Var max_child_num;
  Dimension tree_dim;
  Dimension node_in_tree_dim;
  Dimension node_dim;
  std::map<int, Dimension> child_dims;
  std::map<int, Tensor> child_tensors;
  Tensor tree_lens;
  Tensor tree_data;
  Tensor child_num;
  UninterpFun tree_uf;
  UninterpFun node_in_tree_uf;
  UninterpFun node_uf;

  StaticBatchingState(Var num_nodes_, PrimExpr num_trees_, Var max_tree_len_, Var max_child_num_)
      : num_nodes(num_nodes_),
        num_trees(num_trees_),
        max_tree_len(max_tree_len_),
        max_child_num(max_child_num_) {
    tree_dim = DimensionNode::make("batch", DimensionNode::kRangeDim);
    node_in_tree_dim = DimensionNode::make("node_in_batch", DimensionNode::kRangeDim);
    node_dim = DimensionNode::make("node", DimensionNode::kFunDim);

    tree_uf = UninterpFunNode::from_constant("nbs", num_trees);

    tree_lens =
        PlaceholderOpNode::make(
            "t_l", {num_trees}, DataType::Int(32), {tree_dim}, {tree_dim},
            {IterVarNode::make(Range(0, num_trees), Var("batch", DataType::Int(32)), kDataPar, "")},
            {tree_uf})
            .output(0);

    Var tree_var = Var("tree", DataType::Int(32));
    node_in_tree_uf = UninterpFunNode::make("nt", Range(0, max_tree_len), {tree_dim}, {tree_var},
                                            tree_lens[tree_var]);
    tree_data = PlaceholderOpNode::make(
                    "t_d", {num_trees, max_tree_len}, DataType::Int(32),
                    {tree_dim, node_in_tree_dim}, {tree_dim, node_in_tree_dim},
                    {IterVarNode::make(Range(0, num_trees), tree_var, kDataPar, ""),
                     IterVarNode::make(
                         Range(0, UninterpFun::MakeCallTo(node_in_tree_uf, {tree_var}, {tree_dim})),
                         Var("nt", DataType::Int(32)), kDataPar, "")},
                    {tree_uf, node_in_tree_uf})
                    .output(0);

    auto tree_var2 = Var("tree", DataType::Int(32));
    auto node_pos = Var("nidx", DataType::Int(32));
    node_uf = UninterpFunNode::make("ldnd", Range(0, num_nodes), {tree_dim, node_in_tree_dim},
                                    {tree_var2, node_pos}, tree_data(tree_var2, node_pos));

    auto tree_iv =
        IterVarNode::make(Range(0, num_trees), Var("tree", DataType::Int(32)), kDataPar, "");
    auto node_in_tree_iv = IterVarNode::make(
        Range(0, UninterpFun::MakeCallTo(node_in_tree_uf, {tree_iv->var}, {tree_dim})),
        Var("nidx", DataType::Int(32)), kDataPar, "");
    auto node_iv =
        IterVarNode::make(Range(0, num_nodes), Var("node", DataType::Int(32)), kDataPar, "");
    child_num = PlaceholderOpNode::make("child_num", {num_nodes}, DataType::Int(32), {node_dim},
                                        {tree_dim, node_in_tree_dim, node_dim},
                                        {tree_iv, node_in_tree_iv, node_iv},
                                        {tree_uf, node_in_tree_uf, node_uf})
                    .output(0);
  }

  std::pair<Dimension, Tensor> getChildTensorAndDim(int idx) {
    if (child_dims.count(idx)) {
      CHECK(child_tensors.count(idx));
    } else {
      child_dims[idx] =
          DimensionNode::make("child_dim" + std::to_string(idx), DimensionNode::kFunDim);
      auto batch_iv =
          IterVarNode::make(Range(0, num_trees), Var("batch", DataType::Int(32)), kDataPar, "");
      auto node_in_batch_iv = IterVarNode::make(
          Range(0, UninterpFun::MakeCallTo(node_in_tree_uf, {batch_iv->var}, {tree_dim})),
          Var("nidx", DataType::Int(32)), kDataPar, "");
      auto node_iv =
          IterVarNode::make(Range(0, num_nodes), Var("node", DataType::Int(32)), kDataPar, "");
      child_tensors[idx] =
          PlaceholderOpNode::make(
              "child_data" + std::to_string(idx), {num_nodes}, DataType::Int(32), {node_dim},
              Array<Dimension>({tree_dim, node_in_tree_dim, node_dim}),
              {batch_iv, node_in_batch_iv, node_iv}, {tree_uf, node_in_tree_uf, node_uf})
              .output(0);
    }
    return std::make_pair(child_dims.at(idx), child_tensors.at(idx));
  }
};

class LowerDSIntrinsicsStatic : public ExprMutator {
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->call_type == CallNode::PureIntrinsic) {
      if (op->name == tvm::tir::intrinsic::tvm_get_child) {
        PrimExpr node = ExprMutator::VisitExpr(op->args[0]);
        PrimExpr idx = ExprMutator::VisitExpr(op->args[1]);
        auto imm = idx.as<IntImmNode>();
        CHECK(imm);

        auto pair = sbs->getChildTensorAndDim(imm->value);
        Dimension child_dim = pair.first;
        Tensor child_tensor = pair.second;

        if (extra_dims.count(child_dim)) {
          return extra_dims.at(child_dim)->iv->var;
        } else {
          auto name = "child" + std::to_string(imm->value);
          IterVar iv = IterVarNode::make(Range(0, sbs->num_nodes), Var(name, DataType::Int(32)),
                                         kDataPar, "");
          auto n_var = Var("node", DataType::Int(32));
          auto uf = UninterpFunNode::make(
              name, Range(0, sbs->num_nodes), {sbs->node_dim}, {n_var},
              CallNode::make(DataType::Int(32), child_tensor->op->name, {n_var}, CallNode::Halide,
                             {sbs->node_dim}, child_tensor->op, child_tensor->value_index));
          extra_dims.Set(child_dim, DimInfoNode::make(child_dim, iv, uf));
          return iv->var;
        }
      } else if (op->name == tvm::tir::intrinsic::tvm_num_child) {
        PrimExpr node = ExprMutator::VisitExpr(op->args[0]);
        return CallNode::make(DataType::Int(32), sbs->child_num->op->name, {node}, CallNode::Halide,
                              {sbs->node_dim}, sbs->child_num->op, sbs->child_num->value_index);
      } else if (op->name == tvm::tir::intrinsic::tvm_is_leaf) {
        PrimExpr node = ExprMutator::VisitExpr(op->args[0]);

        return sbs->child_num[node] == 0;
      } else if (op->name == tvm::tir::intrinsic::tvm_if_then_else) {
        PrimExpr condition = this->VisitExpr(op->args[0]);
        if (ana.CanProve(condition == 1))
          return this->VisitExpr(op->args[1]);
        else if (ana.CanProve(condition == 0))
          return this->VisitExpr(op->args[2]);
        else
          return ExprMutator::VisitExpr_(op);
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) {
    auto it = vsub.find(op);
    if (it != vsub.end()) return it->second;
    return GetRef<PrimExpr>(op);
  }

 public:
  LowerDSIntrinsicsStatic(StaticBatchingState* sbs_, Var op_current_node_var_,
                          std::unordered_map<const VarNode*, PrimExpr> vsub_)
      : sbs(sbs_), op_current_node_var(op_current_node_var_), vsub(vsub_) {}

  arith::Analyzer ana;
  StaticBatchingState* sbs;
  Map<Dimension, DimInfo> extra_dims;
  Var op_current_node_var;
  std::unordered_map<const VarNode*, PrimExpr> vsub;
};

Map<Operation, Operation> LowerStatBatchInternal(Array<Operation> outputs,
                                                 StaticBatchingState* p_sbs,
                                                 std::string prefix = "") {
  CHECK_EQ(outputs.size(), 1) << "Only 1 output supported now";
  auto scan = outputs[0].as<ScanOpNode>();
  CHECK(scan) << "Only scan op output suported now";
  CHECK(scan->is_rec_op);

  StaticBatchingState& sbs = *p_sbs;

  Array<Tensor> inputs;
  for (Tensor t : scan->state_placeholder) {
    inputs.push_back(t);
  }
  for (Tensor t : scan->inputs) {
    inputs.push_back(t);
  }

  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rmap;

  Array<Dimension> int_dims;
  Array<Tensor> new_state;
  {
    for (size_t i = 0; i < num_outputs(outputs[0]); ++i) {
      Tensor old_state = scan->state_placeholder[i];

      Array<Dimension> dims;
      Array<Dimension> root_dims;
      Array<IterVar> ivs;
      Array<UninterpFun> ufs;
      Array<PrimExpr> shape;

      auto tree_iv =
          IterVarNode::make(Range(0, sbs.num_trees), Var("tree", DataType::Int(32)), kDataPar, "");
      auto node_in_tree_iv = IterVarNode::make(
          Range(0, UninterpFun::MakeCallTo(sbs.node_in_tree_uf, {tree_iv->var}, {sbs.tree_dim})),
          Var("nidx", DataType::Int(32)), kDataPar, "");
      auto node_iv =
          IterVarNode::make(Range(0, sbs.num_nodes), Var("node", DataType::Int(32)), kDataPar, "");

      dims.push_back(sbs.tree_dim);
      dims.push_back(sbs.node_in_tree_dim);
      dims.push_back(sbs.node_dim);
      root_dims.push_back(sbs.node_dim);

      ivs.push_back(tree_iv);
      ivs.push_back(node_in_tree_iv);
      ivs.push_back(node_iv);

      ufs.push_back(sbs.tree_uf);
      ufs.push_back(sbs.node_in_tree_uf);
      ufs.push_back(sbs.node_uf);

      size_t have_dim_num = int_dims.size();
      size_t need_dim_num = old_state.ndim() - 1;
      if (need_dim_num > have_dim_num) {
        for (size_t i = have_dim_num; i < need_dim_num; ++i) {
          int_dims.push_back(
              DimensionNode::make("h_dim" + std::to_string(i), DimensionNode::kRangeDim));
        }
      }

      shape.push_back(sbs.num_nodes);
      for (size_t j = 1; j < old_state.ndim(); ++j) {
        PrimExpr this_shape = old_state->op->output_shape(0)[j];
        dims.push_back(int_dims[j - 1]);
        root_dims.push_back(int_dims[j - 1]);
        auto new_iv =
            IterVarNode::make(Range(0, this_shape), Var("lv", DataType::Int(32)), kDataPar, "");
        auto uf = UninterpFunNode::from_constant("c", this_shape);
        ivs.push_back(new_iv);
        ufs.push_back(uf);
        shape.push_back(this_shape);
      }

      auto this_state =
          PlaceholderOpNode::make(prefix + old_state->op->name + ".ila", shape,
                                  old_state->op->output_dtype(0), root_dims, dims, ivs, ufs)
              .output(0);
      new_state.push_back(this_state);
      vmap[old_state] = this_state;
      rmap[this_state] = old_state;
    }
  }

  Array<Operation> ops = te::GetSubGraph(scan->update, inputs, false);

  for (auto ra_op : ops) {
    auto compute_op = ra_op.as<ComputeOpNode>();
    CHECK(compute_op) << "Only compute ops supported now";

    Array<IterVar> new_axis;
    Array<IterVar> new_reduce_axis;
    IterVar tree_iv;
    IterVar node_in_tree_iv;
    IterVar node_iv;
    std::unordered_map<const VarNode*, PrimExpr> iv_vsub;
    {
      tree_iv =
          IterVarNode::make(Range(0, sbs.num_trees), Var("tree", DataType::Int(32)), kDataPar, "");
      node_in_tree_iv = IterVarNode::make(
          Range(0, UninterpFun::MakeCallTo(sbs.node_in_tree_uf, {tree_iv->var}, {sbs.tree_dim})),
          Var("nidx", DataType::Int(32)), kDataPar, "");
      new_axis.push_back(tree_iv);
      new_axis.push_back(node_in_tree_iv);
      node_iv = compute_op->axis[0];
      for (size_t i = 1; i < compute_op->axis.size(); ++i) {
        IterVar old_iv = compute_op->axis[i];
        IterVar new_iv = IterVarNode::make(old_iv->dom, old_iv->var.copy_with_suffix(".ila"),
                                           old_iv->iter_type, old_iv->thread_tag);
        new_axis.push_back(new_iv);
        iv_vsub[old_iv->var.as<VarNode>()] = new_iv->var;
      }

      for (size_t i = 0; i < compute_op->reduce_axis.size(); ++i) {
        IterVar old_iv = compute_op->reduce_axis[i];
        IterVar new_iv = IterVarNode::make(old_iv->dom, old_iv->var.copy_with_suffix(".ila"),
                                           old_iv->iter_type, old_iv->thread_tag);
        new_reduce_axis.push_back(new_iv);
        iv_vsub[old_iv->var.as<VarNode>()] = new_iv->var;
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

      new_root_index_dimensions.push_back(sbs.node_dim);
      for (size_t i = 0; i < need_dim_num; ++i) {
        new_root_index_dimensions.push_back(int_dims[i]);
      }
    }

    Array<DimInfo> new_dim_infos;
    {
      new_dim_infos.push_back(DimInfoNode::make(sbs.tree_dim, tree_iv, {}));
      new_dim_infos.push_back(DimInfoNode::make(sbs.node_in_tree_dim, node_in_tree_iv, {}));
      new_dim_infos.push_back(DimInfoNode::make(sbs.node_dim, node_iv, sbs.node_uf));

      for (size_t i = 1; i < compute_op->axis.size(); ++i) {
        new_dim_infos.push_back(DimInfoNode::make(int_dims[i - 1], new_axis[i + 1], {}));
      }
    }

    Array<PrimExpr> new_body;
    Array<PrimExpr> new_pred;
    {
      LowerDSIntrinsicsStatic lower(&sbs, node_iv->var, iv_vsub);
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
            n->axis = new_reduce_axis;
            new_body.push_back(PrimExpr(n));
            // std::cout << "NEWVODY " << PrimExpr(n) << std::endl;
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

      if (compute_op->pred.defined()) {
        for (auto e : compute_op->pred) {
          CHECK(e.dtype().is_bool()) << e << " " << e.dtype() << " " << compute_op->name;
          PrimExpr new_expr = lower(e);
          new_pred.push_back(new_expr);
        }
      }

      for (auto it : lower.extra_dims) {
        new_dim_infos.push_back(it.second);
      }
    }

    Operation ila_op =
        ComputeOpNode::make(prefix + compute_op->name + ".ila", compute_op->tag, compute_op->attrs,
                            new_axis, new_root_index_dimensions, compute_op->output_shape_storage,
                            new_dim_infos, new_body, {new_pred});

    vmap[ra_op.output(0)] = ila_op.output(0);
    rmap[ila_op.output(0)] = ra_op.output(0);
  }

  Operation ila_scan;
  {
    auto n = make_object<ScanOpNode>();
    n->scan_dim = sbs.node_in_tree_dim;
    n->name = scan->name + ".ila";
    n->tag = scan->tag;
    n->attrs = scan->attrs;
    n->init_separate = false;
    n->init = scan->init;
    n->update = scan->update;
    n->state_placeholder = new_state;
    n->inputs = scan->inputs;

    ila_scan = Operation(n);
  }

  vmap[outputs[0].output(0)] = ila_scan.output(0);

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

  ScanOpNode* mut_new_scan = const_cast<ScanOpNode*>(vmap[ila_scan.output(0)]->op.as<ScanOpNode>());
  // mut_new_scan->scan_axis = mut_new_scan->RefreshDimVarMappings(
  // UninterpFunNode::from_constant("z", 0),
  // UninterpFunNode::from_constant("tl", sbs.max_tree_len), {sbs.tree_dim, sbs.node_in_tree_dim},
  // {UninterpFunNode::from_constant("z", 0), UninterpFunNode::from_constant("z", 0)},
  // {sbs.tree_uf, sbs.node_in_tree_uf});

  mut_new_scan->scan_axis = mut_new_scan->RefreshDimVarMappings(
      UninterpFunNode::from_constant("z", 0),
      UninterpFunNode::from_constant("tl", sbs.max_tree_len), {sbs.tree_dim},
      {UninterpFunNode::from_constant("z", 0)}, {sbs.tree_uf});

  Map<Operation, Operation> ra_ila_mapping;
  for (auto it : vmap) {
    ra_ila_mapping.Set(it.first->op, it.second->op);
  }
  ra_ila_mapping.Set(outputs[0], vmap[ila_scan.output(0)]->op);

  return ra_ila_mapping;
}

ILAOps LowerStaticBatching(Array<Operation> outputs, Var num_nodes, PrimExpr num_trees,
                           Var max_tree_len, Var max_child_num) {
  static StaticBatchingState sbs(num_nodes, num_trees, max_tree_len, max_child_num);

  Map<Tensor, Array<Tensor>> ret_mapping;
  Array<Operation> new_outputs;

  auto op_mapping = LowerStatBatchInternal(outputs, &sbs);
  new_outputs.push_back(op_mapping.at(outputs[0]));

  for (auto it : op_mapping) {
    for (size_t i = 0; i < num_outputs(it.first); ++i) {
      ret_mapping.Set(it.first.output(i), {it.second.output(i)});
    }
  }

  Array<Tensor> ds_ops;
  ds_ops.push_back(sbs.tree_lens);
  ds_ops.push_back(sbs.tree_data);
  ds_ops.push_back(sbs.child_num);

  for (auto it : sbs.child_tensors) {
    ds_ops.push_back(it.second);
  }

  // for (auto it : ret_mapping) {
  //   std::cout << "[ReT] " << it.first->op << " " << it.first->value_index << std::endl;
  //   for (auto nt : it.second) {
  //     std::cout << "[ReT]   " << nt->op << " " << nt->value_index << std::endl;
  //   }
  // }

  Map<std::string, Dimension> ds_dims;
  ds_dims.Set(sbs.tree_dim->name, sbs.tree_dim);
  ds_dims.Set(sbs.node_in_tree_dim->name, sbs.node_in_tree_dim);
  ds_dims.Set(sbs.node_dim->name, sbs.node_dim);

  return ILAOpsNode::make(ds_ops, new_outputs, ret_mapping, ds_dims);
}

TVM_REGISTER_GLOBAL("te.LowerStaticBatching")
    .set_body_typed([](Array<Operation> outputs, Var num_nodes, PrimExpr num_trees,
                       Var max_tree_len, Var max_child_num) {
      return LowerStaticBatching(outputs, num_nodes, num_trees, max_tree_len, max_child_num);
    });

}  // namespace te
}  // namespace tvm
