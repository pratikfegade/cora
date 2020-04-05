#include <tvm/te/schedule.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/te/operation.h>
#include "graph.h"

namespace tvm {
namespace te {

class SplitTensorReplacer: public StmtExprMutator {
  PrimExpr VisitExpr_(const CallNode* op) override {
    if (op->func.defined()) {
      Tensor t = Downcast<Operation>(op->func).output(op->value_index);
      if (t->op.defined() && t == this->tensor) {
	PrimExpr parent_arg = op->args[split_dim];
	PrimExpr inner_arg = indexmod(parent_arg, factor);
	PrimExpr outer_arg = indexdiv(parent_arg, factor);

	Array<PrimExpr> args;
	for (size_t i = 0; i < op->args.size(); ++i) {
	  if (i == static_cast<size_t>(split_dim)) {
	    args.push_back(outer_arg);
	    args.push_back(inner_arg);
	  }
	  else {
	    args.push_back(op->args[i]);
	  }
	}

	return CallNode::make(op->dtype, op->name, args, op->call_type,
			      op->argument_dimensions, op->func, op->value_index);
      }
    }

    return StmtExprMutator::VisitExpr_(op);
  }

  ComputeOpNode* reader;
  const Tensor& tensor;
  int split_dim;
  int factor;

public:
  SplitTensorReplacer(ComputeOpNode* reader_, const Tensor& tensor_, int split_dim_, int factor_) :
    reader(reader_), tensor(tensor_), split_dim(split_dim_), factor(factor_) {}

  void replace() {
    // TODO(ppf): Handle reductions here
    Array<PrimExpr> body;
    for (auto e: reader->body) {
      std::cout << "EXPR " << e << std::endl;
      std::cout << "REPL " << this->VisitExpr(e) << std::endl;
      body.push_back(this->VisitExpr(e));
    }

    reader->body = body;
  }
};


Tensor Schedule::split_tensor_dimension(const Tensor& tensor,
					const size_t dim_idx,
					const int factor) {
  auto compute_op = const_cast<ComputeOpNode*>(tensor->op.as<ComputeOpNode>());
  CHECK(compute_op) <<
    "Layout changes allowed only for ComputeOp";
  CHECK(dim_idx < compute_op->output_shape_storage.size());
  Dimension dimension = compute_op->self_index_dimensions[dim_idx];
  PrimExpr dim_shape = compute_op->output_shape_storage[dim_idx];
  PrimExpr inner_shape = factor;
  PrimExpr outer_shape = indexdiv(dim_shape, factor);

  std::cout << inner_shape << " " << outer_shape << std::endl;

  Dimension inner_dimension = DimensionNode::make(dimension->name + ".inner", dimension->type);
  Dimension outer_dimension = DimensionNode::make(dimension->name + ".outer", dimension->type);

  ArrayNode* self_index_dimensions = compute_op->self_index_dimensions.CopyOnWrite();
  self_index_dimensions->data.erase(self_index_dimensions->data.begin() + dim_idx);
  self_index_dimensions->data.insert(self_index_dimensions->data.begin() + dim_idx, outer_dimension);
  self_index_dimensions->data.insert(self_index_dimensions->data.begin() + dim_idx, inner_dimension);

  ArrayNode* output_shape_storage = compute_op->output_shape_storage.CopyOnWrite();
  output_shape_storage->data.erase(output_shape_storage->data.begin() + dim_idx);
  output_shape_storage->data.insert(output_shape_storage->data.begin() + dim_idx, outer_shape);
  output_shape_storage->data.insert(output_shape_storage->data.begin() + dim_idx, inner_shape);

  Var parent_var = compute_op->GetVarFromDim(dimension);

  IterVar inner_var = IterVarNode::make(Range::make_by_min_extent(0, inner_shape),
					parent_var.copy_with_suffix(".inner"), IterVarType::kDataPar, "");
  IterVar outer_var = IterVarNode::make(Range::make_by_min_extent(0, outer_shape),
					parent_var.copy_with_suffix(".outer"), IterVarType::kDataPar, "");

  compute_op->index_variables.push_back(inner_var);
  compute_op->index_variables.push_back(outer_var);

  Var inner_param = Var("inner_param", DataType::Int(32));
  UninterpFun inner_uf = UninterpFunNode::make("inner", Range::make_by_min_extent(0, factor), { dimension },
					       { inner_param }, indexmod(inner_param, factor));
  Var outer_param = Var("outer_param", DataType::Int(32));
  UninterpFun outer_uf = UninterpFunNode::make("inner", Range::make_by_min_extent(0, outer_shape), { dimension },
					       { outer_param }, indexdiv(outer_param, factor));

  compute_op->index_expressions.push_back(inner_uf);
  compute_op->index_expressions.push_back(outer_uf);

  compute_op->index_dimensions.push_back(inner_dimension);
  compute_op->index_dimensions.push_back(outer_dimension);


  Schedule sch = (*this);
  Array<Operation> roots;
  for (Operation op : sch->outputs) {
    roots.push_back(sch->stage_map[op]->op);
  }
  std::vector<Operation> readers = CreateFeedGraph(CreateReadGraph(roots))[tensor];
  for (auto reader: readers) {
    auto reader_op = const_cast<ComputeOpNode*>(reader.as<ComputeOpNode>());
    CHECK(reader_op) << "Only support ComputeOp readers for now";

    SplitTensorReplacer replacer(reader_op, tensor, dim_idx, factor);
    replacer.replace();
  }

  return tensor;
}
}
}
