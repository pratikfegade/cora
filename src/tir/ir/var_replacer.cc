#include "var_replacer.h"

#include "../pass/ir_util.h"

namespace tvm {
namespace tir {

PrimExpr VarReplacer::VisitExpr_(const VarNode* op) {
  auto it = vsub_.find(op);
  if (it != vsub_.end()) return it->second;
  return GetRef<PrimExpr>(op);
}

CommReducer VarReplacer::MutateCommReducer(CommReducer combiner) {
  // Replace free variables in combiner
  auto new_identity = UpdateArray(combiner->identity_element,
                                  [this](const PrimExpr& e) { return this->VisitExpr(e); });
  auto new_result =
      UpdateArray(combiner->result, [this](const PrimExpr& e) { return this->VisitExpr(e); });

  if (combiner->identity_element.same_as(new_identity) &&
      combiner->identity_element.same_as(new_result)) {
    return combiner;
  } else {
    return CommReducerNode::make(combiner->lhs, combiner->rhs, new_result, new_identity);
  }
}

PrimExpr VarReplacer::VisitExpr_(const ReduceNode* op) {
  PrimExpr new_e = StmtExprMutator::VisitExpr_(op);
  const ReduceNode* new_reduce = new_e.as<ReduceNode>();
  CommReducer new_combiner = MutateCommReducer(op->combiner);
  if (op->combiner.same_as(new_combiner)) {
    return new_e;
  } else {
    return ReduceNode::make(new_combiner, new_reduce->source, new_reduce->axis,
                            new_reduce->condition, new_reduce->value_index);
  }
}

PrimExpr VarReplacer::VisitExpr_(const LoadNode* op) {
  Var buffer_var =
      replace_buffers_ ? Downcast<Var>(this->VisitExpr(op->buffer_var)) : op->buffer_var;
  PrimExpr index = this->VisitExpr(op->index);
  PrimExpr predicate = this->VisitExpr(op->predicate);
  if (buffer_var.same_as(op->buffer_var) && index.same_as(op->index) &&
      predicate.same_as(op->predicate)) {
    return GetRef<PrimExpr>(op);
  } else {
    return LoadNode::make(op->dtype, buffer_var, index, predicate, op->sync_type);
  }
}

Stmt VarReplacer::VisitStmt_(const StoreNode* op) {
  Var buffer_var =
      replace_buffers_ ? Downcast<Var>(this->VisitExpr(op->buffer_var)) : op->buffer_var;
  PrimExpr value = this->VisitExpr(op->value);
  PrimExpr index = this->VisitExpr(op->index);
  PrimExpr predicate = this->VisitExpr(op->predicate);
  if (buffer_var.same_as(op->buffer_var) && value.same_as(op->value) && index.same_as(op->index) &&
      predicate.same_as(op->predicate)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->buffer_var = std::move(buffer_var);
    n->value = std::move(value);
    n->index = std::move(index);
    n->predicate = std::move(predicate);
    n->sync_type = std::move(op->sync_type);
    return Stmt(n);
  }
}

Stmt VarReplacer::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == attr::aux_data_structure) {
    std::cout << "[VR]  Replacing NonNeg Attr " << op->node << std::endl;
    ObjectRef node = op->node;
    ObjectRef new_node = node;
    if (auto ufn = op->node.as<UninterpFunNode>()) {
      std::cout << "[VR]   Uf" << std::endl;
      new_node = UninterpFunNode::make(ufn->fname, ufn->range, ufn->dimensions, ufn->parameters,
                                       this->VisitExpr(ufn->body));
    } else if (op->node.as<VarNode>()) {
      new_node = this->VisitExpr(Downcast<Var>(node));
    }
    if (!node.same_as(new_node)) {
      return AttrStmtNode::make(new_node, op->attr_key, op->value, this->VisitStmt(op->body),
                                op->hfuse_group_id);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  return StmtExprMutator::VisitStmt_(op);
}

void VarFinder::VisitExpr_(const VarNode* op) {
  auto it = vset_.find(op);
  if (it != vset_.end()) this->found = true;
}

void VarCollector::VisitExpr_(const VarNode* op) { collected.insert(op); }

void VarCollector::VisitExpr_(const LoadNode* op) {
  if (collect_buffers) {
    this->VisitExpr(op->buffer_var);
  }
  this->VisitExpr(op->index);
  this->VisitExpr(op->predicate);
}

void VarCollector::VisitStmt_(const StoreNode* op) {
  if (collect_buffers) {
    this->VisitExpr(op->buffer_var);
  }
  this->VisitExpr(op->value);
  this->VisitExpr(op->index);
  this->VisitExpr(op->predicate);
}

void TensorCallCollector::VisitExpr_(const CallNode* op) {
  if (op->call_type == CallNode::Halide && op->func.defined()) {
    if (auto operation = op->func.as<te::OperationNode>()) {
      this->collected.insert(operation);
    }
  }
}

}  // namespace tir
}  // namespace tvm
