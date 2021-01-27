#include "var_replacer.h"

#include "../pass/ir_util.h"

namespace tvm {
namespace tir {

PrimExpr VarReplacer::GetReplacementExpr(Var op) {
  auto it = vsub_.find(op.as<VarNode>());
  if (it != vsub_.end()) return it->second;
  return op;
}

Var VarReplacer::GetReplacementVar(Var op) {
  auto expr = this->GetReplacementExpr(op);
  CHECK(expr.as<VarNode>());
  return Downcast<Var>(expr);
}

PrimExpr VarReplacer::VisitExpr_(const VarNode* op) { return GetReplacementExpr(GetRef<Var>(op)); }

PrimExpr VarReplacer::VisitExpr_(const LoadNode* op) {
  return LoadNode::make(op->dtype, GetReplacementVar(op->buffer_var), this->VisitExpr(op->index),
                        this->VisitExpr(op->predicate), op->sync_type);
}

Stmt VarReplacer::VisitStmt_(const StoreNode* op) {
  return StoreNode::make(this->GetReplacementVar(op->buffer_var), this->VisitExpr(op->value),
                         this->VisitExpr(op->index), this->VisitExpr(op->predicate), op->sync_type);
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

void VarFinder::VisitExpr_(const VarNode* op) {
  auto it = vset_.find(op);
  if (it != vset_.end()) this->found = true;
}

bool VarFinder::ContainsVariable(PrimExpr expr, Var var) {
  std::unordered_set<const VarNode*> vset;
  vset.insert(var.as<VarNode>());
  VarFinder finder(vset);
  return finder.find(expr);
}

void VarCollector::VisitExpr_(const VarNode* op) { collected.insert(op); }

void TensorCallCollector::VisitExpr_(const CallNode* op) {
  if (op->call_type == CallNode::Halide && op->func.defined()) {
    if (auto operation = op->func.as<te::OperationNode>()) {
      this->collected.insert(operation);
    }
  }
}

}  // namespace tir
}  // namespace tvm
