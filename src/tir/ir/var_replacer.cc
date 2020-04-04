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
  auto new_identity = UpdateArray(combiner->identity_element, [this] (const PrimExpr& e) {
      return this->VisitExpr(e);
    });
  auto new_result = UpdateArray(combiner->result, [this] (const PrimExpr& e) {
      return this->VisitExpr(e);
    });

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
    return ReduceNode::make(new_combiner,
				 new_reduce->source,
				 new_reduce->axis,
				 new_reduce->condition,
				 new_reduce->value_index);
  }
}
}
}
