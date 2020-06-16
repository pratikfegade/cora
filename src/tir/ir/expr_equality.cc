#include <tvm/tir/expr_equality.h>

namespace tvm {
namespace tir {

bool ExprEquality::VisitExpr_(const VarNode* op1, const VarNode* op2) { return op1 == op2; }

bool ExprEquality::VisitExpr_(const SizeVarNode* op1, const SizeVarNode* op2) {
  return this->VisitExpr_(static_cast<const VarNode*>(op1), static_cast<const VarNode*>(op2));
}

bool ExprEquality::VisitExpr_(const LoadNode* op1, const LoadNode* op2) {
  return this->VisitExpr(op1->index, op2->index) && this->VisitExpr(op1->predicate, op2->predicate);
}

bool ExprEquality::VisitExpr_(const LetNode* op1, const LetNode* op2) {
  return this->VisitExpr(op1->value, op2->value) && this->VisitExpr(op1->body, op2->body);
}

bool ExprEquality::VisitExpr_(const CallNode* op1, const CallNode* op2) {
  return VisitArray(op1->args, op2->args, [this](const PrimExpr& e1, const PrimExpr& e2) {
    return this->VisitExpr(e1, e2);
  });
}

#define DEFINE_BINOP_VISIT_(OP)                                                \
  bool ExprEquality::VisitExpr_(const OP* op1, const OP* op2) {                \
    return this->VisitExpr(op1->a, op2->a) && this->VisitExpr(op1->b, op2->b); \
  }

DEFINE_BINOP_VISIT_(AddNode);
DEFINE_BINOP_VISIT_(SubNode);
DEFINE_BINOP_VISIT_(MulNode);
DEFINE_BINOP_VISIT_(DivNode);
DEFINE_BINOP_VISIT_(ModNode);
DEFINE_BINOP_VISIT_(FloorDivNode);
DEFINE_BINOP_VISIT_(FloorModNode);
DEFINE_BINOP_VISIT_(MinNode);
DEFINE_BINOP_VISIT_(MaxNode);
DEFINE_BINOP_VISIT_(EQNode);
DEFINE_BINOP_VISIT_(NENode);
DEFINE_BINOP_VISIT_(LTNode);
DEFINE_BINOP_VISIT_(LENode);
DEFINE_BINOP_VISIT_(GTNode);
DEFINE_BINOP_VISIT_(GENode);
DEFINE_BINOP_VISIT_(AndNode);
DEFINE_BINOP_VISIT_(OrNode);

bool ExprEquality::VisitExpr_(const IntImmNode* op1, const IntImmNode* op2) {
  return op1->value == op2->value;
}
bool ExprEquality::VisitExpr_(const FloatImmNode* op1, const FloatImmNode* op2) {
  return op1->value == op2->value;
}
bool ExprEquality::VisitExpr_(const StringImmNode* op1, const StringImmNode* op2) {
  return op1->value == op2->value;
}

bool ExprEquality::VisitExpr_(const ReduceNode* op1, const ReduceNode* op2) {
  return VisitArray(op1->axis, op2->axis,
                    [this](const IterVar& r1, const IterVar& r2) {
                      return this->VisitExpr(r1->var, r2->var);
                    }) &&
         VisitArray(
             op1->source, op2->source,
             [this](const PrimExpr& e1, const PrimExpr& e2) { return this->VisitExpr(e1, e2); }) &&
         this->VisitExpr(op1->condition, op2->condition);
}

bool ExprEquality::VisitExpr_(const CastNode* op1, const CastNode* op2) {
  return this->VisitExpr(op1->value, op2->value);
}

bool ExprEquality::VisitExpr_(const NotNode* op1, const NotNode* op2) {
  return this->VisitExpr(op1->a, op2->a);
}

bool ExprEquality::VisitExpr_(const SelectNode* op1, const SelectNode* op2) {
  return this->VisitExpr(op1->condition, op2->condition) &&
         this->VisitExpr(op1->true_value, op2->true_value) &&
         this->VisitExpr(op1->false_value, op2->false_value);
}

bool ExprEquality::VisitExpr_(const RampNode* op1, const RampNode* op2) {
  return this->VisitExpr(op1->base, op2->base) && this->VisitExpr(op1->stride, op2->stride);
}

bool ExprEquality::VisitExpr_(const ShuffleNode* op1, const ShuffleNode* op2) {
  return VisitArray(
             op1->indices, op2->indices,
             [this](const PrimExpr& e1, const PrimExpr& e2) { return this->VisitExpr(e1, e2); }) &&
         VisitArray(op1->vectors, op2->vectors, [this](const PrimExpr& e1, const PrimExpr& e2) {
           return this->VisitExpr(e1, e2);
         });
}

bool ExprEquality::VisitExpr_(const BroadcastNode* op1, const BroadcastNode* op2) {
  return this->VisitExpr(op1->value, op2->value);
}

#define CALL_VISIT_EXPR_(typ, e1, e2)                  \
  {                                                    \
    const typ* op1 = e1.as<typ>();                     \
    const typ* op2 = e2.as<typ>();                     \
    if (op1 && op2) return this->VisitExpr_(op1, op2); \
  }

bool ExprEquality::VisitExpr(PrimExpr e1, PrimExpr e2) {
  CALL_VISIT_EXPR_(AddNode, e1, e2);
  CALL_VISIT_EXPR_(SubNode, e1, e2);
  CALL_VISIT_EXPR_(MulNode, e1, e2);
  CALL_VISIT_EXPR_(DivNode, e1, e2);
  CALL_VISIT_EXPR_(ModNode, e1, e2);
  CALL_VISIT_EXPR_(FloorDivNode, e1, e2);
  CALL_VISIT_EXPR_(FloorModNode, e1, e2);
  CALL_VISIT_EXPR_(MinNode, e1, e2);
  CALL_VISIT_EXPR_(MaxNode, e1, e2);
  CALL_VISIT_EXPR_(EQNode, e1, e2);
  CALL_VISIT_EXPR_(NENode, e1, e2);
  CALL_VISIT_EXPR_(LTNode, e1, e2);
  CALL_VISIT_EXPR_(LENode, e1, e2);
  CALL_VISIT_EXPR_(GTNode, e1, e2);
  CALL_VISIT_EXPR_(GENode, e1, e2);
  CALL_VISIT_EXPR_(AndNode, e1, e2);
  CALL_VISIT_EXPR_(OrNode, e1, e2);
  CALL_VISIT_EXPR_(VarNode, e1, e2);
  CALL_VISIT_EXPR_(SizeVarNode, e1, e2);
  CALL_VISIT_EXPR_(LoadNode, e1, e2);
  CALL_VISIT_EXPR_(LetNode, e1, e2);
  CALL_VISIT_EXPR_(CallNode, e1, e2);
  CALL_VISIT_EXPR_(IntImmNode, e1, e2);
  CALL_VISIT_EXPR_(FloatImmNode, e1, e2);
  CALL_VISIT_EXPR_(StringImmNode, e1, e2);
  CALL_VISIT_EXPR_(ReduceNode, e1, e2);
  CALL_VISIT_EXPR_(CastNode, e1, e2);
  CALL_VISIT_EXPR_(NotNode, e1, e2);
  CALL_VISIT_EXPR_(SelectNode, e1, e2);
  CALL_VISIT_EXPR_(RampNode, e1, e2);
  CALL_VISIT_EXPR_(ShuffleNode, e1, e2);
  CALL_VISIT_EXPR_(BroadcastNode, e1, e2);
  return false;
}
}  // namespace tir
}  // namespace tvm
