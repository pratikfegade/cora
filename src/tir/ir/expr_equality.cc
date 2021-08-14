#include <tvm/tir/expr_equality.h>

namespace tvm {
namespace tir {

bool ExprEquality::VisitExpr_(const VarNode* op1, const VarNode* op2) const { return op1 == op2; }

bool ExprEquality::VisitExpr_(const SizeVarNode* op1, const SizeVarNode* op2) const {
  return this->VisitExpr_(static_cast<const VarNode*>(op1), static_cast<const VarNode*>(op2));
}

bool ExprEquality::VisitExpr_(const LoadNode* op1, const LoadNode* op2) const {
  return this->VisitExpr(op1->index, op2->index) && this->VisitExpr(op1->predicate, op2->predicate);
}

bool ExprEquality::VisitExpr_(const LetNode* op1, const LetNode* op2) const {
  return this->VisitExpr(op1->value, op2->value) && this->VisitExpr(op1->body, op2->body);
}

bool ExprEquality::VisitExpr_(const CallNode* op1, const CallNode* op2) const {
  return VisitArray(op1->args, op2->args, [this](const PrimExpr& e1, const PrimExpr& e2) {
    return this->VisitExpr(e1, e2);
  });
}

#define DEFINE_BINOP_VISIT_EE_(OP)                                             \
  bool ExprEquality::VisitExpr_(const OP* op1, const OP* op2) const {          \
    return this->VisitExpr(op1->a, op2->a) && this->VisitExpr(op1->b, op2->b); \
  }

DEFINE_BINOP_VISIT_EE_(AddNode);
DEFINE_BINOP_VISIT_EE_(SubNode);
DEFINE_BINOP_VISIT_EE_(MulNode);
DEFINE_BINOP_VISIT_EE_(DivNode);
DEFINE_BINOP_VISIT_EE_(ModNode);
DEFINE_BINOP_VISIT_EE_(FloorDivNode);
DEFINE_BINOP_VISIT_EE_(FloorModNode);
DEFINE_BINOP_VISIT_EE_(MinNode);
DEFINE_BINOP_VISIT_EE_(MaxNode);
DEFINE_BINOP_VISIT_EE_(EQNode);
DEFINE_BINOP_VISIT_EE_(NENode);
DEFINE_BINOP_VISIT_EE_(LTNode);
DEFINE_BINOP_VISIT_EE_(LENode);
DEFINE_BINOP_VISIT_EE_(GTNode);
DEFINE_BINOP_VISIT_EE_(GENode);
DEFINE_BINOP_VISIT_EE_(AndNode);
DEFINE_BINOP_VISIT_EE_(OrNode);

bool ExprEquality::VisitExpr_(const IntImmNode* op1, const IntImmNode* op2) const {
  return op1->value == op2->value;
}
bool ExprEquality::VisitExpr_(const FloatImmNode* op1, const FloatImmNode* op2) const {
  return op1->value == op2->value;
}
bool ExprEquality::VisitExpr_(const StringImmNode* op1, const StringImmNode* op2) const {
  return op1->value == op2->value;
}

bool ExprEquality::VisitExpr_(const ReduceNode* op1, const ReduceNode* op2) const {
  return VisitArray(op1->axis, op2->axis,
                    [this](const IterVar& r1, const IterVar& r2) {
                      return this->VisitExpr(r1->var, r2->var);
                    }) &&
         VisitArray(
             op1->source, op2->source,
             [this](const PrimExpr& e1, const PrimExpr& e2) { return this->VisitExpr(e1, e2); }) &&
         this->VisitExpr(op1->condition, op2->condition);
}

bool ExprEquality::VisitExpr_(const CastNode* op1, const CastNode* op2) const {
  return this->VisitExpr(op1->value, op2->value);
}

bool ExprEquality::VisitExpr_(const NotNode* op1, const NotNode* op2) const {
  return this->VisitExpr(op1->a, op2->a);
}

bool ExprEquality::VisitExpr_(const SelectNode* op1, const SelectNode* op2) const {
  return this->VisitExpr(op1->condition, op2->condition) &&
         this->VisitExpr(op1->true_value, op2->true_value) &&
         this->VisitExpr(op1->false_value, op2->false_value);
}

bool ExprEquality::VisitExpr_(const FuseSelectNode* op1, const FuseSelectNode* op2) const {
  return this->VisitExpr(op1->condition, op2->condition) &&
         this->VisitExpr(op1->true_value, op2->true_value) &&
         this->VisitExpr(op1->false_value, op2->false_value);
}

bool ExprEquality::VisitExpr_(const RampNode* op1, const RampNode* op2) const {
  return this->VisitExpr(op1->base, op2->base) && this->VisitExpr(op1->stride, op2->stride);
}

bool ExprEquality::VisitExpr_(const ShuffleNode* op1, const ShuffleNode* op2) const {
  return VisitArray(
             op1->indices, op2->indices,
             [this](const PrimExpr& e1, const PrimExpr& e2) { return this->VisitExpr(e1, e2); }) &&
         VisitArray(op1->vectors, op2->vectors, [this](const PrimExpr& e1, const PrimExpr& e2) {
           return this->VisitExpr(e1, e2);
         });
}

bool ExprEquality::VisitExpr_(const BroadcastNode* op1, const BroadcastNode* op2) const {
  return this->VisitExpr(op1->value, op2->value);
}

#define CALL_VISIT_EXPR_EE_(typ, e1, e2)               \
  {                                                    \
    const typ* op1 = e1.as<typ>();                     \
    const typ* op2 = e2.as<typ>();                     \
    if (op1 && op2) return this->VisitExpr_(op1, op2); \
  }

bool ExprEquality::VisitExpr(PrimExpr e1, PrimExpr e2) const {
  CALL_VISIT_EXPR_EE_(AddNode, e1, e2);
  CALL_VISIT_EXPR_EE_(SubNode, e1, e2);
  CALL_VISIT_EXPR_EE_(MulNode, e1, e2);
  CALL_VISIT_EXPR_EE_(DivNode, e1, e2);
  CALL_VISIT_EXPR_EE_(ModNode, e1, e2);
  CALL_VISIT_EXPR_EE_(FloorDivNode, e1, e2);
  CALL_VISIT_EXPR_EE_(FloorModNode, e1, e2);
  CALL_VISIT_EXPR_EE_(MinNode, e1, e2);
  CALL_VISIT_EXPR_EE_(MaxNode, e1, e2);
  CALL_VISIT_EXPR_EE_(EQNode, e1, e2);
  CALL_VISIT_EXPR_EE_(NENode, e1, e2);
  CALL_VISIT_EXPR_EE_(LTNode, e1, e2);
  CALL_VISIT_EXPR_EE_(LENode, e1, e2);
  CALL_VISIT_EXPR_EE_(GTNode, e1, e2);
  CALL_VISIT_EXPR_EE_(GENode, e1, e2);
  CALL_VISIT_EXPR_EE_(AndNode, e1, e2);
  CALL_VISIT_EXPR_EE_(OrNode, e1, e2);
  CALL_VISIT_EXPR_EE_(VarNode, e1, e2);
  CALL_VISIT_EXPR_EE_(SizeVarNode, e1, e2);
  CALL_VISIT_EXPR_EE_(LoadNode, e1, e2);
  CALL_VISIT_EXPR_EE_(LetNode, e1, e2);
  CALL_VISIT_EXPR_EE_(CallNode, e1, e2);
  CALL_VISIT_EXPR_EE_(IntImmNode, e1, e2);
  CALL_VISIT_EXPR_EE_(FloatImmNode, e1, e2);
  CALL_VISIT_EXPR_EE_(StringImmNode, e1, e2);
  CALL_VISIT_EXPR_EE_(ReduceNode, e1, e2);
  CALL_VISIT_EXPR_EE_(CastNode, e1, e2);
  CALL_VISIT_EXPR_EE_(NotNode, e1, e2);
  CALL_VISIT_EXPR_EE_(SelectNode, e1, e2);
  CALL_VISIT_EXPR_EE_(FuseSelectNode, e1, e2);
  CALL_VISIT_EXPR_EE_(RampNode, e1, e2);
  CALL_VISIT_EXPR_EE_(ShuffleNode, e1, e2);
  CALL_VISIT_EXPR_EE_(BroadcastNode, e1, e2);
  return false;
}

bool ExprEquality::VisitExprConst(const PrimExpr e1, const PrimExpr e2) const {
  CALL_VISIT_EXPR_EE_(AddNode, e1, e2);
  CALL_VISIT_EXPR_EE_(SubNode, e1, e2);
  CALL_VISIT_EXPR_EE_(MulNode, e1, e2);
  CALL_VISIT_EXPR_EE_(DivNode, e1, e2);
  CALL_VISIT_EXPR_EE_(ModNode, e1, e2);
  CALL_VISIT_EXPR_EE_(FloorDivNode, e1, e2);
  CALL_VISIT_EXPR_EE_(FloorModNode, e1, e2);
  CALL_VISIT_EXPR_EE_(MinNode, e1, e2);
  CALL_VISIT_EXPR_EE_(MaxNode, e1, e2);
  CALL_VISIT_EXPR_EE_(EQNode, e1, e2);
  CALL_VISIT_EXPR_EE_(NENode, e1, e2);
  CALL_VISIT_EXPR_EE_(LTNode, e1, e2);
  CALL_VISIT_EXPR_EE_(LENode, e1, e2);
  CALL_VISIT_EXPR_EE_(GTNode, e1, e2);
  CALL_VISIT_EXPR_EE_(GENode, e1, e2);
  CALL_VISIT_EXPR_EE_(AndNode, e1, e2);
  CALL_VISIT_EXPR_EE_(OrNode, e1, e2);
  CALL_VISIT_EXPR_EE_(VarNode, e1, e2);
  CALL_VISIT_EXPR_EE_(SizeVarNode, e1, e2);
  CALL_VISIT_EXPR_EE_(LoadNode, e1, e2);
  CALL_VISIT_EXPR_EE_(LetNode, e1, e2);
  CALL_VISIT_EXPR_EE_(CallNode, e1, e2);
  CALL_VISIT_EXPR_EE_(IntImmNode, e1, e2);
  CALL_VISIT_EXPR_EE_(FloatImmNode, e1, e2);
  CALL_VISIT_EXPR_EE_(StringImmNode, e1, e2);
  CALL_VISIT_EXPR_EE_(ReduceNode, e1, e2);
  CALL_VISIT_EXPR_EE_(CastNode, e1, e2);
  CALL_VISIT_EXPR_EE_(NotNode, e1, e2);
  CALL_VISIT_EXPR_EE_(SelectNode, e1, e2);
  CALL_VISIT_EXPR_EE_(FuseSelectNode, e1, e2);
  CALL_VISIT_EXPR_EE_(RampNode, e1, e2);
  CALL_VISIT_EXPR_EE_(ShuffleNode, e1, e2);
  CALL_VISIT_EXPR_EE_(BroadcastNode, e1, e2);
  return false;
}

bool DeeperExprEquality::VisitExpr_(const LoadNode* op1, const LoadNode* op2) const {
  return this->VisitExpr(op1->buffer_var, op2->buffer_var) &&
         this->VisitExpr(op1->index, op2->index) && this->VisitExpr(op1->predicate, op2->predicate);
}

bool DeeperExprEquality::VisitExpr_(const CallNode* op1, const CallNode* op2) const {
  return op1->func == op2->func &&
         VisitArray(op1->args, op2->args, [this](const PrimExpr& e1, const PrimExpr& e2) {
           return this->VisitExpr(e1, e2);
         });
}

/////////////////////////////////////////////////////////////////////////////////

size_t ExprHash::VisitExpr_(const VarNode* op) const { return std::hash<const VarNode*>()(op); }

size_t ExprHash::VisitExpr_(const SizeVarNode* op) const {
  return this->VisitExpr_(static_cast<const VarNode*>(op));
}

size_t ExprHash::VisitExpr_(const LoadNode* op) const {
  return this->VisitExpr(op->index) ^ this->VisitExpr(op->predicate);
}

size_t ExprHash::VisitExpr_(const LetNode* op) const {
  return this->VisitExpr(op->value) ^ this->VisitExpr(op->body);
}

size_t ExprHash::VisitExpr_(const CallNode* op) const {
  return VisitArray(op->args, [this](const PrimExpr& e) { return this->VisitExpr(e); });
}

#define DEFINE_BINOP_VISIT_EH_(OP)                          \
  size_t ExprHash::VisitExpr_(const OP* op) const {         \
    return this->VisitExpr(op->a) ^ this->VisitExpr(op->b); \
  }

DEFINE_BINOP_VISIT_EH_(AddNode);
DEFINE_BINOP_VISIT_EH_(SubNode);
DEFINE_BINOP_VISIT_EH_(MulNode);
DEFINE_BINOP_VISIT_EH_(DivNode);
DEFINE_BINOP_VISIT_EH_(ModNode);
DEFINE_BINOP_VISIT_EH_(FloorDivNode);
DEFINE_BINOP_VISIT_EH_(FloorModNode);
DEFINE_BINOP_VISIT_EH_(MinNode);
DEFINE_BINOP_VISIT_EH_(MaxNode);
DEFINE_BINOP_VISIT_EH_(EQNode);
DEFINE_BINOP_VISIT_EH_(NENode);
DEFINE_BINOP_VISIT_EH_(LTNode);
DEFINE_BINOP_VISIT_EH_(LENode);
DEFINE_BINOP_VISIT_EH_(GTNode);
DEFINE_BINOP_VISIT_EH_(GENode);
DEFINE_BINOP_VISIT_EH_(AndNode);
DEFINE_BINOP_VISIT_EH_(OrNode);

size_t ExprHash::VisitExpr_(const IntImmNode* op) const { return std::hash<int64_t>()(op->value); }
size_t ExprHash::VisitExpr_(const FloatImmNode* op) const { return std::hash<double>()(op->value); }
size_t ExprHash::VisitExpr_(const StringImmNode* op) const {
  return std::hash<std::string>()(op->value);
}

size_t ExprHash::VisitExpr_(const ReduceNode* op) const {
  return VisitArray(op->axis, [this](const IterVar& r) { return this->VisitExpr(r->var); }) ^
         VisitArray(op->source, [this](const PrimExpr& e) { return this->VisitExpr(e); }) ^
         this->VisitExpr(op->condition);
}

size_t ExprHash::VisitExpr_(const CastNode* op) const { return this->VisitExpr(op->value); }

size_t ExprHash::VisitExpr_(const NotNode* op) const { return this->VisitExpr(op->a); }

size_t ExprHash::VisitExpr_(const SelectNode* op) const {
  return this->VisitExpr(op->condition) ^ this->VisitExpr(op->true_value) ^
         this->VisitExpr(op->false_value);
}

size_t ExprHash::VisitExpr_(const FuseSelectNode* op) const {
  return this->VisitExpr(op->condition) ^ this->VisitExpr(op->true_value) ^
         this->VisitExpr(op->false_value);
}

size_t ExprHash::VisitExpr_(const RampNode* op) const {
  return this->VisitExpr(op->base) ^ this->VisitExpr(op->stride);
}

size_t ExprHash::VisitExpr_(const ShuffleNode* op) const {
  return VisitArray(op->indices, [this](const PrimExpr& e) { return this->VisitExpr(e); }) ^
         VisitArray(op->vectors, [this](const PrimExpr& e) { return this->VisitExpr(e); });
}

size_t ExprHash::VisitExpr_(const BroadcastNode* op) const { return this->VisitExpr(op->value); }

#define CALL_VISIT_EXPR_EH_(typ, e)      \
  {                                      \
    const typ* op = e.as<typ>();         \
    if (op) return this->VisitExpr_(op); \
  }

size_t ExprHash::VisitExpr(PrimExpr e) const {
  CALL_VISIT_EXPR_EH_(AddNode, e);
  CALL_VISIT_EXPR_EH_(SubNode, e);
  CALL_VISIT_EXPR_EH_(MulNode, e);
  CALL_VISIT_EXPR_EH_(DivNode, e);
  CALL_VISIT_EXPR_EH_(ModNode, e);
  CALL_VISIT_EXPR_EH_(FloorDivNode, e);
  CALL_VISIT_EXPR_EH_(FloorModNode, e);
  CALL_VISIT_EXPR_EH_(MinNode, e);
  CALL_VISIT_EXPR_EH_(MaxNode, e);
  CALL_VISIT_EXPR_EH_(EQNode, e);
  CALL_VISIT_EXPR_EH_(NENode, e);
  CALL_VISIT_EXPR_EH_(LTNode, e);
  CALL_VISIT_EXPR_EH_(LENode, e);
  CALL_VISIT_EXPR_EH_(GTNode, e);
  CALL_VISIT_EXPR_EH_(GENode, e);
  CALL_VISIT_EXPR_EH_(AndNode, e);
  CALL_VISIT_EXPR_EH_(OrNode, e);
  CALL_VISIT_EXPR_EH_(VarNode, e);
  CALL_VISIT_EXPR_EH_(SizeVarNode, e);
  CALL_VISIT_EXPR_EH_(LoadNode, e);
  CALL_VISIT_EXPR_EH_(LetNode, e);
  CALL_VISIT_EXPR_EH_(CallNode, e);
  CALL_VISIT_EXPR_EH_(IntImmNode, e);
  CALL_VISIT_EXPR_EH_(FloatImmNode, e);
  CALL_VISIT_EXPR_EH_(StringImmNode, e);
  CALL_VISIT_EXPR_EH_(ReduceNode, e);
  CALL_VISIT_EXPR_EH_(CastNode, e);
  CALL_VISIT_EXPR_EH_(NotNode, e);
  CALL_VISIT_EXPR_EH_(SelectNode, e);
  CALL_VISIT_EXPR_EH_(FuseSelectNode, e);
  CALL_VISIT_EXPR_EH_(RampNode, e);
  CALL_VISIT_EXPR_EH_(ShuffleNode, e);
  CALL_VISIT_EXPR_EH_(BroadcastNode, e);
  return 0;
}

size_t ExprHash::VisitExprConst(const PrimExpr e) const {
  CALL_VISIT_EXPR_EH_(AddNode, e);
  CALL_VISIT_EXPR_EH_(SubNode, e);
  CALL_VISIT_EXPR_EH_(MulNode, e);
  CALL_VISIT_EXPR_EH_(DivNode, e);
  CALL_VISIT_EXPR_EH_(ModNode, e);
  CALL_VISIT_EXPR_EH_(FloorDivNode, e);
  CALL_VISIT_EXPR_EH_(FloorModNode, e);
  CALL_VISIT_EXPR_EH_(MinNode, e);
  CALL_VISIT_EXPR_EH_(MaxNode, e);
  CALL_VISIT_EXPR_EH_(EQNode, e);
  CALL_VISIT_EXPR_EH_(NENode, e);
  CALL_VISIT_EXPR_EH_(LTNode, e);
  CALL_VISIT_EXPR_EH_(LENode, e);
  CALL_VISIT_EXPR_EH_(GTNode, e);
  CALL_VISIT_EXPR_EH_(GENode, e);
  CALL_VISIT_EXPR_EH_(AndNode, e);
  CALL_VISIT_EXPR_EH_(OrNode, e);
  CALL_VISIT_EXPR_EH_(VarNode, e);
  CALL_VISIT_EXPR_EH_(SizeVarNode, e);
  CALL_VISIT_EXPR_EH_(LoadNode, e);
  CALL_VISIT_EXPR_EH_(LetNode, e);
  CALL_VISIT_EXPR_EH_(CallNode, e);
  CALL_VISIT_EXPR_EH_(IntImmNode, e);
  CALL_VISIT_EXPR_EH_(FloatImmNode, e);
  CALL_VISIT_EXPR_EH_(StringImmNode, e);
  CALL_VISIT_EXPR_EH_(ReduceNode, e);
  CALL_VISIT_EXPR_EH_(CastNode, e);
  CALL_VISIT_EXPR_EH_(NotNode, e);
  CALL_VISIT_EXPR_EH_(SelectNode, e);
  CALL_VISIT_EXPR_EH_(FuseSelectNode, e);
  CALL_VISIT_EXPR_EH_(RampNode, e);
  CALL_VISIT_EXPR_EH_(ShuffleNode, e);
  CALL_VISIT_EXPR_EH_(BroadcastNode, e);
  return 0;
}

size_t DeeperExprHash::VisitExpr_(const LoadNode* op) const {
  return this->VisitExpr(op->buffer_var) ^ this->VisitExpr(op->index) ^
         this->VisitExpr(op->predicate);
}

size_t DeeperExprHash::VisitExpr_(const CallNode* op) const {
  return std::hash<const Object*>()(op->func.get()) ^
         VisitArray(op->args, [this](const PrimExpr& e) { return this->VisitExpr(e); });
}

}  // namespace tir
}  // namespace tvm
