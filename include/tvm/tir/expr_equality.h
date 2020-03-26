#pragma once
#include <tvm/tir/expr.h>

namespace tvm {
  namespace tir {
    class ExprEquality {
      bool VisitExpr_(const VarNode* op1, const VarNode* op2);

      bool VisitExpr_(const SizeVarNode* op1, const SizeVarNode* op2);

      bool VisitExpr_(const LoadNode* op1, const LoadNode* op2);

      bool VisitExpr_(const LetNode* op1, const LetNode* op2);

      bool VisitExpr_(const CallNode* op1, const CallNode* op2);

#define DEFINE_BINOP_VISIT_DECL_(OP)				\
      bool VisitExpr_(const OP* op1, const OP* op2);    \

      DEFINE_BINOP_VISIT_DECL_(AddNode);
      DEFINE_BINOP_VISIT_DECL_(SubNode);
      DEFINE_BINOP_VISIT_DECL_(MulNode);
      DEFINE_BINOP_VISIT_DECL_(DivNode);
      DEFINE_BINOP_VISIT_DECL_(ModNode);
      DEFINE_BINOP_VISIT_DECL_(FloorDivNode);
      DEFINE_BINOP_VISIT_DECL_(FloorModNode);
      DEFINE_BINOP_VISIT_DECL_(MinNode);
      DEFINE_BINOP_VISIT_DECL_(MaxNode);
      DEFINE_BINOP_VISIT_DECL_(EQNode);
      DEFINE_BINOP_VISIT_DECL_(NENode);
      DEFINE_BINOP_VISIT_DECL_(LTNode);
      DEFINE_BINOP_VISIT_DECL_(LENode);
      DEFINE_BINOP_VISIT_DECL_(GTNode);
      DEFINE_BINOP_VISIT_DECL_(GENode);
      DEFINE_BINOP_VISIT_DECL_(AndNode);
      DEFINE_BINOP_VISIT_DECL_(OrNode);

      bool VisitExpr_(const IntImmNode* op1, const IntImmNode* op2);
      bool VisitExpr_(const FloatImmNode* op1, const FloatImmNode* op2);
      bool VisitExpr_(const StringImmNode* op1, const StringImmNode* op2);

      bool VisitExpr_(const ReduceNode* op1, const ReduceNode* op2);

      bool VisitExpr_(const CastNode* op1, const CastNode* op2);

      bool VisitExpr_(const NotNode* op1, const NotNode* op2);

      bool VisitExpr_(const SelectNode* op1, const SelectNode* op2);

      bool VisitExpr_(const RampNode* op1, const RampNode* op2);

      bool VisitExpr_(const ShuffleNode* op1, const ShuffleNode* op2);

      bool VisitExpr_(const BroadcastNode* op1, const BroadcastNode* op2);

    public:
      bool VisitExpr(PrimExpr e1, PrimExpr e2);
    };
  }
}
