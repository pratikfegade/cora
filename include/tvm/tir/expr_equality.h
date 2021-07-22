#pragma once
#include <tvm/tir/expr.h>

namespace tvm {
namespace tir {
class ExprEquality {
  bool VisitExpr_(const VarNode* op1, const VarNode* op2) const;

  bool VisitExpr_(const SizeVarNode* op1, const SizeVarNode* op2) const;

  bool VisitExpr_(const LoadNode* op1, const LoadNode* op2) const;

  bool VisitExpr_(const LetNode* op1, const LetNode* op2) const;

  virtual bool VisitExpr_(const CallNode* op1, const CallNode* op2) const;

#define DEFINE_BINOP_VISIT_DECL_EE_(OP) bool VisitExpr_(const OP* op1, const OP* op2) const;

  DEFINE_BINOP_VISIT_DECL_EE_(AddNode);
  DEFINE_BINOP_VISIT_DECL_EE_(SubNode);
  DEFINE_BINOP_VISIT_DECL_EE_(MulNode);
  DEFINE_BINOP_VISIT_DECL_EE_(DivNode);
  DEFINE_BINOP_VISIT_DECL_EE_(ModNode);
  DEFINE_BINOP_VISIT_DECL_EE_(FloorDivNode);
  DEFINE_BINOP_VISIT_DECL_EE_(FloorModNode);
  DEFINE_BINOP_VISIT_DECL_EE_(MinNode);
  DEFINE_BINOP_VISIT_DECL_EE_(MaxNode);
  DEFINE_BINOP_VISIT_DECL_EE_(EQNode);
  DEFINE_BINOP_VISIT_DECL_EE_(NENode);
  DEFINE_BINOP_VISIT_DECL_EE_(LTNode);
  DEFINE_BINOP_VISIT_DECL_EE_(LENode);
  DEFINE_BINOP_VISIT_DECL_EE_(GTNode);
  DEFINE_BINOP_VISIT_DECL_EE_(GENode);
  DEFINE_BINOP_VISIT_DECL_EE_(AndNode);
  DEFINE_BINOP_VISIT_DECL_EE_(OrNode);

  bool VisitExpr_(const IntImmNode* op1, const IntImmNode* op2) const;
  bool VisitExpr_(const FloatImmNode* op1, const FloatImmNode* op2) const;
  bool VisitExpr_(const StringImmNode* op1, const StringImmNode* op2) const;

  bool VisitExpr_(const ReduceNode* op1, const ReduceNode* op2) const;

  bool VisitExpr_(const CastNode* op1, const CastNode* op2) const;

  bool VisitExpr_(const NotNode* op1, const NotNode* op2) const;

  bool VisitExpr_(const SelectNode* op1, const SelectNode* op2) const;

  bool VisitExpr_(const RampNode* op1, const RampNode* op2) const;

  bool VisitExpr_(const ShuffleNode* op1, const ShuffleNode* op2) const;

  bool VisitExpr_(const BroadcastNode* op1, const BroadcastNode* op2) const;

 public:
  template <typename T, typename F>
  inline bool VisitArray(const Array<T>& arr1, const Array<T>& arr2, F fvisit) const {
    if (arr1.size() != arr2.size()) return false;

    for (size_t i = 0; i < arr1.size(); i++) {
      if (!fvisit(arr1[i], arr2[i])) return false;
    }
    return true;
  }

  bool VisitExpr(PrimExpr e1, PrimExpr e2) const;

  bool VisitExprConst(const PrimExpr e1, const PrimExpr e2) const;

  bool operator()(const PrimExpr& e1, const PrimExpr& e2) const {
    return this->VisitExprConst(e1, e2);
  };
};

class ExprHash {
  size_t VisitExpr_(const VarNode* op1) const;

  size_t VisitExpr_(const SizeVarNode* op1) const;

  size_t VisitExpr_(const LoadNode* op1) const;

  size_t VisitExpr_(const LetNode* op1) const;

  virtual size_t VisitExpr_(const CallNode* op1) const;

#define DEFINE_BINOP_VISIT_DECL_EH_(OP) size_t VisitExpr_(const OP* op1) const;

  DEFINE_BINOP_VISIT_DECL_EH_(AddNode);
  DEFINE_BINOP_VISIT_DECL_EH_(SubNode);
  DEFINE_BINOP_VISIT_DECL_EH_(MulNode);
  DEFINE_BINOP_VISIT_DECL_EH_(DivNode);
  DEFINE_BINOP_VISIT_DECL_EH_(ModNode);
  DEFINE_BINOP_VISIT_DECL_EH_(FloorDivNode);
  DEFINE_BINOP_VISIT_DECL_EH_(FloorModNode);
  DEFINE_BINOP_VISIT_DECL_EH_(MinNode);
  DEFINE_BINOP_VISIT_DECL_EH_(MaxNode);
  DEFINE_BINOP_VISIT_DECL_EH_(EQNode);
  DEFINE_BINOP_VISIT_DECL_EH_(NENode);
  DEFINE_BINOP_VISIT_DECL_EH_(LTNode);
  DEFINE_BINOP_VISIT_DECL_EH_(LENode);
  DEFINE_BINOP_VISIT_DECL_EH_(GTNode);
  DEFINE_BINOP_VISIT_DECL_EH_(GENode);
  DEFINE_BINOP_VISIT_DECL_EH_(AndNode);
  DEFINE_BINOP_VISIT_DECL_EH_(OrNode);

  size_t VisitExpr_(const IntImmNode* op1) const;
  size_t VisitExpr_(const FloatImmNode* op1) const;
  size_t VisitExpr_(const StringImmNode* op1) const;

  size_t VisitExpr_(const ReduceNode* op1) const;

  size_t VisitExpr_(const CastNode* op1) const;

  size_t VisitExpr_(const NotNode* op1) const;

  size_t VisitExpr_(const SelectNode* op1) const;

  size_t VisitExpr_(const RampNode* op1) const;

  size_t VisitExpr_(const ShuffleNode* op1) const;

  size_t VisitExpr_(const BroadcastNode* op1) const;

 public:
  template <typename T, typename F>
  inline size_t VisitArray(const Array<T>& arr, F fvisit) const {
    auto hash = 67;
    for (size_t i = 0; i < arr.size(); i++) {
      hash = hash ^ fvisit(arr[i]);
    }
    return hash;
  }

  size_t VisitExpr(PrimExpr e1) const;

  size_t VisitExprConst(const PrimExpr e1) const;

  size_t operator()(const PrimExpr& e) const { return this->VisitExprConst(e); };
};
}  // namespace tir
}  // namespace tvm
