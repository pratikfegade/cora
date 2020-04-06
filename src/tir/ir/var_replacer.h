#ifndef TVM_TIR_VAR_REPLACER_H_
#define TVM_TIR_VAR_REPLACER_H_

#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace tir {
class VarReplacer : public StmtExprMutator {
 public:
  explicit VarReplacer(
      const std::unordered_map<const VarNode*, PrimExpr>& vsub)
      : vsub_(vsub) {}

  CommReducer MutateCommReducer(CommReducer combiner);

  PrimExpr VisitExpr_(const VarNode* op) final;

  PrimExpr VisitExpr_(const tir::ReduceNode* op) final;

 private:
  const std::unordered_map<const VarNode*, PrimExpr>& vsub_;
};

class VarFinder : public StmtExprVisitor {
 public:
  explicit VarFinder(
      const std::unordered_set<const VarNode*>& vset)
      : vset_(vset) {}

  void VisitExpr_(const VarNode* op) final;

  bool find(const PrimExpr& e) {
    this->VisitExpr(e);
    return this->found;
  }

 private:
  const std::unordered_set<const VarNode*>& vset_;
  bool found{false};
};

class VarCollector : public StmtExprVisitor {
 public:
  void VisitExpr_(const VarNode* op) final;

  std::unordered_set<const VarNode*> collect(const PrimExpr& e) {
    this->VisitExpr(e);
    return this->collected;
  }

 private:
  std::unordered_set<const VarNode*> collected;
};
}
}

#endif
