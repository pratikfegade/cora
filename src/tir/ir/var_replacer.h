#ifndef TVM_TIR_VAR_REPLACER_H_
#define TVM_TIR_VAR_REPLACER_H_

#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>
#include <unordered_map>

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
}
}

#endif
