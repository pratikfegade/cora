#ifndef TVM_TIR_VAR_REPLACER_H_
#define TVM_TIR_VAR_REPLACER_H_

#include <tvm/te/operation.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace tir {
class VarReplacer : public StmtExprMutator {
 public:
  explicit VarReplacer(const std::unordered_map<const VarNode*, PrimExpr>& vsub,
                       bool replace_buffers = false)
      : vsub_(vsub), replace_buffers_(replace_buffers) {}

  CommReducer MutateCommReducer(CommReducer combiner);

  PrimExpr VisitExpr_(const VarNode* op) final;

  PrimExpr VisitExpr_(const tir::ReduceNode* op) final;

  PrimExpr VisitExpr_(const LoadNode* op) final;

  Stmt VisitStmt_(const StoreNode* op) final;

  Stmt VisitStmt_(const AttrStmtNode* op) final;

  Range replace(const Range& r) {
    return Range::make_by_min_extent(this->VisitExpr(r->min), this->VisitExpr(r->extent));
  }

 private:
  const std::unordered_map<const VarNode*, PrimExpr>& vsub_;
  bool replace_buffers_;
};

class VarFinder : public StmtExprVisitor {
 public:
  explicit VarFinder(const std::unordered_set<const VarNode*>& vset) : vset_(vset) {}

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
  VarCollector(bool collect_buffers_ = false) : collect_buffers(collect_buffers_) {}

  void VisitExpr_(const VarNode* op) final;

  void VisitExpr_(const LoadNode* op) final;

  void VisitStmt_(const StoreNode* op) final;

  std::unordered_set<const VarNode*> collect(const PrimExpr& e) {
    this->VisitExpr(e);
    return this->collected;
  }

  std::unordered_set<const VarNode*> collect(const Stmt& s) {
    this->VisitStmt(s);
    return this->collected;
  }

  std::unordered_set<const VarNode*> collect(const Range& r) {
    this->VisitExpr(r->min);
    this->VisitExpr(r->extent);
    return this->collected;
  }

  std::unordered_set<const VarNode*> getCollected() { return collected; }

 private:
  bool collect_buffers;
  std::unordered_set<const VarNode*> collected;
};

class TensorCallCollector : public StmtExprVisitor {
 public:
  void VisitExpr_(const CallNode* op) final;

  void collect(const PrimExpr& e) { this->VisitExpr(e); }

  std::unordered_set<const te::OperationNode*> getCollected() { return this->collected; }

 private:
  std::unordered_set<const te::OperationNode*> collected;
};
}  // namespace tir
}  // namespace tvm

#endif
