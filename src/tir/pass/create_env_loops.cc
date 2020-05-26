#include <tvm/arith/analyzer.h>
#include <tvm/arith/z3_analyzer.h>
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_equality.h>
#include <tvm/tir/lowered_func.h>
#include <tvm/tir/stmt_functor.h>

#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "../../arith/interval_set.h"
#include "../../runtime/thread_storage_scope.h"

#define COUT std::cout << "[RIfR] "
namespace tvm {
namespace tir {
class EnvLoopsCreator : public StmtMutator {
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      Var var = Downcast<IterVar>(op->node)->var;
      PrimExpr max = op->value;
      Stmt ret = ForNode::make(var, 0, op->value, ForType::Parallel, DeviceAPI::None,
                               StmtMutator::VisitStmt(op->body));
      return ret;
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }

 public:
  EnvLoopsCreator() {}
};

LoweredFunc CreateEnvLoopsForFunc(LoweredFunc f, std::string target) {
  if (target != "llvm" && target != "c") return f;
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  Stmt body = f->body;
  body = EnvLoopsCreator()(body);
  n->body = body;
  return LoweredFunc(n);
}

Stmt CreateEnvLoopsForStmt(Stmt stmt, std::string target) {
  if (target != "llvm" && target != "c") return stmt;
  Stmt ret = EnvLoopsCreator()(stmt);
  return ret;
}
}  // namespace tir
}  // namespace tvm
#undef COUT
