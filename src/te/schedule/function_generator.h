#ifndef TVM_TE_FUNCTION_GENERATOR_H_
#define TVM_TE_FUNCTION_GENERATOR_H_

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <set>
#include <unordered_map>

namespace tvm {
namespace te {

class AFunGenerator {
 public:
  AFunGenerator(const Schedule& sch_) : sch(sch_) {}

  Stmt GenerateAndSetAFuns();

  struct FunKey {
    Dimension dimension;
    std::multiset<const Object*> dependent_dimensions;
  };

 private:
  class FunKeyHasher {
   public:
    size_t operator()(const FunKey& pattern) const;
  };

  class FunKeyEquality {
   public:
    bool operator()(const FunKey& p1, const FunKey& p2) const;
  };

  UninterpFun SetAFun(Modes layout, int idx, UninterpFun a_fun_shell);

  Schedule sch;
  std::unordered_map<FunKey, UninterpFun, FunKeyHasher, FunKeyEquality> dim_afun_map;
  Array<Stmt> stmts;
  int count{0};
};

class RaggedFusionBoundStmtsGenerator : public StmtExprMutator {
 public:
  RaggedFusionBoundStmtsGenerator(Schedule& sch_, std::unordered_map<IterVar, Range>& dom_map_)
      : sch(sch_), dom_map(dom_map_), count(0) {}

  Stmt generate(Stmt main_body);

 private:
  PrimExpr root_ivs_fused(Stage& stage, Array<IterVar> fused_ivs);

  Stmt generate_fusion_statements(Stage& stage, const RaggedFuseNode* rel, Stmt main_body);

  Array<PrimExpr> get_iter_var_values(Array<IterVar> vars, Stage& stage);

  Schedule& sch;
  std::unordered_map<IterVar, Range>& dom_map;
  int count;
};

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_FUNCTION_GENERATOR_H_
