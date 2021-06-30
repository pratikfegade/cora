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

 private:
  struct FunKey {
    Dimension dimension;
    std::set<const Object*> dependent_dimensions;
  };

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
};
}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_FUNCTION_GENERATOR_H_
