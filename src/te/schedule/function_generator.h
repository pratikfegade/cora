#ifndef TVM_TE_FUNCTION_GENERATOR_H_
#define TVM_TE_FUNCTION_GENERATOR_H_

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace te {

class AFunGenerator {
 public:
  AFunGenerator(const Schedule& sch_) : sch(sch_) {}

  Stmt GenerateAndSetAFuns();

 private:
  UninterpFun SetAFun(Modes layout, int idx, UninterpFun a_fun_shell);

  Schedule sch;
  Map<Dimension, UninterpFun> dim_afun_map;
  Array<Stmt> stmts;
};
}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_FUNCTION_GENERATOR_H_
