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
  AFunGenerator(Modes layout_) : layout(layout_) {
    for (size_t i = 0; i < layout->ndim(); ++i) {
      if (layout->a_funs[i].defined() && layout->a_funs[i]->body.defined()) {
        generated_a_funs.Set(layout->dimensions[i], layout->a_funs[i]);
      }
    }
  }

  Stmt GenerateAndSetAFuns();

  UninterpFun SetAFun(int idx, UninterpFun a_fun_shell);

  Modes layout;
  Map<Dimension, UninterpFun> generated_a_funs;
  Array<Stmt> stmts;
};
}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_FUNCTION_GENERATOR_H_
