#include "function_generator.h"

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

Stmt AFunGenerator::GenerateAndSetAFuns() {
  for (int i = layout->ndim() - 1; i >= 0; --i) {
    if (!layout->has_dependent_dims(i)) {
      continue;
    }
    SetAFun(i, layout->a_funs[i]);
  }
  return SeqStmt(stmts);
}

UninterpFun AFunGenerator::SetAFun(int idx, UninterpFun a_fun_shell) {
  if (a_fun_shell->body.defined()) {
    return a_fun_shell;
  }

  Dimension dim = layout->dimensions[idx];
  Var loop_var = Var(dim->name + "a_fun_idx", DataType::Int(32));

  PrimExpr body_expr = 1;
  auto transitive_dependent_dims = layout->get_transitive_dependent_dims(idx);

  std::unordered_set<int> handled_already;
  PrimExpr a_fun_max_extent = 1;
  for (auto dependent_dim : transitive_dependent_dims) {
    int dependent_dim_idx = layout->dimensions.GetIdx(dependent_dim);
    a_fun_max_extent = a_fun_max_extent * layout->l_funs[dependent_dim_idx]->range->extent;
    if (handled_already.count(dependent_dim_idx)) {
      continue;
    }
    if (layout->has_dependent_dims(dependent_dim_idx)) {
      UninterpFun a_fun = SetAFun(dependent_dim_idx, layout->a_funs[dependent_dim_idx]);
      PrimExpr a_fun_call = UninterpFun::MakeCallTo(a_fun, {loop_var}, {dim});
      body_expr = body_expr * a_fun_call;

      for (auto dim : layout->get_transitive_dependent_dims(dependent_dim_idx)) {
        handled_already.insert(layout->dimensions.GetIdx(dim));
      }
    } else {
      UninterpFun l_fun = layout->l_funs[dependent_dim_idx];
      PrimExpr l_fun_call = UninterpFun::MakeCallTo(l_fun, {loop_var}, {dim});
      body_expr = body_expr * l_fun_call;
    }
  }

  PrimExpr buf_extent = layout->l_funs[idx]->range->min + layout->l_funs[idx]->range->extent - 1;
  std::cout << "[ASDC]   Buffer range " << layout->l_funs[idx]->range << std::endl;
  Buffer a_fun_buffer = decl_buffer({buf_extent}, DataType::Int(32), dim->name + "a_fun_buf");
  Buffer a_fun_counter = decl_buffer({1}, DataType::Int(32), dim->name + "a_fun_ctr");

  Stmt fun_store = a_fun_buffer.vstore({loop_var}, a_fun_counter.vload({0}, DataType::Int(32)));
  Stmt counter_incr =
      a_fun_counter.vstore({loop_var}, a_fun_counter.vload({0}, DataType::Int(32)) + body_expr);
  SeqStmt loop_stmts = SeqStmt({fun_store, counter_incr});
  Stmt stmt = ForNode::make(loop_var, 0, buf_extent, ForType::Serial, DeviceAPI::None, loop_stmts);

  Stmt counter_init = a_fun_counter.vstore({0}, 0);
  stmt = SeqStmt({counter_init, stmt});
  stmt = AttrStmtNode::make(a_fun_buffer->data, attr::storage_scope, StringImmNode::make("global"),
                            AllocateNode::make(a_fun_buffer->data, DataType::Int(32), {buf_extent},
                                               IntImm(DataType::Bool(1), 1), stmt));
  stmt = AttrStmtNode::make(a_fun_counter->data, attr::storage_scope, StringImmNode::make("global"),
                            AllocateNode::make(a_fun_counter->data, DataType::Int(32), {1},
                                               IntImm(DataType::Bool(1), 1), stmt));
  stmts.push_back(stmt);

  CHECK_EQ(a_fun_shell->parameters.size(), 1);
  Var param = a_fun_shell->parameters[0];
  const_cast<UninterpFunNode*>(a_fun_shell.as<UninterpFunNode>())
      ->SetBody(a_fun_counter.vload({param}, DataType::Int(32)));
  return a_fun_shell;
}

}  // namespace te
}  // namespace tvm
