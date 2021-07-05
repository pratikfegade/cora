#include "function_generator.h"

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <unordered_set>

#include "../../tir/ir/var_replacer.h"
#include "../../tir/pass/ir_util.h"
#include "message_passing.h"

namespace tvm {
namespace te {

size_t AFunGenerator::FunKeyHasher::operator()(const FunKey& pattern) const {
  size_t hash = std::hash<const Object*>{}(pattern.dimension.get());

  for (auto dim : pattern.dependent_dimensions) {
    hash ^= std::hash<const Object*>{}(dim);
  }

  return hash;
}

// TODO: Consider using the associated l_funs to determine dependent
// dimension equality as multiple dimensions might have the same
// l_funs
bool AFunGenerator::FunKeyEquality::operator()(const FunKey& p1, const FunKey& p2) const {
  bool ret = false;

  if (p1.dimension != p2.dimension) return false;
  return (p1.dependent_dimensions == p2.dependent_dimensions);
}

AFunGenerator::FunKey make_key(const Modes& layout, const int& idx) {
  std::cout << "[AFG] Making key " << layout->dimensions[idx] << std::endl;
  auto transitive_dependent_dims = layout->get_transitive_dependent_dims(idx);
  std::multiset<const Object*> transitive_dependent_dims_set;
  for (auto dim : transitive_dependent_dims) {
    auto l_fun = layout->l_funs[layout->dimensions.GetIdx(dim)];
    std::cout << "[AFG]   Dep " << dim << " " << l_fun << std::endl;
    transitive_dependent_dims_set.insert(l_fun.get());
  }
  return {layout->dimensions[idx], transitive_dependent_dims_set};
}

Stmt AFunGenerator::GenerateAndSetAFuns() {
  for (Stage s : sch->stages) {
    for (size_t i = 0; i < s->op->num_outputs(); ++i) {
      Modes layout = s->op->output_layout(i);
      if (layout.defined()) {
        for (size_t i = 0; i < layout->ndim(); ++i) {
          if (layout->a_funs[i].defined() && layout->a_funs[i]->body.defined()) {
            FunKey key = make_key(layout, i);
            dim_afun_map[key] = layout->a_funs[i];
          }
        }
      }
    }
  }

  for (Stage s : sch->stages) {
    for (size_t i = 0; i < s->op->num_outputs(); ++i) {
      Modes layout = s->op->output_layout(i);
      if (layout.defined()) {
        // std::cout << "[AFG] Op " << s->op << std::endl;
        for (int i = layout->ndim() - 1; i >= 0; --i) {
          if (!layout->has_dependent_dims(i)) {
            continue;
          }
          SetAFun(layout, i, layout->a_funs[i]);
        }
      }
    }
  }

  return SeqStmt(stmts);
}

void copy_body_to_ufun_shell(UninterpFun fun, UninterpFun shell) {
  PrimExpr body = fun->body;
  CHECK(body.defined());
  CHECK_EQ(fun->arity(), shell->arity());
  std::unordered_map<const VarNode*, PrimExpr> sub;
  for (size_t i = 0; i < fun->arity(); ++i) {
    sub[fun->parameters[i].as<VarNode>()] = shell->parameters[i];
  }
  const_cast<UninterpFunNode*>(shell.as<UninterpFunNode>())->SetBody(VarReplacer(sub)(body));
  const_cast<UninterpFunNode*>(shell.as<UninterpFunNode>())->SetRange(fun->range);
}

UninterpFun AFunGenerator::SetAFun(Modes layout, int idx, UninterpFun a_fun_shell) {
  // std::cout << "[AFG] Wanting to generate body for " << a_fun_shell << std::endl;
  if (a_fun_shell->body.defined()) {
    return a_fun_shell;
  }

  auto transitive_dependent_dims = layout->get_transitive_dependent_dims(idx);
  Dimension dim = layout->dimensions[idx];
  FunKey key = make_key(layout, idx);
  if (dim_afun_map.count(key)) {
    // std::cout << "[AFG]   Copying body to " << a_fun_shell << std::endl;
    copy_body_to_ufun_shell(dim_afun_map[key], a_fun_shell);
  } else {
    std::string prefix = dim->name + "_af" + std::to_string(count++) + "_";
    Var loop_var = Var(prefix + "i", DataType::Int(32));
    PrimExpr body_expr = 1;

    std::unordered_set<int> handled_already;
    PrimExpr a_fun_max_extent = 1;
    for (auto dependent_dim : transitive_dependent_dims) {
      int dependent_dim_idx = layout->dimensions.GetIdx(dependent_dim);
      a_fun_max_extent = a_fun_max_extent * layout->l_funs[dependent_dim_idx]->range->extent;
      if (handled_already.count(dependent_dim_idx)) {
        continue;
      }
      if (layout->has_dependent_dims(dependent_dim_idx)) {
        UninterpFun a_fun = SetAFun(layout, dependent_dim_idx, layout->a_funs[dependent_dim_idx]);
        PrimExpr a_fun_call = a_fun.MakeCallTo({loop_var}, {dim});
        body_expr = body_expr * a_fun_call;

        for (auto dim : layout->get_transitive_dependent_dims(dependent_dim_idx)) {
          handled_already.insert(layout->dimensions.GetIdx(dim));
        }
      } else {
        UninterpFun l_fun = layout->l_funs[dependent_dim_idx];
        PrimExpr l_fun_call = l_fun.MakeCallTo({loop_var}, {dim});
        body_expr = body_expr * l_fun_call;
      }
    }

    PrimExpr buf_extent = layout->l_funs[idx]->range->max_inclusive();
    // std::cout << "[ASDC]   Buffer range " << layout->l_funs[idx]->range << std::endl;
    Buffer a_fun_buffer = decl_buffer({buf_extent}, DataType::Int(32), prefix + "buf");
    Buffer a_fun_counter = decl_buffer({1}, DataType::Int(32), prefix + "ctr");

    Stmt fun_store = a_fun_buffer.vstore({loop_var}, a_fun_counter.vload({0}, DataType::Int(32)));
    Stmt counter_incr =
        a_fun_counter.vstore({loop_var}, a_fun_counter.vload({0}, DataType::Int(32)) + body_expr);
    SeqStmt loop_stmts = SeqStmt({fun_store, counter_incr});
    Stmt stmt =
        ForNode::make(loop_var, 0, buf_extent, ForType::Serial, DeviceAPI::None, loop_stmts);

    Stmt counter_init = a_fun_counter.vstore({0}, 0);
    stmt = SeqStmt({counter_init, stmt});
    stmt =
        AttrStmtNode::make(a_fun_buffer->data, attr::storage_scope, StringImmNode::make("global"),
                           AllocateNode::make(a_fun_buffer->data, DataType::Int(32), {buf_extent},
                                              IntImm(DataType::Bool(1), 1), stmt));
    stmt =
        AttrStmtNode::make(a_fun_counter->data, attr::storage_scope, StringImmNode::make("global"),
                           AllocateNode::make(a_fun_counter->data, DataType::Int(32), {1},
                                              IntImm(DataType::Bool(1), 1), stmt));
    stmts.push_back(stmt);

    CHECK_EQ(a_fun_shell->parameters.size(), 1);
    Var param = a_fun_shell->parameters[0];
    const_cast<UninterpFunNode*>(a_fun_shell.as<UninterpFunNode>())
        ->SetBody(a_fun_counter.vload({param}, DataType::Int(32)));

    dim_afun_map[key] = a_fun_shell;
    // std::cout << "[AFG]   Generated body for " << a_fun_shell << std::endl;
  }
  return a_fun_shell;
}

Stmt RaggedFusionBoundStmtsGenerator::generate(Stmt main_body) {
  Array<Stmt> fusion_stmts;
  for (Stage s : sch->stages) {
    for (int i = s->relations.size() - 1; i >= 0; --i) {
      if (auto frel = s->relations[i].as<RaggedFuseNode>()) {
        main_body = generate_fusion_statements(s, frel, main_body);
      }
    }
  }

  return main_body;
}

PrimExpr RaggedFusionBoundStmtsGenerator::root_ivs_fused(Stage& stage, Array<IterVar> fused_ivs) {
  // Modes loop_layout = stage->op->loop_layout();
  // if (!loop_layout.defined() || loop_layout->a_funs.size() == 0) {
  //   return NullValue<PrimExpr>();
  // }

  // std::unordered_map<const Object*, IterVar> root_vars_to_ivs;
  // Array<IterVar> root_vars = stage->op->root_iter_vars();
  // for (auto rv : stage->op->root_iter_vars()) {
  //   root_vars_to_ivs[rv->var.get()] = rv;
  // }

  // Array<PrimExpr> var_values = get_iter_var_values(fused_ivs, stage);
  // Array<Dimension> fused_dims;
  // Array<PrimExpr> fused_vars;
  // // These checks can be made less conservative
  // for (size_t i = 0; i < fused_ivs.size(); ++i) {
  //   Var var = fused_ivs[i]->var;
  //   PrimExpr value = var_values[i];
  //   std::cout << "[GFS]  Value " << var << " " << value << std::endl;
  //   auto var_node = value.as<VarNode>();
  //   if (var_node == nullptr || !root_vars_to_ivs.count(var_node)) {
  //     return NullValue<PrimExpr>();
  //   }
  //   IterVar root_iv = root_vars_to_ivs.at(var_node);
  //   fused_dims.push_back(loop_layout->dimensions[root_vars.GetIdx(root_iv)]);
  //   fused_vars.push_back(root_iv->var);
  // }

  // for (auto dim : fused_dims) {
  //   for (auto dependent_dim : loop_layout->get_dependent_dimensions(dim)) {
  //     if (!fused_dims.Contains(dependent_dim)) return NullValue<PrimExpr>();
  //   }
  // }

  // PrimExpr fused_var_val = loop_layout->ComputePosition("mummy", fused_vars, fused_dims);
  // std::cout << "[GFS]   Fused var val " << fused_var_val << std::endl;
  // return fused_var_val;
  return NullValue<PrimExpr>();
}

bool is_constant(PrimExpr expr, Array<IterVar> iter_vars) {
  class Visitor : public ExprVisitor {
   public:
    Visitor(Array<IterVar> iter_vars_) {
      for (auto iv : iter_vars_) {
        iter_vars.insert(iv->var.get());
      }
    }

    void VisitExpr_(const VarNode* op) {
      if (iter_vars.count(op)) {
        // std::cout << "[ISC] Found var " << op->name_hint << std::endl;
        non_constant = true;
      }
    }

    std::unordered_set<const Object*> iter_vars;
    bool non_constant{false};
  };

  Visitor v(iter_vars);
  v(expr);
  return !v.non_constant;
}

Stmt RaggedFusionBoundStmtsGenerator::generate_fusion_statements(Stage& stage,
                                                                 const RaggedFuseNode* rel,
                                                                 Stmt main_body) {
  // std::cout << "[GFS] Generating fusion for " << stage << std::endl;
  CHECK(stage.is_ancestor_attached_at_root());

  IterVar outer = rel->outer;
  IterVar inner = rel->inner;
  IterVar fused = rel->fused;
  Range outer_dom = dom_map.at(outer);
  Range inner_dom = dom_map.at(inner);
  Range fused_dom = dom_map.at(fused);
  CHECK(is_zero(outer_dom->min));
  CHECK(is_zero(inner_dom->min));
  CHECK(is_zero(fused_dom->min));

  PrimExpr fused_var_val = root_ivs_fused(stage, {outer, inner});

  PrimExpr outer_extent_relaxed = Simplify(UninterpFun::InlineUninterpFunCalls(
      UninterpFun::RelaxComplexUninterpCallsMaxInclusive(outer_dom->max_exclusive())));
  PrimExpr inner_extent_relaxed = Simplify(UninterpFun::InlineUninterpFunCalls(
      UninterpFun::RelaxComplexUninterpCallsMaxInclusive(inner_dom->max_exclusive())));
  PrimExpr fused_extent_relaxed = outer_extent_relaxed * inner_extent_relaxed;

  // std::cout << "[GFS]   Outer " << outer_dom << " " << outer_extent_relaxed << std::endl;
  // std::cout << "[GFS]   Inner " << inner_dom << " " << inner_extent_relaxed << std::endl;
  // std::cout << "[GFS]   Fused " << fused_dom << " " << fused_extent_relaxed << std::endl;

  // std::cout << "[GFS]   Extents " << outer_extent << " " << inner_extent << " " << fused_extent
  // << std::endl;

  // Allocate buffers
  Buffer fused_to_inner =
      decl_buffer({fused_extent_relaxed}, DataType::Int(32), "fi" + std::to_string(count));
  Buffer fused_to_outer =
      decl_buffer({fused_extent_relaxed}, DataType::Int(32), "fo" + std::to_string(count));
  Buffer outer_inner_to_fused = decl_buffer({outer_extent_relaxed, inner_extent_relaxed},
                                            DataType::Int(32), "oif" + std::to_string(count));
  Buffer fused_val = decl_buffer({1}, DataType::Int(32), "f" + std::to_string(count));
  count++;

  PrimExpr outer_loop_extent = outer_dom->extent;
  PrimExpr inner_loop_extent_unreplaced = inner_dom->extent;
  PrimExpr inner_loop_extent = inner_loop_extent_unreplaced;
  CHECK(is_constant(outer_loop_extent, stage->all_iter_vars)) << " " << outer_loop_extent;
  {
    std::unordered_map<IterVar, PrimExpr> state;
    state[outer] = outer->var;
    PassUpIndex(stage, dom_map, &state, true);
    std::unordered_map<const VarNode*, PrimExpr> vsub;
    for (auto it : state) {
      // std::cout << "[GFS]     Replace " << it.first->var << " " << it.second << std::endl;
      vsub[it.first->var.as<VarNode>()] = it.second;
    }
    inner_loop_extent = VarReplacer(vsub)(inner_loop_extent);
    // std::cout << "[GFS]       Replaced " << inner_loop_extent_unreplaced << " " <<
    // inner_loop_extent
    // << std::endl;
  }

  // std::cout << "[FFG] Outer loop extent " << outer_loop_extent << std::endl;

  // Compute the outer and inner variables in terms of the root itervars
  PrimExpr outer_value = outer->var;
  PrimExpr inner_value = inner->var;

  Stmt no_op = EvaluateNode::make(0);
  std::vector<Stmt> for_loops = {
      ForNode::make(outer->var, 0, outer_loop_extent, ForType::Serial, DeviceAPI::None, no_op),
      ForNode::make(inner->var, 0, inner_loop_extent, ForType::Serial, DeviceAPI::None, no_op)};

  Stmt body = NullValue<Stmt>();
  {
    PrimExpr fused_val_load = fused_val.vload({0}, DataType::Int(32));
    Stmt fused_store = EvaluateNode::make(0);
    if (!fused_var_val.defined()) {
      fused_store = outer_inner_to_fused.vstore({outer_value, inner_value}, fused_val_load);
    }
    Stmt outer_store = fused_to_outer.vstore({fused_val_load}, outer_value);
    Stmt inner_store = fused_to_inner.vstore({fused_val_load}, inner_value);
    Stmt fused_incr = fused_val.vstore({0}, fused_val_load + 1);
    body = SeqStmt({fused_store, outer_store, inner_store, fused_incr});
  }

  body = MergeNest(for_loops, body);
  body = SeqStmt({fused_val.vstore({0}, 0), body, main_body});
  body = AttrStmtNode::make(
      fused_to_inner->data, attr::storage_scope, StringImmNode::make("global"),
      AllocateNode::make(fused_to_inner->data, DataType::Int(32), {fused_extent_relaxed},
                         IntImm(DataType::Bool(1), 1), body));
  body = AttrStmtNode::make(
      fused_to_outer->data, attr::storage_scope, StringImmNode::make("global"),
      AllocateNode::make(fused_to_outer->data, DataType::Int(32), {fused_extent_relaxed},
                         IntImm(DataType::Bool(1), 1), body));
  body = AttrStmtNode::make(
      outer_inner_to_fused->data, attr::storage_scope, StringImmNode::make("global"),
      AllocateNode::make(outer_inner_to_fused->data, DataType::Int(32), {fused_extent_relaxed},
                         IntImm(DataType::Bool(1), 1), body));
  body = AttrStmtNode::make(fused_val->data, attr::storage_scope, StringImmNode::make("global"),
                            AllocateNode::make(fused_val->data, DataType::Int(32), {1},
                                               IntImm(DataType::Bool(1), 1), body));

  // std::cout << "[GFS]  Stmt\n" << body << std::endl;

  auto init_uf = [&](UninterpFun uf, PrimExpr max_extent, Buffer loadee,
                     PrimExpr body = NullValue<PrimExpr>()) {
    UninterpFunNode* uf_node = const_cast<UninterpFunNode*>(uf.as<UninterpFunNode>());
    Array<PrimExpr> extents;
    for (auto param : uf->parameters) extents.push_back(param);
    if (body.defined()) {
      uf_node->SetBody(body);
    } else {
      uf_node->SetBody(loadee.vload(extents, DataType::Int(32)));
    }
    uf_node->SetRange(Range::make_by_min_extent(0, max_extent));
  };

  init_uf(rel->fused_to_outer_uf, outer_extent_relaxed, fused_to_outer);
  init_uf(rel->fused_to_inner_uf, inner_extent_relaxed, fused_to_inner);
  init_uf(rel->outer_inner_to_fused_uf, fused_extent_relaxed, outer_inner_to_fused, fused_var_val);

  return body;
}

Array<PrimExpr> RaggedFusionBoundStmtsGenerator::get_iter_var_values(Array<IterVar> vars,
                                                                     Stage& stage) {
  std::unordered_map<IterVar, PrimExpr> state;
  for (auto rv : stage->op->root_iter_vars()) {
    state[rv] = rv->var;
  }
  PassDownIndex(stage, dom_map, &state, true);
  Array<PrimExpr> results;
  for (auto var : vars) {
    CHECK(state.count(var));
    results.push_back(state.at(var));
  }
  return results;
}
}  // namespace te
}  // namespace tvm
