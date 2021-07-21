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

Buffer AllocationAggregator::create_buffer(Array<PrimExpr> extents, DataType buf_dtype,
                                           std::string name) {
  // CHECK_EQ(extents.size(), 1);
  // std::cout << "[ALLOC] Allocating buffer " << name << " " << extents[0] << " "
  // << aggregate_allocated_size << std::endl;
  CHECK_EQ(buf_dtype, dtype);
  Buffer buf = BufferNode::make(aggregate_buffer_var, dtype, extents, {}, aggregate_allocated_size,
                                name, "global", 0, 0, kDefault, kAll);

  PrimExpr size = 1;
  for (auto ext : extents) {
    size = size * ext;
  }

  aggregate_allocated_size = aggregate_allocated_size + size;

  return buf;
}

Buffer AllocationAggregator::aggregate_buffer() {
  return BufferNode::make(aggregate_buffer_var, dtype, {aggregate_allocated_size}, {}, 0,
                          aggregate_name, "global", 0, 0, kDefault, kAll);
}

size_t AFunctionGenerator::FunKeyHasher::operator()(const FunKey& pattern) const {
  size_t hash = std::hash<const Object*>{}(pattern.dimension.get());

  for (auto dim : pattern.dependent_dimensions) {
    hash ^= std::hash<const Object*>{}(dim);
  }

  return hash;
}

// TODO: Consider using the associated l_funs to determine dependent
// dimension equality as multiple dimensions might have the same
// l_funs
bool AFunctionGenerator::FunKeyEquality::operator()(const FunKey& p1, const FunKey& p2) const {
  if (p1.dimension != p2.dimension) return false;
  return (p1.dependent_dimensions == p2.dependent_dimensions);
}

AFunctionGenerator::FunKey make_key(const Modes& layout, const int& idx) {
  // std::cout << "[AFG] Making key " << layout->dimensions[idx] << std::endl;
  auto transitive_dependent_dims = layout->get_transitive_dependent_dims(idx);
  std::multiset<const Object*> transitive_dependent_dims_set;
  for (auto dim : transitive_dependent_dims) {
    auto l_fun = layout->l_funs[layout->dimensions.GetIdx(dim)];
    // std::cout << "[AFG]   Dep " << dim << " " << l_fun << std::endl;
    transitive_dependent_dims_set.insert(l_fun.get());
  }
  return {layout->dimensions[idx], transitive_dependent_dims_set};
}

Stmt allocate_both_bufs(std::pair<Buffer, Buffer> bufs, Array<PrimExpr> extents, Stmt body) {
  body = AttrStmtNode::make(bufs.first->data, attr::storage_scope, StringImmNode::make("global"),
                            AllocateNode::make(bufs.first->data, DataType::Int(32), extents,
                                               IntImm(DataType::Bool(1), 1), body));
  // body = AttrStmtNode::make(bufs.second->data, attr::storage_scope,
  // StringImmNode::make("global"), AllocateNode::make(bufs.second->data, DataType::Int(32),
  // extents, IntImm(DataType::Bool(1), 1), body));
  return body;
}

Stmt copy_bufs(std::pair<Buffer, Buffer> bufs, PrimExpr extent, DataType dtype) {
  return EvaluateNode::make(copy_to_device(
      bufs.first->data, 0, bufs.second->data, 0, extent * dtype.bytes(),
      Var("src_devtype_dummy", DataType::Handle()), Var("src_devid_dummy", DataType::Handle()),
      Var("dst_devtype_dummy", DataType::Handle()), Var("dst_devid_dummy", DataType::Handle()),
      kDLInt, 32));
}

Stmt AFunctionGenerator::Generate() {
  for (Stage s : sch->stages) {
    for (size_t i = 0; i < static_cast<size_t>(s->op->num_outputs()); ++i) {
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
    for (size_t i = 0; i < static_cast<size_t>(s->op->num_outputs()); ++i) {
      Modes layout = s->op->output_layout(i);
      if (layout.defined()) {
        // std::cout << "[AFG] Op " << s->op << std::endl;
        for (int i = layout->ndim() - 1; i >= 0; --i) {
          if (!layout->has_dependent_dims(i)) {
            continue;
          }
          set_afun(layout, i, layout->a_funs[i]);
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

UninterpFun AFunctionGenerator::set_afun(Modes layout, int idx, UninterpFun afun_shell) {
  // std::cout << "[AFG] Wanting to generate body for " << afun_shell << std::endl;
  if (afun_shell->body.defined()) {
    return afun_shell;
  }

  auto transitive_dependent_dims = layout->get_transitive_dependent_dims(idx);
  Dimension dim = layout->dimensions[idx];
  FunKey key = make_key(layout, idx);
  if (dim_afun_map.count(key)) {
    // std::cout << "[AFG]   Copying body to " << afun_shell << std::endl;
    copy_body_to_ufun_shell(dim_afun_map[key], afun_shell);
  } else {
    std::string prefix = dim->name + "_af" + std::to_string(count++) + "_";
    Var loop_var = Var(prefix + "i", DataType::Int(32));
    PrimExpr body_expr = 1;

    std::unordered_set<int> handled_already;
    PrimExpr afun_max_extent = 1;
    for (auto dependent_dim : transitive_dependent_dims) {
      int dependent_dim_idx = layout->dimensions.GetIdx(dependent_dim);
      afun_max_extent = afun_max_extent * layout->l_funs[dependent_dim_idx]->range->extent;
      if (handled_already.count(dependent_dim_idx)) {
        continue;
      }
      if (layout->has_dependent_dims(dependent_dim_idx)) {
        UninterpFun afun = set_afun(layout, dependent_dim_idx, layout->a_funs[dependent_dim_idx]);
        PrimExpr afun_call = afun.MakeCallTo({loop_var}, {dim});
        body_expr = body_expr * afun_call;

        for (auto dim : layout->get_transitive_dependent_dims(dependent_dim_idx)) {
          handled_already.insert(layout->dimensions.GetIdx(dim));
        }
      } else {
        UninterpFun l_fun = layout->l_funs[dependent_dim_idx];
        PrimExpr l_fun_call = l_fun.MakeCallTo({loop_var}, {dim});
        body_expr = body_expr * l_fun_call;
      }
    }

    PrimExpr loop_extent = layout->l_funs[idx]->range->max_inclusive();
    PrimExpr buf_extent = loop_extent + 1;
    // std::cout << "[ASDC]   Buffer range " << layout->l_funs[idx]->range << std::endl;
    Buffer afun_buffer_host =
        host_agg.create_buffer({buf_extent}, DataType::Int(32), prefix + "b_h");
    Buffer afun_buffer_dev = dev_agg.create_buffer({buf_extent}, DataType::Int(32), prefix + "b_d");
    Buffer afun_counter = decl_buffer({1}, DataType::Int(32), prefix + "ctr");

    Stmt fun_store =
        afun_buffer_host.vstore({loop_var}, afun_counter.vload({0}, DataType::Int(32)));
    Stmt counter_incr =
        afun_counter.vstore({0}, afun_counter.vload({0}, DataType::Int(32)) + body_expr);
    SeqStmt loop_stmts = SeqStmt({fun_store, counter_incr});
    Stmt stmt =
        ForNode::make(loop_var, 0, loop_extent, ForType::Serial, DeviceAPI::None, loop_stmts);

    Stmt counter_init = afun_counter.vstore({0}, 0);
    Stmt last_element =
        afun_buffer_host.vstore({loop_extent}, afun_counter.vload({0}, DataType::Int(32)));
    stmt = SeqStmt({counter_init, stmt, last_element});

    stmt =
        AttrStmtNode::make(afun_counter->data, attr::storage_scope, StringImmNode::make("global"),
                           AllocateNode::make(afun_counter->data, DataType::Int(32), {1},
                                              IntImm(DataType::Bool(1), 1), stmt));
    stmts.push_back(stmt);

    CHECK_EQ(afun_shell->parameters.size(), 1);
    Var param = afun_shell->parameters[0];
    const_cast<UninterpFunNode*>(afun_shell.as<UninterpFunNode>())
        ->SetBody(afun_buffer_dev.vload({param}, DataType::Int(32)));

    dim_afun_map[key] = afun_shell;
    // std::cout << "[AFG]   Generated body for " << afun_shell << std::endl;
  }
  return afun_shell;
}

Stmt FusionFunctionGenerator::Generate() {
  for (Stage s : sch->stages) {
    if (s->op->loop_layout().defined()) {
      for (auto lf : s->op->loop_layout()->l_funs) {
        if (lf->arity() > 0) {
          non_negative_objects.push_back(lf);
        }
      }
    }
  }

  Array<Stmt> fusion_stmts;
  for (Stage s : sch->stages) {
    for (int i = s->relations.size() - 1; i >= 0; --i) {
      if (auto frel = s->relations[i].as<RaggedFuseNode>()) {
        fusion_stmts.push_back(generate_fusion_statements(s, frel));
      }
    }
  }

  return SeqStmt(fusion_stmts);
}

PrimExpr FusionFunctionGenerator::root_ivs_fused(Stage& stage, Array<IterVar> fused_ivs) {
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
  std::unordered_set<const Object*> iter_vars_set;
  for (auto iv : iter_vars) {
    iter_vars_set.insert(iv->var.get());
  }
  for (auto var_needed : VarCollector().collect(expr)) {
    if (iter_vars_set.count(var_needed)) {
      return false;
    }
  }
  return true;
}

Stmt FusionFunctionGenerator::generate_fusion_statements(Stage& stage, const RaggedFuseNode* rel) {
  // std::cout << "[GFS] Generating fusion for " << stage << std::endl;
  CHECK(stage.is_ancestor_attached_at_root());

  IterVar outer = rel->outer;
  IterVar inner = rel->inner;
  IterVar fused = rel->fused;
  Range outer_dom = dom_map.at(outer);
  Range inner_dom = dom_map.at(inner);
  Range fused_dom = dom_map.at(fused);
  CHECK(is_zero(outer_dom->min)) << outer << " " << outer_dom;
  CHECK(is_zero(inner_dom->min)) << inner << " " << inner_dom;
  CHECK(is_zero(fused_dom->min)) << fused << " " << fused_dom;

  PrimExpr fused_var_val = root_ivs_fused(stage, {outer, inner});

  PrimExpr outer_extent_relaxed = Simplify(UninterpFun::InlineUninterpFunCalls(
      UninterpFun::RelaxUninterpCallsMaxInclusive(outer_dom->max_exclusive(), false)));
  PrimExpr inner_extent_relaxed = Simplify(UninterpFun::InlineUninterpFunCalls(
      UninterpFun::RelaxUninterpCallsMaxInclusive(inner_dom->max_exclusive(), false)));
  PrimExpr fused_extent_relaxed = outer_extent_relaxed * inner_extent_relaxed;

  // std::cout << "[GFS]   Outer " << outer_dom << " " << outer_extent_relaxed << std::endl;
  // std::cout << "[GFS]   Inner " << inner_dom << " " << inner_extent_relaxed << std::endl;
  // std::cout << "[GFS]   Fused " << fused_dom << " " << fused_extent_relaxed << std::endl;

  auto decl_both_buffers = [&](Array<PrimExpr> shape, std::string prefix) {
    prefix = prefix + std::to_string(count);
    Buffer host_buffer = host_agg.create_buffer(shape, DataType::Int(32), prefix + "_h");
    Buffer dev_buffer = dev_agg.create_buffer(shape, DataType::Int(32), prefix + "_d");
    return std::make_pair(host_buffer, dev_buffer);
  };

  // Allocate buffers
  auto fused_to_inner_bufs = decl_both_buffers({fused_extent_relaxed}, "fi");
  auto fused_to_outer_bufs = decl_both_buffers({fused_extent_relaxed}, "fo");
  auto outer_to_fused_pos_bufs = decl_both_buffers({outer_extent_relaxed}, "ofp");
  Buffer fused_val = decl_buffer({1}, DataType::Int(32), "f" + std::to_string(count));
  count++;

  PrimExpr outer_loop_extent = outer_dom->extent;
  PrimExpr inner_loop_extent_unreplaced = inner_dom->extent;
  PrimExpr inner_loop_extent = inner_loop_extent_unreplaced;
  // CHECK(is_constant(outer_loop_extent, stage->all_iter_vars)) << " " << outer_loop_extent;
  if (!is_constant(outer_loop_extent, stage->all_iter_vars)) {
    outer_loop_extent = outer_extent_relaxed;
    std::cerr << "[WARNING] Using relaxed outer loop extent for generating fusion loops as the "
                 "actual extent is not a constant"
              << std::endl;
  }
  {
    std::unordered_map<IterVar, int> outer_descendant_state;
    outer_descendant_state[outer] = 1;
    PassDownBitMaskOr(stage, &outer_descendant_state, true);

    std::unordered_map<IterVar, PrimExpr> state;
    for (auto lv : stage->leaf_iter_vars) {
      if (outer_descendant_state.count(lv) && outer_descendant_state.at(lv)) continue;
      // std::cout << "[GFS]    LF " << lv << std::endl;
      state[lv] = lv->var;
    }
    state[outer] = outer->var;
    PassUpIndex(stage, dom_map, &state, true);
    std::unordered_map<const VarNode*, PrimExpr> vsub;
    for (auto it : state) {
      if (it.first == outer) continue;
      // std::cout << "[GFS]    Replace " << it.first->var << " " << it.second << std::endl;
      // std::cout << "[GFS]    Replace " << it.first->var << " " << it.first.get() << " " << outer
      // << " " << outer.get() << std::endl;
      vsub[it.first->var.as<VarNode>()] = it.second;
    }
    inner_loop_extent = VarReplacer(vsub)(inner_loop_extent);
    // std::cout << "[GFS]       Replaced " << inner_loop_extent_unreplaced << " " <<
    // inner_loop_extent
    // << std::endl;

    {
      arith::Analyzer analyzer;
      for (auto iv : stage->all_iter_vars) {
        analyzer.Bind(iv->var, dom_map.at(iv));
      }
      // std::cout << "[GFS]       Simplified "
      // << analyzer.Simplify(UninterpFun::InlineUninterpFunCalls(inner_loop_extent))
      // << std::endl;
      inner_loop_extent = analyzer.Simplify(UninterpFun::InlineUninterpFunCalls(inner_loop_extent));
    }

    // std::unordered_map<IterVar, PrimExpr> state;
    // state[outer] = outer->var;
    // PassUpIndex(stage, dom_map, &state, true);
    // std::unordered_map<const VarNode*, PrimExpr> vsub;
    // for (auto it : state) {
    //   std::cout << "[GFS]    Replace " << it.first->var << " " << it.second << std::endl;
    //   vsub[it.first->var.as<VarNode>()] = it.second;
    // }
    // inner_loop_extent = VarReplacer(vsub)(inner_loop_extent);
    // std::cout << "[GFS]       Replaced " << inner_loop_extent_unreplaced << " " <<
    // inner_loop_extent
    //           << std::endl;
  }

  // Compute the outer and inner variables in terms of the root itervars
  PrimExpr outer_value = outer->var;
  PrimExpr inner_value = inner->var;

  Stmt no_op = EvaluateNode::make(0);
  Stmt body = NullValue<Stmt>();
  PrimExpr fused_val_load = fused_val.vload({0}, DataType::Int(32));
  {
    Stmt outer_store = fused_to_outer_bufs.first.vstore({fused_val_load}, outer_value);
    Stmt inner_store = fused_to_inner_bufs.first.vstore({fused_val_load}, inner_value);
    Stmt fused_incr = fused_val.vstore({0}, fused_val_load + 1);
    body = SeqStmt({outer_store, inner_store, fused_incr});
  }

  body = ForNode::make(inner->var, 0, inner_loop_extent, ForType::Serial, DeviceAPI::None, body);
  if (!fused_var_val.defined()) {
    body = SeqStmt({outer_to_fused_pos_bufs.first.vstore({outer_value}, fused_val_load), body});
  }
  body = ForNode::make(outer->var, 0, outer_loop_extent, ForType::Serial, DeviceAPI::None, body);

  // Add annotations stating that the buffers we create all contain
  // non-negative integers
  non_negative_objects.push_back(fused_to_outer_bufs.second->data);
  non_negative_objects.push_back(fused_to_inner_bufs.second->data);
  non_negative_objects.push_back(outer_to_fused_pos_bufs.second->data);

  body = SeqStmt({fused_val.vstore({0}, 0), body});

  // body = allocate_both_bufs(fused_to_inner_bufs, {fused_extent_relaxed}, body);
  // body = allocate_both_bufs(fused_to_outer_bufs, {fused_extent_relaxed}, body);
  // body = allocate_both_bufs(outer_to_fused_pos_bufs, {outer_extent_relaxed}, body);
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

  init_uf(rel->fused_to_outer_uf, outer_extent_relaxed, fused_to_outer_bufs.second);
  init_uf(rel->fused_to_inner_uf, inner_extent_relaxed, fused_to_inner_bufs.second);
  auto oif_body = outer_to_fused_pos_bufs.second.vload(
                      {rel->outer_inner_to_fused_uf->parameters[0]}, DataType::Int(32)) +
                  rel->outer_inner_to_fused_uf->parameters[1];
  init_uf(rel->outer_inner_to_fused_uf, fused_extent_relaxed, outer_to_fused_pos_bufs.second,
          oif_body);
  return body;
}

Array<PrimExpr> FusionFunctionGenerator::get_iter_var_values(Array<IterVar> vars, Stage& stage) {
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

void FunctionGenerator::GenerateAFunctions() {
  AFunctionGenerator generator(sch, &buffer_map, &host_agg, &dev_agg);
  afun_stmt = generator.Generate();
}

void FunctionGenerator::GenerateFusionFunctions() {
  FusionFunctionGenerator generator(sch, dom_map, &non_negative_objects, &buffer_map, &host_agg,
                                    &dev_agg);
  ffun_stmt = generator.Generate();
}

Stmt FunctionGenerator::CreateBody(Stmt body) {
  for (ObjectRef obj : non_negative_objects) {
    body = AttrStmtNode::make(obj, attr::non_negative_annotation, 0, body);
  }

  auto dev_agg_buf = dev_agg.aggregate_buffer();
  auto host_agg_buf = host_agg.aggregate_buffer();
  buffer_map.Set(host_agg_buf, dev_agg_buf);
  CHECK(is_zero(Simplify(host_agg.aggregate_size() - dev_agg.aggregate_size())));
  Stmt copy_stmt = copy_bufs(std::make_pair(host_agg_buf, dev_agg_buf), dev_agg.aggregate_size(),
                             DataType::Int(32));

  // Stmt copy_stmt = EvaluateNode::make(0);

  Stmt prep_code_body = SeqStmt({ffun_stmt, afun_stmt, copy_stmt});
  Stmt prep_code = AttrStmtNode::make(buffer_map, attr::prep_code_scope, 0, prep_code_body);
  return SeqStmt({prep_code, body});
}

}  // namespace te
}  // namespace tvm
