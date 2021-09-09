#include "function_generator.h"

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "../../tir/ir/var_replacer.h"
#include "../../tir/pass/ir_util.h"
#include "message_passing.h"

namespace tvm {
namespace te {

Buffer AllocationAggregator::create_buffer(Array<PrimExpr> extents, DataType buf_dtype,
                                           std::string name) {
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
    // std::cout << "[ADep " << dim << " " << l_fun << std::endl;
    transitive_dependent_dims_set.insert(l_fun.get());
  }
  return {layout->dimensions[idx], transitive_dependent_dims_set};
}

Stmt AFunctionGenerator::Generate() {
  auto lambda1 = [this](Modes layout) {
    if (layout.defined()) {
      for (size_t i = 0; i < layout->ndim(); ++i) {
        if (layout->a_funs[i].defined() && layout->a_funs[i]->body.defined()) {
          FunKey key = make_key(layout, i);
          this->dim_afun_map[key] = layout->a_funs[i];
        }
      }
    }
  };
  for (Stage s : sch->stages) {
    for (size_t i = 0; i < static_cast<size_t>(s->op->num_outputs()); ++i) {
      lambda1(s->op->output_layout(i));
    }
  }
  for (Buffer b : afuns_needed_for) {
    if (b->shape->is_ragged()) {
      lambda1(b->shape);
    }
  }

  auto lambda2 = [this](Modes layout) {
    if (layout.defined()) {
      // std::cout << "[AFG] Op " << s->op << std::endl;
      for (int i = layout->ndim() - 1; i >= 0; --i) {
        if (!layout->has_dependent_dims(i)) {
          continue;
        }
        // std::cout << "[FG] DimAFunc " << s << std::endl;
        this->set_afun(layout, i, layout->a_funs[i]);
      }
    }
  };

  for (Stage s : sch->stages) {
    for (size_t i = 0; i < static_cast<size_t>(s->op->num_outputs()); ++i) {
      lambda2(s->op->output_layout(i));
    }
  }

  for (Buffer b : afuns_needed_for) {
    if (b->shape->is_ragged()) {
      lambda2(b->shape);
    }
  }

  return SeqStmt(stmts);
}

void copy_body_to_ufun_shell(UninterpFun fun, UninterpFun shell) {
  // std::cout << "[FG] Setting body for " << fun << std::endl;
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
  // std::cout << "[AFG] Wanting to generate body for " << afun_shell << " " <<
  // layout->dimensions[idx]
  // << std::endl;
  if (afun_shell->body.defined()) {
    return afun_shell;
  }

  Dimension dim = layout->dimensions[idx];
  FunKey key = make_key(layout, idx);
  if (dim_afun_map.count(key)) {
    // std::cout << "[AFG]   Copying body to " << afun_shell << std::endl;
    if (debug_fill_function_bodies) {
      copy_body_to_ufun_shell(dim_afun_map[key], afun_shell);
    }
  } else {
    std::string prefix = dim->name + "_af" + std::to_string(count++) + "_";
    Var loop_var = Var(prefix + "i", DataType::Int(32));
    PrimExpr body_expr = 1;

    PrimExpr afun_max_extent = 1;
    for (auto dependent_dim : layout->get_immediate_dependent_dims(idx)) {
      int dependent_dim_idx = layout->dimensions.GetIdx(dependent_dim);
      afun_max_extent = afun_max_extent * layout->l_funs[dependent_dim_idx]->range->extent;
      UninterpFun l_fun = layout->l_funs[dependent_dim_idx];
      PrimExpr l_fun_call = l_fun.MakeCallTo(Array<PrimExpr>({loop_var}), {dim});
      if (layout->has_dependent_dims(dependent_dim_idx)) {
        UninterpFun afun = set_afun(layout, dependent_dim_idx, layout->a_funs[dependent_dim_idx]);
        PrimExpr afun_call = afun.MakeCallTo(Array<PrimExpr>({l_fun_call}), {dependent_dim});
        body_expr = body_expr * afun_call;
      } else {
        body_expr = body_expr * l_fun_call;
      }
    }

    PrimExpr loop_extent = layout->l_funs[idx]->range->max_inclusive();
    PrimExpr buf_extent = loop_extent + 1;
    // std::cout << "[ASDC]   Buffer range " << layout->l_funs[idx]->range << std::endl;
    auto buffer_pair = agg_pair.create_buffer_pair({buf_extent}, DataType::Int(32), prefix);
    Buffer afun_buffer_host = buffer_pair.first;
    Buffer afun_buffer_dev = buffer_pair.second;
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
    if (debug_fill_function_bodies) {
      // std::cout << "[FG] Setting body for " << afun_shell << std::endl;
      const_cast<UninterpFunNode*>(afun_shell.as<UninterpFunNode>())
          ->SetBody(afun_buffer_dev.vload({param}, DataType::Int(32)));
    }

    dim_afun_map[key] = afun_shell;
    // std::cout << "[AFG]   Generated body for " << afun_shell << std::endl;
  }
  return afun_shell;
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
  for (Stage s : stages_to_generate_for) {
    // std::cout << "[FG] LoopFusionFunc for " << s << std::endl;
    // for (int i = s->relations.size() - 1; i >= 0; --i) {
    for (size_t i = 0; i < s->relations.size(); ++i) {
      if (auto frel = s->relations[i].as<RaggedFuseNode>()) {
        auto body = generate_fusion_statements(s, frel);
        // std::cout << "[FG]  Body " << body << std::endl;
        fusion_stmts.push_back(body);
      }
    }

    // std::cout << "[FG] StorageFusionFunc for " << s << std::endl;
    for (auto rel : s->dim_relation_graph->relations) {
      if (auto frel = rel.as<RaggedDimensionFuseNode>()) {
        // std::cout << "[FG] Need FusionFunc for " << s << " " << frel->outer_inner_to_fused_uf
        // << std::endl;
        fusion_stmts.push_back(generate_fusion_statements(s, frel));
      }
    }
  }

  return SeqStmt(fusion_stmts);
}

Stmt FusionFunctionGenerator::generate_fusion_statements(Stage& stage, const RaggedFuseNode* rel) {
  // std::cout << "[GFS] Generating fusion for " << stage << " " << rel << std::endl;
  // CHECK(stage.is_ancestor_attached_at_root());

  IterVar outer = rel->outer;
  IterVar inner = rel->inner;
  IterVar fused = rel->fused;
  Range outer_dom = dom_map.at(outer);
  Range inner_dom = dom_map.at(inner);
  Range fused_dom = dom_map.at(fused);
  // CHECK(is_zero(outer_dom->min)) << outer << " " << outer_dom;
  // CHECK(is_zero(inner_dom->min)) << inner << " " << inner_dom;
  // CHECK(is_zero(fused_dom->min)) << fused << " " << fused_dom;

  PrimExpr outer_extent_relaxed = Simplify(UninterpFun::InlineUninterpFunCalls(
      UninterpFun::RelaxUninterpCallsMaxInclusive(outer_dom->max_exclusive(), false)));
  PrimExpr inner_extent_relaxed = Simplify(UninterpFun::InlineUninterpFunCalls(
      UninterpFun::RelaxUninterpCallsMaxInclusive(inner_dom->max_exclusive(), false)));
  PrimExpr fused_extent_relaxed = outer_extent_relaxed * inner_extent_relaxed;

  // std::cout << "[GFS]   Outer " << outer->var << " " << outer_dom << " " << outer_extent_relaxed
  //           << std::endl;
  // std::cout << "[GFS]   Inner " << inner->var << " " << inner_dom << " " << inner_extent_relaxed
  //           << std::endl;
  // std::cout << "[GFS]   Fused " << fused->var << " " << fused_dom << " " << fused_extent_relaxed
  //           << std::endl;

  auto decl_both_buffers = [&](Array<PrimExpr> shape, std::string prefix) {
    prefix = prefix + std::to_string(count);
    auto buffer_pair = agg_pair.create_buffer_pair(shape, DataType::Int(32), prefix);
    Buffer host_buffer = buffer_pair.first;
    Buffer dev_buffer = buffer_pair.second;
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
      inner_loop_extent = analyzer.Simplify(UninterpFun::InlineUninterpFunCalls(inner_loop_extent));
    }
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

  body = ForNode::make(inner->var, inner_dom->min, inner_loop_extent, ForType::Serial,
                       DeviceAPI::None, body);
  body = SeqStmt(
      {outer_to_fused_pos_bufs.first.vstore({outer_value - outer_dom->min}, fused_val_load), body});
  body = ForNode::make(outer->var, outer_dom->min, outer_loop_extent, ForType::Serial,
                       DeviceAPI::None, body);

  // Add annotations stating that the buffers we create all contain
  // non-negative integers
  non_negative_objects.push_back(fused_to_outer_bufs.second->data);
  non_negative_objects.push_back(fused_to_inner_bufs.second->data);
  non_negative_objects.push_back(outer_to_fused_pos_bufs.second->data);

  body = SeqStmt({fused_val.vstore({0}, 0), body});

  body = AttrStmtNode::make(fused_val->data, attr::storage_scope, StringImmNode::make("global"),
                            AllocateNode::make(fused_val->data, DataType::Int(32), {1},
                                               IntImm(DataType::Bool(1), 1), body));

  PrimExpr fused_min = VarReplacer({{outer->var.get(), outer_dom->min}})(inner_dom->min);
  auto init_uf = [&](UninterpFun uf, PrimExpr max_extent, Buffer loadee,
                     PrimExpr body = NullValue<PrimExpr>()) {
    UninterpFunNode* uf_node = const_cast<UninterpFunNode*>(uf.as<UninterpFunNode>());

    if (!body.defined()) {
      CHECK_EQ(uf->arity(), 1);
      Array<PrimExpr> extents;
      for (auto param : uf->parameters) extents.push_back(param - fused_min);
      body = loadee.vload(extents, DataType::Int(32));
    }

    // std::cout << "[FPL]   Setting body " << uf->func_name() << " " << body << std::endl;
    if (debug_fill_function_bodies) {
      uf_node->SetBody(body);
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

Stmt FusionFunctionGenerator::generate_fusion_statements(Stage& stage,
                                                         const RaggedDimensionFuseNode* rel) {
  // std::cout << "[GFS] Generating dim fusion for " << stage << std::endl;
  // CHECK(stage.is_ancestor_attached_at_root()) << stage;

  auto layout = root_layout_map.at(stage);
  CHECK(layout->dimensions.Contains(rel->outer)) << "Only root dimension fusion allowed for now";
  CHECK(layout->dimensions.Contains(rel->inner)) << "Only root dimension fusion allowed for now";
  std::unordered_map<const DimensionNode*, Range> pdd_state;
  for (size_t i = 0; i < layout->dimensions.size(); ++i) {
    Range r = Range::make_by_min_max_exclusive(0, layout->l_funs[i]->range->max_inclusive());
    // std::cout << "[GFS]  Root Extent: " << layout->dimensions[i] << " " << r << std::endl;
    pdd_state[layout->dimensions[i].operator->()] = r;
  }
  DimensionPassDownDomain(stage, stage->op.as<BaseVarDimOpNode>(), &pdd_state, true);

  PrimExpr outer_extent = pdd_state[rel->outer.operator->()]->extent;
  PrimExpr inner_extent = pdd_state[rel->inner.operator->()]->extent;
  PrimExpr fused_extent = outer_extent * inner_extent;

  // std::cout << "[GFS]  Extents: " << outer_extent << std::endl;
  // std::cout << "[GFS]           " << inner_extent << std::endl;
  // std::cout << "[GFS]           " << fused_extent << std::endl;

  auto decl_both_buffers = [&](Array<PrimExpr> shape, std::string prefix) {
    prefix = "d_" + prefix + std::to_string(count);
    auto buffer_pair = agg_pair.create_buffer_pair(shape, DataType::Int(32), prefix);
    Buffer host_buffer = buffer_pair.first;
    Buffer dev_buffer = buffer_pair.second;
    return std::make_pair(host_buffer, dev_buffer);
  };

  // Allocate buffers
  auto fused_to_inner_bufs = decl_both_buffers({fused_extent}, "fi");
  auto fused_to_outer_bufs = decl_both_buffers({fused_extent}, "fo");
  auto outer_to_fused_pos_bufs = decl_both_buffers({outer_extent}, "ofp");
  Buffer fused_val = decl_buffer({1}, DataType::Int(32), "fb" + std::to_string(count));
  count++;

  CHECK(is_constant(outer_extent, stage->all_iter_vars));

  // Compute the outer and inner variables in terms of the root itervars
  Var outer_loop_var = Var("out", DataType::Int(32));
  Var inner_loop_var = Var("in", DataType::Int(32));

  Stmt no_op = EvaluateNode::make(0);
  Stmt body = NullValue<Stmt>();
  PrimExpr fused_val_load = fused_val.vload({0}, DataType::Int(32));
  {
    Stmt outer_store = fused_to_outer_bufs.first.vstore({fused_val_load}, outer_loop_var);
    Stmt inner_store = fused_to_inner_bufs.first.vstore({fused_val_load}, inner_loop_var);
    Stmt fused_incr = fused_val.vstore({0}, fused_val_load + 1);
    body = SeqStmt({outer_store, inner_store, fused_incr});
  }

  // std::cout << "[GFS]  LFun: " << layout->l_funs[layout->dimensions.GetIdx(rel->inner)]
  //           << std::endl;
  body = ForNode::make(inner_loop_var, 0,
                       layout->l_funs[layout->dimensions.GetIdx(rel->inner)].MakeCallTo(
                           Array<Var>({outer_loop_var}), {rel->outer}),
                       ForType::Serial, DeviceAPI::None, body);
  body = SeqStmt({outer_to_fused_pos_bufs.first.vstore({outer_loop_var}, fused_val_load), body});
  body = ForNode::make(outer_loop_var, 0, outer_extent, ForType::Serial, DeviceAPI::None, body);

  // Add annotations stating that the buffers we create all contain
  // non-negative integers
  non_negative_objects.push_back(fused_to_outer_bufs.second->data);
  non_negative_objects.push_back(fused_to_inner_bufs.second->data);
  non_negative_objects.push_back(outer_to_fused_pos_bufs.second->data);

  body = SeqStmt({fused_val.vstore({0}, 0), body});

  body = AttrStmtNode::make(fused_val->data, attr::storage_scope, StringImmNode::make("global"),
                            AllocateNode::make(fused_val->data, DataType::Int(32), {1},
                                               IntImm(DataType::Bool(1), 1), body));

  auto init_uf = [&](UninterpFun uf, PrimExpr max_extent, Buffer loadee,
                     PrimExpr body = NullValue<PrimExpr>()) {
    UninterpFunNode* uf_node = const_cast<UninterpFunNode*>(uf.as<UninterpFunNode>());
    Array<PrimExpr> extents;
    for (auto param : uf->parameters) extents.push_back(param);

    if (debug_fill_function_bodies) {
      if (body.defined()) {
        uf_node->SetBody(body);
        // std::cout << "[FG]   Custom body " << uf << std::endl;
      } else {
        uf_node->SetBody(loadee.vload(extents, DataType::Int(32)));
        // std::cout << "[FG]   Loadee body " << uf << std::endl;
      }
    }
    uf_node->SetRange(Range::make_by_min_extent(0, max_extent));
  };

  init_uf(rel->fused_to_outer_uf, outer_extent, fused_to_outer_bufs.second);
  init_uf(rel->fused_to_inner_uf, inner_extent, fused_to_inner_bufs.second);
  auto oif_body = outer_to_fused_pos_bufs.second.vload(
                      {rel->outer_inner_to_fused_uf->parameters[0]}, DataType::Int(32)) +
                  rel->outer_inner_to_fused_uf->parameters[1];
  init_uf(rel->outer_inner_to_fused_uf, fused_extent, outer_to_fused_pos_bufs.second, oif_body);
  return body;
}

std::pair<Buffer, Buffer> AggregatorPair::create_buffer_pair(Array<PrimExpr> extents,
                                                             DataType buf_dtype, std::string name) {
  if (distinct_device) {
    return std::make_pair(host_agg.create_buffer(extents, buf_dtype, name + "_h"),
                          dev_agg.create_buffer(extents, buf_dtype, name + "_d"));
  } else {
    Buffer buffer = host_agg.create_buffer(extents, buf_dtype, name);
    return std::make_pair(buffer, buffer);
  }
}

std::pair<Buffer, Buffer> AggregatorPair::aggregate_buffers() {
  if (distinct_device) {
    CHECK(is_zero(Simplify(host_agg.aggregate_size() - dev_agg.aggregate_size())));
    return std::make_pair(host_agg.aggregate_buffer(), dev_agg.aggregate_buffer());
  } else {
    Buffer buffer = host_agg.aggregate_buffer();
    return std::make_pair(buffer, buffer);
  }
}

Stmt FusionFunctionSimplifier::Simplify(Stmt body,
                                        std::vector<Stage>& stages_to_generate_fusion_funcs_for) {
  struct FuncTriple {
    UninterpFun fused_to_outer_uf;
    UninterpFun fused_to_inner_uf;
    UninterpFun outer_inner_to_fused_uf;
  };

  std::unordered_map<const Object*, FuncTriple> fused_fun_map;

  auto handle_rel = [&](Dimension fused_dim, UninterpFun fused_to_outer_uf,
                        UninterpFun fused_to_inner_uf, UninterpFun outer_inner_to_fused_uf) {
    auto it = fused_fun_map.find(fused_dim.get());
    if (it != fused_fun_map.end()) {
      auto& funs = it->second;
      fsub[fused_to_outer_uf.get()] = funs.fused_to_outer_uf;
      fsub[fused_to_inner_uf.get()] = funs.fused_to_inner_uf;
      fsub[outer_inner_to_fused_uf.get()] = funs.outer_inner_to_fused_uf;
      std::cout << "[FSD_SIMPL] FSub: " << funs.outer_inner_to_fused_uf << " "
                << outer_inner_to_fused_uf << std::endl;
      return false;
    } else {
      fused_fun_map[fused_dim.get()] = {fused_to_outer_uf, fused_to_inner_uf,
                                        outer_inner_to_fused_uf};
      return true;
    }
  };

  // Process stages in sorted order so that global stages are
  // processed before shared stages which in turn are processed before
  // local stages. This ensures that the most exhaustive fusion
  // function is selected for generation in case stages can share
  // fusion functions

  std::vector<Stage> stages;
  for (auto s : sch->stages) {
    stages.push_back(s);
  }
  struct less_than_stage {
    inline bool operator()(const Stage& stage1, const Stage& stage2) {
      auto r1 = (stage1->scope.length() == 0) ? runtime::StorageRank::kGlobal
                                              : runtime::StorageScope::make(stage1->scope).rank;
      auto r2 = (stage2->scope.length() == 0) ? runtime::StorageRank::kGlobal
                                              : runtime::StorageScope::make(stage2->scope).rank;
      return (r1 < r2);
    }
  };
  std::sort(stages.begin(), stages.end(), less_than_stage());

  for (auto s : stages) {
    bool to_add = false;
    for (auto rel : s->relations) {
      if (auto frel = rel.as<RaggedFuseNode>()) {
        to_add |= handle_rel(frel->fused_to_outer_uf->dimensions[0], frel->fused_to_outer_uf,
                             frel->fused_to_inner_uf, frel->outer_inner_to_fused_uf);
      }
    }

    if (s->dim_relation_graph.defined()) {
      for (auto rel : s->dim_relation_graph->relations) {
        if (auto frel = rel.as<RaggedDimensionFuseNode>()) {
          to_add |= handle_rel(frel->fused_to_outer_uf->dimensions[0], frel->fused_to_outer_uf,
                               frel->fused_to_inner_uf, frel->outer_inner_to_fused_uf);
        }
      }
    }
    if (to_add) {
      stages_to_generate_fusion_funcs_for.push_back(s);
    }
  }
  body = this->VisitStmt(body);
  body = tir::Simplify(body);
  return body;
}

PrimExpr FusionFunctionSimplifier::VisitExpr_(const CallNode* op) {
  // std::cout << "[FG]   CallEXpre " << GetRef<PrimExpr>(op) << " " << op->func << std::endl;
  auto it = fsub.find(op->func.get());
  if (it != fsub.end()) {
    // std::cout << "[FG]     Found " << it->second << std::endl;
    Array<PrimExpr> new_args;
    for (auto arg : op->args) {
      new_args.push_back(this->VisitExpr(arg));
    }
    Array<Range> new_realize_bounds;
    for (auto r : op->custom_realize_bounds) {
      new_realize_bounds.push_back(
          Range::make_by_min_extent(this->VisitExpr(r->min), this->VisitExpr(r->extent)));
    }
    auto ret = CallNode::make(op->dtype, it->second->fname, new_args, op->call_type, op->arg_dims,
                              it->second, op->value_index, new_realize_bounds);
    // std::cout << "[FG]       Ret " << ret << std::endl;
    return ret;
  } else {
    return StmtExprMutator::VisitExpr_(op);
  }
}

PrimExpr FusionFunctionSimplifier::VisitExpr_(const FuseSelectNode* op) {
  return SelectNode::make(this->VisitExpr(op->condition), this->VisitExpr(op->true_value),
                          this->VisitExpr(op->false_value));
}

Stmt FunctionGenerator::SimplifyFusionFunctions(Stmt body) {
  FusionFunctionSimplifier simplifier(sch, dom_map);
  return simplifier.Simplify(body, stages_to_generate_fusion_funcs_for);
}

void FunctionGenerator::GenerateAFunctions() {
  AFunctionGenerator generator(sch, &buffer_map, &agg_pair, debug_fill_function_bodies,
                               afuns_needed_for);
  afun_stmt = generator.Generate();
  // std::cout << "[AFUNSTMT]\n " << afun_stmt << std::endl;
  // exit(0);
}

void FunctionGenerator::GenerateFusionFunctions() {
  FusionFunctionGenerator generator(sch, dom_map, root_layout_map,
                                    stages_to_generate_fusion_funcs_for, &non_negative_objects,
                                    &buffer_map, &agg_pair, debug_fill_function_bodies);
  // std::cout << "[MAPMAP11] " << generator.root_layout_map.defined() << std::endl;
  // std::cout << "[MAPMAP12] " << generator.root_layout_map.size() << std::endl;
  ffun_stmt = generator.Generate();
}

Stmt FunctionGenerator::CreateBody(Stmt body) {
  std::unordered_set<const Object*> processed;
  for (ObjectRef obj : non_negative_objects) {
    if (processed.count(obj.get())) {
      continue;
    } else {
      body = AttrStmtNode::make(obj, attr::aux_data_structure, 0, body);
      processed.insert(obj.get());
    }
  }

  auto agg_buf_pair = agg_pair.aggregate_buffers();
  auto host_agg_buf = agg_buf_pair.first;
  auto dev_agg_buf = agg_buf_pair.second;
  buffer_map.Set(host_agg_buf, dev_agg_buf);
  Stmt prep_code_body;
  if (is_zero(agg_pair.aggregate_size()) || dev_agg_buf == host_agg_buf) {
    prep_code_body = SeqStmt({ffun_stmt, afun_stmt});
  } else {
    Stmt copy_stmt = EvaluateNode::make(copy_to_device(
        host_agg_buf->data, 0, dev_agg_buf->data, 0,
        agg_pair.aggregate_size() * DataType::Int(32).bytes(),
        Var("src_devtype_dummy", DataType::Handle()), Var("src_devid_dummy", DataType::Handle()),
        Var("dst_devtype_dummy", DataType::Handle()), Var("dst_devid_dummy", DataType::Handle()),
        kDLInt, 32));
    prep_code_body = SeqStmt({ffun_stmt, afun_stmt, copy_stmt});
  }
  Stmt prep_code = AttrStmtNode::make(buffer_map, attr::prep_code_scope, 0, prep_code_body);
  // std::cout << "[PREPSTMT]\n " << prep_code << std::endl;
  // exit(0);
  return SeqStmt({prep_code, body});
}

}  // namespace te
}  // namespace tvm
