/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Utility to make loop nest.
 * \file op_util.cc
 */
#include "op_util.h"

#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <string>

#include "../../arith/compute_expr.h"
#include "../../tir/ir/var_replacer.h"
#include "../schedule/message_passing.h"

namespace tvm {
namespace te {

using namespace arith;
using namespace tir;

Map<IterVar, Range> RelaxOutOfOrderLoopBounds(const Stage& stage,
                                              const std::unordered_map<IterVar, Range>& dom_map) {
  using VarNodeSet = std::unordered_set<const VarNode*>;
  using IterVarNodeSet = std::unordered_set<const IterVarNode*>;

  IterVarNodeSet prefix_vars;
  std::unordered_map<IterVar, int> to_relax_state;
  Array<IterVar> to_relax_leaf_vars;
  for (auto lv : stage->leaf_iter_vars) {
    VarNodeSet root_vars_needed =
        VarCollector().collect(UninterpFun::InlineUninterpFunCalls(dom_map.at(lv)->extent));
    std::unordered_map<IterVar, int> state;
    for (auto rv : stage->all_iter_vars) {
      if (root_vars_needed.find(rv->var.as<VarNode>()) != root_vars_needed.end()) {
        state[rv] = 1;
      }
    }
    PassDownBitMaskOr(stage, &state, true);

    for (auto lv2 : stage->leaf_iter_vars) {
      if (state[lv2] && (prefix_vars.find(lv2.as<IterVarNode>()) == prefix_vars.end())) {
        to_relax_state[lv] = 1;
        to_relax_leaf_vars.push_back(lv);
        break;
      }
    }
    prefix_vars.insert(lv.as<IterVarNode>());
  }

  PassUpBitMaskOr(stage, &to_relax_state, true);

  std::unordered_map<IterVar, Range> relaxed_dom_map;
  Analyzer analyzer;
  for (auto rv : stage->op->root_iter_vars()) {
    Range range = dom_map.at(rv);
    Range relaxed_range = range;
    if (to_relax_state[rv]) {
      if (auto call = range->extent.as<CallNode>()) {
        if (auto ufun = call->func.as<UninterpFunNode>()) {
          relaxed_range = ufun->range;
        }
      }
    }
    relaxed_dom_map[rv] = relaxed_range;
    analyzer.Bind(rv->var, relaxed_range);
  }

  PassDownDomain(stage, &relaxed_dom_map, &analyzer, true);

  Map<IterVar, Range> ret;
  for (auto lv : to_relax_leaf_vars) {
    ret.Set(lv, relaxed_dom_map.at(lv));
  }

  return ret;
}

std::vector<int> OrderIndexVariables(Array<UninterpFun> index_expressions,
                                     Array<Dimension> index_dimensions,
                                     Array<Dimension> loop_dimensions) {
  std::unordered_set<const Object*> ordered;
  std::vector<int> order;

  for (const auto& dim : loop_dimensions) {
    ordered.insert(dim.get());
  }

  for (int k = 0; k < 100; ++k) {
    if (order.size() == index_dimensions.size()) break;
    for (size_t i = 0; i < index_dimensions.size(); ++i) {
      Dimension dim = index_dimensions[i];
      if (ordered.count(dim.get())) continue;

      bool preds_ordered = true;
      Array<Dimension> this_preds = index_expressions[i]->dimensions;
      for (const auto& pred : this_preds) {
        if (!ordered.count(pred.get())) {
          preds_ordered = false;
          break;
        }
      }

      if (preds_ordered) {
        ordered.insert(dim.get());
        order.push_back(i);
      }
    }
  }

  CHECK_EQ(order.size(), index_dimensions.size()) << "Cyclic dependence amongst index dimensions";

  return order;
}

void IndexLoopVarDeps(const Stage& stage, Array<DimInfo> all_dimensions,
                      const std::unordered_map<IterVar, Range>& dom_map,
                      Map<Var, Array<Var>>& index_vars_loop_vars_depend_on,
                      Map<Var, Array<DimInfo>>& index_vars_loop_vars_are_needed_for,
                      std::unordered_map<const VarNode*, int>& index_vars_dep_count) {
  bool print = false;  //(stage->op->name == "i_c_sum.d");
  if (print) std::cout << "[ILVD] Op " << stage->op << std::endl;
  auto var_dim_op = stage->op.as<BaseVarDimOpNode>();
  CHECK(var_dim_op);
  std::unordered_map<const VarNode*, const DimInfoNode*> index_vars;
  std::unordered_set<const VarNode*> root_vars;
  for (const auto di : all_dimensions) {
    if (di->dim->isFunDim()) {
      index_vars[di->iv->var.as<VarNode>()] = di.as<DimInfoNode>();
    } else {
      root_vars.insert(di->iv->var.as<VarNode>());
    }
  }

  for (const auto lv : stage->leaf_iter_vars) {
    Array<Var> dep_idx_vars;
    PrimExpr extent = dom_map.at(lv)->extent;
    auto input_vars = VarCollector().collect(extent);
    for (auto inp : input_vars) {
      if (index_vars.count(inp)) dep_idx_vars.push_back(GetRef<Var>(inp));
    }
    index_vars_loop_vars_depend_on.Set(lv->var, dep_idx_vars);
  }

  for (const auto di : all_dimensions) {
    if (print) std::cout << "[ILVD]  Dim " << di->dim << std::endl;
    if (di->dim->isLoopDim()) {
    } else {
      if (print) std::cout << "[ILVD]   Var " << di->iv << std::endl;
      std::unordered_map<IterVar, int> state;
      for (auto dim : di->ufun->dimensions) {
        if (print) std::cout << "[ILVD]     DepDim " << dim << std::endl;
        state[var_dim_op->GetIterVarFromDim(0, dim)] = 1;
      }
      PassDownBitMaskOr(stage, &state, true);
      int dep_count = 0;
      for (auto it : state) {
        if (!root_vars.count(it.first->var.as<VarNode>()) && it.second == 1) {
          dep_count++;
          if (index_vars_loop_vars_are_needed_for.count(it.first->var)) {
            auto idx_vars = index_vars_loop_vars_are_needed_for.at(it.first->var);
            idx_vars.push_back(di);
            index_vars_loop_vars_are_needed_for.Set(it.first->var, idx_vars);
          } else {
            index_vars_loop_vars_are_needed_for.Set(it.first->var, {di});
          }
        }
      }
      index_vars_dep_count[di->iv->var.as<VarNode>()] = dep_count;
    }
  }
}

Map<IterVar, Array<IterVar>> RootToLeafVarMapping(const Stage& stage) {
  Map<IterVar, Array<IterVar>> mapping;
  for (const auto& iv : stage->op->root_iter_vars()) {
    std::unordered_map<IterVar, int> state;
    state[iv] = 1;
    PassDownBitMaskOr(stage, &state, true);
    Array<IterVar> leaf_vars;
    for (const auto lv : stage->leaf_iter_vars) {
      if (state.count(lv) && state[lv] == 1) leaf_vars.push_back(lv);
    }
    mapping.Set(iv, leaf_vars);
  }
  return mapping;
}

void MakeLoopNestFromDependentVars(
    const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map, size_t begin_iter_pos,
    bool new_loop_var, const std::unordered_set<IterVar>& skip_iter,
    std::unordered_map<IterVar, PrimExpr>* p_value_map, std::vector<std::vector<Stmt>>* p_nest,
    bool debug_keep_trivial_loop, const Array<DimInfo> fun_dimensions,
    const Map<Var, Array<Var>>& index_vars_loop_vars_depend_on,
    const Map<Var, Array<DimInfo>>& index_vars_loop_vars_are_needed_for,
    std::unordered_map<const VarNode*, int>& index_vars_dep_count) {
  auto var_dim_op = stage->op.as<BaseVarDimOpNode>();
  bool print = false;  //(stage->op->name == "i_c_sum.d");
  Stmt no_op = EvaluateNode::make(0);
  auto leaf_iter_vars = stage->leaf_iter_vars;

  std::vector<std::vector<Stmt>>& nest = *p_nest;
  std::unordered_map<IterVar, PrimExpr>& value_map = *p_value_map;
  nest.resize(leaf_iter_vars.size() + 1);

  std::unordered_set<const VarNode*> generated_loop_vars;
  std::unordered_set<const VarNode*> generated_index_vars;

  if (print) {
    std::cout << "[MLN] SEEDHA" << std::endl;
    for (const auto& it : index_vars_loop_vars_depend_on) {
      std::cout << "[MLN]  Loop var: " << it.first << std::endl;
      for (const auto& v : it.second) {
        std::cout << "[MLN]   Idx var: " << v << std::endl;
      }
    }

    std::cout << "[MLN] ULTA" << std::endl;
    for (const auto& it : index_vars_loop_vars_are_needed_for) {
      std::cout << "[MLN]  Loop var: " << it.first << std::endl;
      for (const auto& di : it.second) {
        std::cout << "[MLN]   Idx var: " << di->iv->var << std::endl;
      }
    }

    for (const auto& it : index_vars_dep_count) {
      std::cout << "[MLN]  Dep count: " << it.first << " " << it.second << std::endl;
    }
  }

  if (print) std::cout << "[MLN] GEN" << std::endl;
  for (size_t i = begin_iter_pos; i < leaf_iter_vars.size(); ++i) {
    auto iv = leaf_iter_vars[i];
    if (skip_iter.count(iv) || iv->iter_type == kOpaque || iv->iter_type == kLoopNestOpaque) {
      // skip this iteration.
      value_map[iv] = iv->var;
      continue;
    }

    if (print) std::cout << "[MLN]  Leaf var " << iv << std::endl;

    bool all_index_var_dependencies_satisfied = true;
    for (auto idx_var : index_vars_loop_vars_depend_on.at(iv->var)) {
      if (!generated_index_vars.count(idx_var.as<VarNode>()))
        all_index_var_dependencies_satisfied = false;
    }
    generated_loop_vars.insert(iv->var.as<VarNode>());

    // Bind iv could be another thread.
    IterVar bind_iv = iv;
    if (stage->iter_var_attrs.count(iv)) {
      IterVar bind_thread = stage->iter_var_attrs[iv]->bind_thread;
      if (bind_thread.defined()) bind_iv = bind_thread;
    }

    // PPF: Ranges of bound thread vars and the original itervars may
    // not be the same now. If we are a bound var, get the correct
    // range instead of assuming that the range will be the same as
    // the original var.
    Range dom = dom_map.at(iv);
    if (bind_iv != iv) {
      dom = dom_map.at(bind_iv);
    } else {
      dom = dom_map.at(iv);
      if (!all_index_var_dependencies_satisfied) {
        if (print) std::cout << "[MLN]   Relax" << std::endl;
        dom = Range::make_by_min_extent(UninterpFun::RelaxComplexUninterpCalls(dom->min),
                                        UninterpFun::RelaxComplexUninterpCalls(dom->extent));
      }
    }
    dom = UninterpFun::InlineUninterpFunCalls(dom);

    // initialize the offset and loop_level
    Var var = bind_iv->var;

    // Mark the iter var in the IR, to remember the point
    if (bind_iv->thread_tag.length() == 0) {
      // Only generate new loop if we're not bound to a thread.
      if (new_loop_var) {
        var = Var(iv->var->name_hint + ".init", bind_iv->var.dtype());
      }

      ForType for_type = ForType::Serial;
      IterVarAttr it_attr;
      if (stage->iter_var_attrs.count(iv)) {
        it_attr = stage->iter_var_attrs[iv];
      }
      if (it_attr.defined()) {
        switch (it_attr->iter_type) {
          case kUnrolled:
            for_type = ForType::Unrolled;
            break;
          case kPeeled:
            for_type = ForType::Peeled;
            break;
          case kVectorized:
            for_type = ForType::Vectorized;
            break;
          case kParallelized:
            for_type = ForType::Parallel;
            break;
          case kDataPar:
            break;
          case kTensorized:
            break;
          default:
            LOG(FATAL) << "Unknown iter type" << it_attr->iter_type << " in the iter_var_attrs";
        }
        CHECK_EQ(it_attr->pragma_keys.size(), it_attr->pragma_values.size());
        for (size_t k = 0; k < it_attr->pragma_keys.size(); ++k) {
          const std::string& pkey = it_attr->pragma_keys[k].as<StringImmNode>()->value;
          PrimExpr pvalue = it_attr->pragma_values[k];
          if (!pvalue.defined()) {
            pvalue = make_const(DataType::Int(32), 1);
          }
          nest[i + 1].emplace_back(
              AttrStmtNode::make(iv, tir::attr::pragma_scope_prefix + pkey, pvalue, no_op));
        }
      }
      if (!debug_keep_trivial_loop && is_one(dom->extent)) {
        nest[i + 1].emplace_back(LetStmtNode::make(var, dom->min, no_op));
        value_map[iv] = dom->min;
      } else if (is_zero(dom->min)) {
        nest[i + 1].emplace_back(
            ForNode::make(var, 0, dom->extent, for_type, DeviceAPI::None, no_op));
        value_map[iv] = var;
      } else {
        Var idx(bind_iv->var->name_hint + ".idx", bind_iv->var.dtype());

        nest[i + 1].emplace_back(
            ForNode::make(idx, 0, dom->extent, for_type, DeviceAPI::None, no_op));
        PrimExpr new_value = dom->min + idx;
        value_map[iv] = new_value;
        nest[i + 1].emplace_back(LetStmtNode::make(var, new_value, no_op));
      }

      if (it_attr.defined() && it_attr->prefetch_data.size() != 0) {
        CHECK(!is_one(dom->extent)) << "Cannot prefetch on trivial loop with extent=1";
        CHECK_EQ(it_attr->prefetch_data.size(), it_attr->prefetch_offset.size());
        for (size_t j = 0; j < it_attr->prefetch_data.size(); ++j) {
          nest[i + 1].emplace_back(AttrStmtNode::make(it_attr->prefetch_data[j],
                                                      tir::attr::prefetch_scope,
                                                      it_attr->prefetch_offset[j], no_op));
        }
      }
    } else if (bind_iv->thread_tag == "vthread" || bind_iv->thread_tag == "cthread") {
      // virtual thread
      // Always restrict threaded IterVar to starts from 0.
      CHECK(is_zero(dom->min));
      CHECK(is_positive_const(dom->extent));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmtNode::make(bind_iv, tir::attr::virtual_thread, dom->extent, no_op));
      value_map[iv] = var;
    } else if (bind_iv->thread_tag == "pipeline") {
      // pipeline marker.
      CHECK(is_zero(dom->min));
      CHECK(is_one(dom->extent));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmtNode::make(bind_iv, tir::attr::pipeline_exec_scope, dom->extent, no_op));
      value_map[iv] = dom->min;
    } else {
      // Always restrict threaded IterVar to starts from 0.
      CHECK(is_zero(dom->min));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmtNode::make(bind_iv, tir::attr::thread_extent, dom->extent, no_op));
      if (!debug_keep_trivial_loop && is_one(dom->extent)) {
        value_map[iv] = dom->min;
      } else {
        value_map[iv] = var;
      }
    }
    // annotate the extent of the IterVar
    if (!new_loop_var) {
      nest[i + 1].emplace_back(AttrStmtNode::make(iv, attr::loop_scope, iv->var, no_op));
    }

    if (index_vars_loop_vars_are_needed_for.count(iv->var)) {
      for (auto di : index_vars_loop_vars_are_needed_for.at(iv->var)) {
        auto var_node = di->iv->var.as<VarNode>();
        if (index_vars_dep_count.count(var_node)) {
          if (index_vars_dep_count.at(var_node) == 1) {
            if (print) std::cout << "[MLN]  Generating IV " << di->iv << std::endl;
            Array<PrimExpr> args;
            Array<Dimension> arg_dims;
            for (auto dim : di->ufun->dimensions) {
              arg_dims.push_back(dim);
              args.push_back(var_dim_op->GetIterVarFromDim(0, dim)->var);
            }
            // Generate index var here
            nest[i + 1].emplace_back(LetStmtNode::make(
                di->iv->var,
                di->ufun->substitute(Array<PrimExpr>(args), Array<Dimension>(arg_dims)), no_op));
            generated_index_vars.insert(var_node);
          } else if (index_vars_dep_count.at(var_node) > 1) {
            if (print) std::cout << "[MLN]  Decrementing IV " << di->iv << std::endl;
            index_vars_dep_count[var_node] = index_vars_dep_count.at(var_node) - 1;
          }
        }
      }
    }
  }

  for (const auto& di : fun_dimensions) {
    if (di->dim->isFunDim()) {
      Array<PrimExpr> args;
      Array<Dimension> arg_dims;
      for (auto dim : di->ufun->dimensions) {
        arg_dims.push_back(dim);
        args.push_back(var_dim_op->GetIterVarFromDim(0, dim)->var);
      }
      nest[leaf_iter_vars.size()].emplace_back(LetStmtNode::make(
          di->iv->var, di->ufun->substitute(Array<PrimExpr>(args), Array<Dimension>(arg_dims)),
          no_op));
    }
  }
}

std::vector<std::vector<Stmt>> MakeComputeOpLoopNest(
    const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map, size_t begin_iter_pos,
    bool new_loop_var, const std::unordered_set<IterVar>& skip_iter,
    std::unordered_map<IterVar, PrimExpr>* p_value_map, bool debug_keep_trivial_loop,
    Array<DimInfo> all_dimensions) {
  bool print = false;  //(stage->op->name == "Wh2h.local");
  if (print) std::cout << "[MLN] For " << stage->op->name << std::endl;
  // create the loop nest
  std::vector<std::vector<Stmt>> nest;
  std::unordered_map<IterVar, PrimExpr>& value_map = *p_value_map;

  std::unordered_map<const VarNode*, int> index_vars_dep_count;
  Map<Var, Array<Var>> index_vars_loop_vars_depend_on;
  Map<Var, Array<DimInfo>> index_vars_loop_vars_are_needed_for;

  IndexLoopVarDeps(stage, all_dimensions, dom_map, index_vars_loop_vars_depend_on,
                   index_vars_loop_vars_are_needed_for, index_vars_dep_count);

  Array<DimInfo> fun_dimensions;
  for (const auto& di : all_dimensions) {
    fun_dimensions.push_back(di);
  }

  MakeLoopNestFromDependentVars(stage, dom_map, begin_iter_pos, new_loop_var, skip_iter,
                                p_value_map, &nest, debug_keep_trivial_loop, fun_dimensions,
                                index_vars_loop_vars_depend_on, index_vars_loop_vars_are_needed_for,
                                index_vars_dep_count);

  // message passing to get offset of root iter vars.
  te::PassUpIndex(stage, dom_map, &value_map);
  return nest;
}

std::vector<std::vector<Stmt>> MakeScanOpLoopNest(
    const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map, size_t begin_iter_pos,
    bool new_loop_var, const std::unordered_set<IterVar>& skip_iter,
    std::unordered_map<IterVar, PrimExpr>* p_value_map, bool debug_keep_trivial_loop,
    Array<Dimension> explicit_dims) {
  auto scan_op = stage->op.as<ScanOpNode>();
  bool print = false;  //(stage->op->name == "Wh2h.local");
  if (print) std::cout << "[MLNs] For " << stage->op->name << std::endl;
  // create the loop nest
  std::vector<std::vector<Stmt>> nest;
  std::unordered_map<IterVar, PrimExpr>& value_map = *p_value_map;

  std::unordered_map<const VarNode*, int> index_vars_dep_count;
  Map<Var, Array<Var>> index_vars_loop_vars_depend_on;
  Map<Var, Array<DimInfo>> index_vars_loop_vars_are_needed_for;

  Array<DimInfo> explicit_dim_infos;
  for (const auto& dim : explicit_dims) {
    auto entry = scan_op->GetDimVarEntry(0, dim);
    if (print) std::cout << "[MLNs]   ExpDim " << dim << std::endl;
    explicit_dim_infos.push_back(DimInfoNode::make(dim, entry.iv, entry.value_expr));
  }

  IndexLoopVarDeps(stage, explicit_dim_infos, dom_map, index_vars_loop_vars_depend_on,
                   index_vars_loop_vars_are_needed_for, index_vars_dep_count);

  MakeLoopNestFromDependentVars(stage, dom_map, begin_iter_pos, new_loop_var, skip_iter,
                                p_value_map, &nest, debug_keep_trivial_loop, explicit_dim_infos,
                                index_vars_loop_vars_depend_on, index_vars_loop_vars_are_needed_for,
                                index_vars_dep_count);

  // message passing to get offset of root iter vars.
  te::PassUpIndex(stage, dom_map, &value_map);
  return nest;
}

std::vector<std::vector<Stmt>> MakeLoopNest(const Stage& stage,
                                            const std::unordered_map<IterVar, Range>& dom_map,
                                            size_t begin_iter_pos, bool new_loop_var,
                                            const std::unordered_set<IterVar>& skip_iter,
                                            std::unordered_map<IterVar, PrimExpr>* p_value_map,
                                            bool debug_keep_trivial_loop) {
  bool print = (stage->op.as<ScanOpNode>());  //(stage->op->name == "i_c_sum");
  if (print) std::cout << "[MLNi] Op: " << stage->op << std::endl;
  auto leaf_iter_vars = stage->leaf_iter_vars;
  Stmt no_op = EvaluateNode::make(0);
  // create the loop nest
  std::vector<std::vector<Stmt>> nest;
  nest.resize(leaf_iter_vars.size() + 1);
  std::unordered_map<IterVar, PrimExpr>& value_map = *p_value_map;

  for (size_t i = begin_iter_pos; i < leaf_iter_vars.size(); ++i) {
    auto iv = leaf_iter_vars[i];
    if (print) std::cout << "[MLNi]   IV: " << iv << std::endl;
    if (skip_iter.count(iv) || iv->iter_type == kOpaque || iv->iter_type == kLoopNestOpaque) {
      // skip this iteration.
      value_map[iv] = iv->var;
      continue;
    }
    if (print) std::cout << "[MLNi]     Unskipped" << std::endl;
    // Bind iv could be another thread.
    IterVar bind_iv = iv;
    if (stage->iter_var_attrs.count(iv)) {
      IterVar bind_thread = stage->iter_var_attrs[iv]->bind_thread;
      if (bind_thread.defined()) bind_iv = bind_thread;
    }

    Range dom = dom_map.at(iv);
    dom = UninterpFun::InlineUninterpFunCalls(dom);
    // std::cout << "[MLNi]     Dom: " << dom << std::endl;

    // initialize the offset and loop_level
    Var var = bind_iv->var;

    // Mark the iter var in the IR, to remember the point
    if (bind_iv->thread_tag.length() == 0) {
      // Only generate new loop if we're not bound to a thread.
      if (new_loop_var) {
        var = Var(iv->var->name_hint + ".init", bind_iv->var.dtype());
      }

      ForType for_type = ForType::Serial;
      IterVarAttr it_attr;
      if (stage->iter_var_attrs.count(iv)) {
        it_attr = stage->iter_var_attrs[iv];
      }
      if (it_attr.defined()) {
        switch (it_attr->iter_type) {
          case kUnrolled:
            for_type = ForType::Unrolled;
            break;
          case kPeeled:
            for_type = ForType::Peeled;
            break;
          case kVectorized:
            for_type = ForType::Vectorized;
            break;
          case kParallelized:
            for_type = ForType::Parallel;
            break;
          case kDataPar:
            break;
          case kTensorized:
            break;
          default:
            LOG(FATAL) << "Unknown iter type" << it_attr->iter_type << " in the iter_var_attrs";
        }
        CHECK_EQ(it_attr->pragma_keys.size(), it_attr->pragma_values.size());
        for (size_t k = 0; k < it_attr->pragma_keys.size(); ++k) {
          const std::string& pkey = it_attr->pragma_keys[k].as<StringImmNode>()->value;
          PrimExpr pvalue = it_attr->pragma_values[k];
          if (!pvalue.defined()) {
            pvalue = make_const(DataType::Int(32), 1);
          }
          nest[i + 1].emplace_back(
              AttrStmtNode::make(iv, tir::attr::pragma_scope_prefix + pkey, pvalue, no_op));
        }
      }
      if (!debug_keep_trivial_loop && is_one(dom->extent)) {
        nest[i + 1].emplace_back(LetStmtNode::make(var, dom->min, no_op));
        value_map[iv] = dom->min;
      } else if (is_zero(dom->min)) {
        nest[i + 1].emplace_back(
            ForNode::make(var, 0, dom->extent, for_type, DeviceAPI::None, no_op));
        value_map[iv] = var;
        // std::cout << "YO11 " << var << std::endl;
      } else {
        Var idx(bind_iv->var->name_hint + ".idx", bind_iv->var.dtype());
        nest[i + 1].emplace_back(
            ForNode::make(idx, 0, dom->extent, for_type, DeviceAPI::None, no_op));
        PrimExpr new_value = dom->min + idx;
        value_map[iv] = new_value;
        // std::cout << "YO11 " << new_value << std::endl;
        nest[i + 1].emplace_back(LetStmtNode::make(var, new_value, no_op));
      }
      if (it_attr.defined() && it_attr->prefetch_data.size() != 0) {
        CHECK(!is_one(dom->extent)) << "Cannot prefetch on trivial loop with extent=1";
        CHECK_EQ(it_attr->prefetch_data.size(), it_attr->prefetch_offset.size());
        for (size_t j = 0; j < it_attr->prefetch_data.size(); ++j) {
          nest[i + 1].emplace_back(AttrStmtNode::make(it_attr->prefetch_data[j],
                                                      tir::attr::prefetch_scope,
                                                      it_attr->prefetch_offset[j], no_op));
        }
      }
    } else if (bind_iv->thread_tag == "vthread" || bind_iv->thread_tag == "cthread") {
      // virtual thread
      // Always restrict threaded IterVar to starts from 0.
      CHECK(is_zero(dom->min));
      CHECK(is_positive_const(dom->extent));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmtNode::make(bind_iv, tir::attr::virtual_thread, dom->extent, no_op));
      value_map[iv] = var;
    } else if (bind_iv->thread_tag == "pipeline") {
      // pipeline marker.
      CHECK(is_zero(dom->min));
      CHECK(is_one(dom->extent));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmtNode::make(bind_iv, tir::attr::pipeline_exec_scope, dom->extent, no_op));
      value_map[iv] = dom->min;
    } else {
      // Always restrict threaded IterVar to starts from 0.
      CHECK(is_zero(dom->min));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmtNode::make(bind_iv, tir::attr::thread_extent, dom->extent, no_op));
      if (!debug_keep_trivial_loop && is_one(dom->extent)) {
        value_map[iv] = dom->min;
      } else {
        value_map[iv] = var;
      }
    }
    // annotate the extent of the IterVar
    if (!new_loop_var) {
      nest[i + 1].emplace_back(AttrStmtNode::make(iv, attr::loop_scope, iv->var, no_op));
    }
  }
  // message passing to get offset of root iter vars.
  te::PassUpIndex(stage, dom_map, &value_map);
  return nest;
}

std::vector<Stmt> MakeIfNest(const std::vector<PrimExpr>& predicates) {
  Stmt no_op = EvaluateNode::make(0);
  std::vector<Stmt> nest;
  for (const PrimExpr& cond : predicates) {
    nest.emplace_back(IfThenElseNode::make(cond, no_op));
  }
  return nest;
}

// replacer to replace tensors
class TensorReplacer : public tir::StmtExprMutator {
 public:
  explicit TensorReplacer(const std::unordered_map<Tensor, Tensor>& vmap) : vmap_(vmap) {
    for (auto it : vmap) {
      if (it.first->op == it.second->op) {
        std::cout << "How'd this happen?" << std::endl;
        CHECK(false);
      }
    }
  }

  PrimExpr VisitExpr_(const tir::CallNode* op) final {
    if (auto ufun = op->func.as<UninterpFunNode>()) {
      PrimExpr old_body = ufun->body;
      bool old_found = false;
      std::swap(found, old_found);
      found = false;
      PrimExpr new_body = this->VisitExpr(old_body);
      UninterpFun new_ufun = Downcast<UninterpFun>(op->func);
      if (found) {
        new_ufun = UninterpFunNode::make(ufun->fname, ufun->range, ufun->dimensions,
                                         ufun->parameters, new_body);
        found = true;
      } else {
        std::swap(found, old_found);
      }

      PrimExpr ret = tir::CallNode::make(op->dtype, op->name, op->args, op->call_type,
                                         op->argument_dimensions, new_ufun, op->value_index);
      // std::cout << "[TR] ReplacedU " << op->func << " " << it->second->op << std::endl;
      return ret;
    } else if (op->func.as<OperationNode>()) {
      Tensor t = Downcast<Operation>(op->func).output(op->value_index);
      auto it = vmap_.find(t);
      if (it != vmap_.end()) {
        PrimExpr ret = tir::CallNode::make(op->dtype, it->second->op->name + ".r", op->args,
                                           op->call_type, it->second->op, it->second->value_index);
        found = true;
        // std::cout << "[TR] Replaced " << op->func << " " << it->second->op << std::endl;
        return this->VisitExpr(ret);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  // whether it is found.
  bool found{false};

 private:
  const std::unordered_map<Tensor, Tensor>& vmap_;
};

Stmt ReplaceTensor(Stmt stmt, const std::unordered_map<Tensor, Tensor>& replace) {
  TensorReplacer repl(replace);
  Stmt ret = repl(stmt);
  return repl.found ? ret : stmt;
}
PrimExpr ReplaceTensor(PrimExpr expr, const std::unordered_map<Tensor, Tensor>& replace) {
  TensorReplacer repl(replace);
  PrimExpr ret = repl(expr);
  return repl.found ? ret : expr;
}

void CollectTensors(Array<Tensor>& collected_tensors, Array<PrimExpr> exprs) {
  std::unordered_set<Tensor> visited;
  auto collector = [&collected_tensors, &visited](const ObjectRef& n) {
    const tir::CallNode* call = n.as<tir::CallNode>();
    if (call != nullptr && call->func.defined()) {
      if (call->func.as<UninterpFunNode>()) {
      } else {
        Tensor t = Downcast<Operation>(call->func).output(call->value_index);
        if (!visited.count(t)) {
          // if (t->op->name == "b_d.shared") std::cout << "[CT]   Found " << t->op << std::endl;
          collected_tensors.push_back(t);
          visited.insert(t);
        }
      }
    }
  };

  for (auto e : exprs) {
    // std::cout << "[CT] Collecting in " << e << std::endl;
    tir::PostOrderVisit(e, collector);
  }
}

Stmt Substitute(Stmt s, const std::unordered_map<IterVar, PrimExpr>& value_map) {
  std::unordered_map<const VarNode*, PrimExpr> init;
  for (const auto& kv : value_map) {
    init[kv.first->var.get()] = kv.second;
  }
  return tir::Substitute(s, init);
}

IterVarType ForTypeToIterVarType(tir::ForType for_type) {
  switch (for_type) {
    case ForType::Serial:
      return kDataPar;
    case ForType::Parallel:
      return kParallelized;
    case ForType::Vectorized:
      return kVectorized;
    case ForType::Unrolled:
      return kUnrolled;
    default:
      return kDataPar;
  }
}

tir::ForType IterVarTypeToForType(IterVarType iter_type) {
  switch (iter_type) {
    case kDataPar:
      return ForType::Serial;
    case kParallelized:
      return ForType::Parallel;
    case kVectorized:
      return ForType::Vectorized;
    case kUnrolled:
      return ForType::Unrolled;
    default:
      return ForType::Serial;
  }
}
}  // namespace te
}  // namespace tvm
