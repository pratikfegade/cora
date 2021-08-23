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

#include <tvm/arith/int_set.h>
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

const BaseVarDimOpNode* GetBaseVarDimOp(Operation op) {
  if (op.as<ScanOpNode>()) return op.as<ScanOpNode>();
  if (op.as<SingleKernelEnvelopeOpNode>()) return op.as<SingleKernelEnvelopeOpNode>();
  if (op.as<SpecializationEnvelopeOpNode>()) return op.as<SpecializationEnvelopeOpNode>();
  if (op.as<ComputeOpNode>()) return op.as<ComputeOpNode>();
  return nullptr;
}

IntSet TranslateIterVarsFromConsumerToProducer(IntSet set, Operation consumer, Tensor tensor) {
  const BaseVarDimOpNode* c = GetBaseVarDimOp(consumer);
  const BaseVarDimOpNode* p = GetBaseVarDimOp(tensor->op);

  bool print = false;  //(tensor->op->name == "Aexp");
  if (print) {
    std::cout << "[TIV] P/C " << consumer << " " << tensor->op << std::endl;
  }

  if (c == nullptr || p == nullptr) return set;

  // if (print) {
  // std::cout << "[TIV]   Map1Size " << c->dim2var_maps.size() << std::endl;
  // }

  std::unordered_map<const VarNode*, PrimExpr> vsub;
  for (const auto& dim2var_map : c->dim2var_maps) {
    // if (print) {
    // std::cout << "[TIV]    Map2Size " << dim2var_map.size() << std::endl;
    // }
    for (const auto& it : dim2var_map) {
      auto dim = it.first;
      auto var_node = it.second.iv->var.as<VarNode>();

      CHECK(p->dim2var_maps.size() > tensor->value_index)
          << p->dim2var_maps.size() << " " << tensor << " " << consumer;

      // if (print) {
      //   std::cout << "[TIV]    Dim " << dim->name << " " << dim << std::endl;
      //   for (auto it : p->dim2var_maps[tensor->value_index]) {
      //     std::cout << "[TIV]      PDim " << it.first->name << " " << it.first << std::endl;
      //   }
      // }

      if (p->dim2var_maps[tensor->value_index].count(dim)) {
        if (print) {
          std::cout << "[TIV]   Var " << var_node->name_hint << " "
                    << p->dim2var_maps[tensor->value_index].at(dim).iv->var << std::endl;
        }
        vsub[var_node] = p->dim2var_maps[tensor->value_index].at(dim).iv->var;
      } else {
        if (print) {
          std::cout << "[TIV]   Dim not found " << dim->name << " " << it.second.iv << std::endl;
        }
      }
    }
  }

  return arith::ReplaceIntSet(set, vsub);
}

void IndexLoopVarDeps(const Stage& stage, Array<DimInfo> all_dimensions,
                      const std::unordered_map<IterVar, Range>& dom_map,
                      std::unordered_map<IterVar, PrimExpr>* p_value_map,
                      Map<Var, Array<Var>>& index_vars_loop_vars_depend_on,
                      Map<Var, Array<Var>>& root_vars_loop_vars_depend_on,
                      std::unordered_map<const VarNode*, int>& index_vars_dep_count) {
  std::unordered_map<IterVar, PrimExpr>& value_map = *p_value_map;
  bool print = false;  //(stage->op->name == "lf_h2h.ila");
  if (print) std::cout << "[ILVD] Op " << stage->op << std::endl;
  auto var_dim_op = stage->op.as<BaseVarDimOpNode>();
  CHECK(var_dim_op);
  std::unordered_map<const VarNode*, const DimInfoNode*> index_vars;
  std::unordered_set<const VarNode*> root_vars;
  std::unordered_set<const VarNode*> generated_vars;

  for (const auto di : all_dimensions) {
    CHECK(!di->dim->isFunDim());
    root_vars.insert(di->iv->var.as<VarNode>());
  }

  for (const auto lv : stage->leaf_iter_vars) {
    generated_vars.insert(lv->var.as<VarNode>());
  }

  std::unordered_set<const VarNode*> already_generated_vars;
  for (auto it : value_map) {
    already_generated_vars.insert(it.first->var.as<VarNode>());
  }

  for (const auto lv : stage->leaf_iter_vars) {
    if (print) std::cout << "[ILVD]  LV " << lv << std::endl;
    Array<Var> dep_idx_vars;
    Array<Var> dep_loop_vars;
    PrimExpr extent = dom_map.at(lv)->extent;
    auto input_vars = VarCollector().collect(extent);
    for (auto inp : input_vars) {
      if (print)
        std::cout << "[ILVD]  Inp " << inp->name_hint << " " << index_vars.count(inp) << " "
                  << root_vars.count(inp) << std::endl;
      if (index_vars.count(inp) && !already_generated_vars.count(inp))
        dep_idx_vars.push_back(GetRef<Var>(inp));
      if (root_vars.count(inp) && !already_generated_vars.count(inp))
        dep_loop_vars.push_back(GetRef<Var>(inp));
    }
    index_vars_loop_vars_depend_on.Set(lv->var, dep_idx_vars);
    root_vars_loop_vars_depend_on.Set(lv->var, dep_loop_vars);
  }
}

void MakeLoopNestFromDependentVars(
    const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map, size_t begin_iter_pos,
    bool new_loop_var, const std::unordered_set<IterVar>& skip_iter,
    std::unordered_map<IterVar, PrimExpr>* p_value_map, std::vector<std::vector<Stmt>>* p_nest,
    bool debug_keep_trivial_loop, const Array<DimInfo> fun_dimensions,
    const Map<Var, Array<Var>>& index_vars_loop_vars_depend_on,
    const Map<Var, Array<Var>>& root_vars_loop_vars_depend_on,
    std::unordered_map<const VarNode*, int>& index_vars_dep_count) {
  // debug_keep_trivial_loop = true;
  auto var_dim_op = stage->op.as<BaseVarDimOpNode>();
  bool print = false;
  // bool print = (stage->op->name == "O");
  if (print) std::cout << "[MLN] Op " << stage->op << std::endl;
  Stmt no_op = EvaluateNode::make(0);
  auto leaf_iter_vars = stage->leaf_iter_vars;

  std::vector<std::vector<Stmt>>& nest = *p_nest;
  std::unordered_map<IterVar, PrimExpr>& value_map = *p_value_map;

  nest.resize(leaf_iter_vars.size() + 1);

  std::unordered_set<const VarNode*> generated_loop_vars;
  std::unordered_set<const VarNode*> generated_index_vars;

  if (print) std::cout << "[MLN] GEN" << std::endl;
  for (size_t i = begin_iter_pos; i < leaf_iter_vars.size(); ++i) {
    auto iv = leaf_iter_vars[i];
    IterVarAttr it_attr;
    if (stage->iter_var_attrs.count(iv)) {
      it_attr = stage->iter_var_attrs[iv];
    }
    if (skip_iter.count(iv) || iv->iter_type == kOpaque || iv->iter_type == kLoopNestOpaque ||
        iv->iter_type == kSplit || (it_attr.defined() && it_attr->iter_type == kSplit)) {
      if (print) std::cout << "[MLN]  Skipping " << iv << " " << iv->iter_type << " " << std::endl;
      value_map[iv] = iv->var;
      continue;
    }

    int hfuse_group_id = it_attr.defined() ? it_attr->hfuse_group_id : -1;

    bool all_dependencies_satisfied = true;
    for (auto idx_var : index_vars_loop_vars_depend_on.at(iv->var)) {
      if (!generated_index_vars.count(idx_var.as<VarNode>())) all_dependencies_satisfied = false;
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
      if (!all_dependencies_satisfied) {
        if (print) std::cout << "[MLN]   Relax" << std::endl;
        dom = Range::make_by_min_extent(UninterpFun::RelaxUninterpCallsMaxInclusive(dom->min),
                                        UninterpFun::RelaxUninterpCallsMaxInclusive(dom->extent));
      }  // else if (relaxed_ranges.count(iv)) {
      //   dom = relaxed_ranges.at(iv);
      // }
    }
    dom = UninterpFun::InlineUninterpFunCalls(dom);

    if (print) {
      std::cout << "[MLN]  Leaf var " << iv << " " << bind_iv->thread_tag << " " << dom
                << std::endl;
    }

    // initialize the offset and loop_level
    Var var = bind_iv->var;

    bool created_thread_extent = true;
    // Mark the iter var in the IR, to remember the point
    if (bind_iv->thread_tag.length() == 0) {
      // Only generate new loop if we're not bound to a thread.
      if (new_loop_var) {
        var = Var(iv->var->name_hint + ".init", bind_iv->var.dtype());
      }

      ForType for_type = ForType::Serial;
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
      if (!debug_keep_trivial_loop && is_one(tir::Simplify(dom->extent))) {
        CHECK(hfuse_group_id < 0) << "Trying to hfuse iv of extent 1";
        nest[i + 1].emplace_back(LetStmtNode::make(var, dom->min, no_op));
        value_map[iv] = dom->min;
      } else if (is_zero(dom->min)) {
        nest[i + 1].emplace_back(
            ForNode::make(var, 0, dom->extent, for_type, DeviceAPI::None, no_op, hfuse_group_id));
        value_map[iv] = var;
      } else {
        Var idx(bind_iv->var->name_hint + ".idx", bind_iv->var.dtype());

        nest[i + 1].emplace_back(
            ForNode::make(idx, 0, dom->extent, for_type, DeviceAPI::None, no_op, hfuse_group_id));
        PrimExpr new_value = dom->min + idx;
        value_map[iv] = new_value;
        nest[i + 1].emplace_back(LetStmtNode::make(var, new_value, no_op));
      }

      if (it_attr.defined() && it_attr->prefetch_data.size() != 0) {
        CHECK(!is_one(dom->extent)) << "Cannot prefetch on trivial loop with extent=1";
        CHECK_EQ(it_attr->prefetch_data.size(), it_attr->prefetch_offset.size());
        for (size_t j = 0; j < it_attr->prefetch_data.size(); ++j) {
          nest[i + 1].emplace_back(
              AttrStmtNode::make(it_attr->prefetch_data[j], tir::attr::prefetch_scope,
                                 it_attr->prefetch_offset[j], no_op, hfuse_group_id));
        }
      }
    } else if (bind_iv->thread_tag == "vthread" || bind_iv->thread_tag == "cthread") {
      CHECK(hfuse_group_id < 0) << "Trying to hfuse v/c thread iv";
      // virtual thread
      // Always restrict threaded IterVar to starts from 0.
      CHECK(is_zero(dom->min));
      CHECK(is_positive_const(Simplify(dom->extent))) << iv << " " << bind_iv << " " << dom;
      // annotate the extent of the IterVar
      if (stage->iter_var_attrs.count(iv)) {
        auto it_attr = stage->iter_var_attrs[iv];
        if (it_attr.defined() && !it_attr->unroll_vthread) {
          nest[i + 1].emplace_back(
              AttrStmtNode::make(bind_iv, tir::attr::virtual_thread_no_unroll, 1, no_op));
        }
      }
      nest[i + 1].emplace_back(
          AttrStmtNode::make(bind_iv, tir::attr::virtual_thread, dom->extent, no_op));
      value_map[iv] = var;
    } else if (bind_iv->thread_tag == "pipeline") {
      CHECK(hfuse_group_id < 0) << "Trying to hfuse a pipelined iv";
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

      PrimExpr extent = dom->extent;
      // Check if the specified range and the inferred ranges are
      // equal. If not, we print out a warn and defer to the specified
      // range that the user probably specified
      if (bind_iv->dom.defined() && !is_zero(Simplify(bind_iv->dom->extent - dom->extent))) {
        LOG(WARNING) << "Specified and inferred extents do not match for thread var "
                     << bind_iv->var << " for op " << stage->op->name << ". They are "
                     << bind_iv->dom << " and " << dom;
        extent = bind_iv->dom->extent;
      }

      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmtNode::make(bind_iv, tir::attr::thread_extent, extent, no_op, hfuse_group_id));
      created_thread_extent = true;
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

    // Check if ragged dimensions are reordered in ways that violate
    // dependencies
  }
}

std::vector<std::vector<Stmt>> MakeComputeOpLoopNest(
    const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map, size_t begin_iter_pos,
    bool new_loop_var, const std::unordered_set<IterVar>& skip_iter,
    std::unordered_map<IterVar, PrimExpr>* p_value_map, bool debug_keep_trivial_loop,
    Array<DimInfo> all_dimensions) {
  bool print = false;  //(stage->op->name == "l_rz_mv");
  if (print) std::cout << "[MLN] For " << stage->op->name << std::endl;
  // create the loop nest
  std::vector<std::vector<Stmt>> nest;
  std::unordered_map<IterVar, PrimExpr>& value_map = *p_value_map;

  std::unordered_map<const VarNode*, int> index_vars_dep_count;
  Map<Var, Array<Var>> index_vars_loop_vars_depend_on;
  Map<Var, Array<Var>> root_vars_loop_vars_depend_on;

  IndexLoopVarDeps(stage, all_dimensions, dom_map, p_value_map, index_vars_loop_vars_depend_on,
                   root_vars_loop_vars_depend_on, index_vars_dep_count);

  Array<DimInfo> fun_dimensions;
  for (const auto& di : all_dimensions) {
    fun_dimensions.push_back(di);
  }

  MakeLoopNestFromDependentVars(stage, dom_map, begin_iter_pos, new_loop_var, skip_iter,
                                p_value_map, &nest, debug_keep_trivial_loop, fun_dimensions,
                                index_vars_loop_vars_depend_on, root_vars_loop_vars_depend_on,
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
  const BaseVarDimOpNode* gen_op = stage->op.as<ScanOpNode>();
  if (gen_op == nullptr) {
    gen_op = stage->op.as<ConditionalOpNode>();
  }
  CHECK(gen_op);

  bool print = false;  //(stage->op->name == "lf_if");
  if (print) std::cout << "[MLNs] For " << stage->op->name << std::endl;
  // create the loop nest
  std::vector<std::vector<Stmt>> nest;
  std::unordered_map<IterVar, PrimExpr>& value_map = *p_value_map;

  std::unordered_map<const VarNode*, int> index_vars_dep_count;
  Map<Var, Array<Var>> index_vars_loop_vars_depend_on;
  Map<Var, Array<Var>> root_vars_loop_vars_depend_on;

  Array<DimInfo> explicit_dim_infos;
  for (const auto& dim : explicit_dims) {
    if (print) std::cout << "[MLNs]   ExpDim " << dim << std::endl;
    auto entry = gen_op->GetDimVarEntry(0, dim);
    explicit_dim_infos.push_back(DimInfoNode::make(dim, entry.iv));
  }

  IndexLoopVarDeps(stage, explicit_dim_infos, dom_map, p_value_map, index_vars_loop_vars_depend_on,
                   root_vars_loop_vars_depend_on, index_vars_dep_count);

  MakeLoopNestFromDependentVars(stage, dom_map, begin_iter_pos, new_loop_var, skip_iter,
                                p_value_map, &nest, debug_keep_trivial_loop, explicit_dim_infos,
                                index_vars_loop_vars_depend_on, root_vars_loop_vars_depend_on,
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
  bool print = false;  // (stage->op->name == "unified");
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
    if (skip_iter.count(iv) || iv->iter_type == kOpaque || iv->iter_type == kLoopNestOpaque ||
        iv->iter_type == kSplit) {
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
      if (print) std::cout << "[MLNi]     Loop type " << for_type << std::endl;
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
      CHECK(is_positive_const(Simplify(dom->extent)));
      // annotate the extent of the IterVar
      std::cout << "[MLN] Vthread " << iv << std::endl;
      if (stage->iter_var_attrs.count(iv)) {
        auto it_attr = stage->iter_var_attrs[iv];
        if (it_attr.defined() && !it_attr->unroll_vthread) {
          std::cout << "[MLN]  No unroll" << std::endl;
          nest[i + 1].emplace_back(
              AttrStmtNode::make(bind_iv, tir::attr::virtual_thread_no_unroll, 1, no_op));
        }
      }
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

      PrimExpr extent = dom->extent;
      // Check if the specified range and the inferred ranges are
      // equal. If not, we print out a warn and defer to the specified
      // range that the user probably specified
      if (bind_iv->dom.defined() && !bind_iv->dom->extent.same_as(dom->extent)) {
        LOG(WARNING) << "Specified and inferred extents do not match for thread var "
                     << bind_iv->var << " for op " << stage->op->name << ". They are "
                     << bind_iv->dom << " and " << dom;
        extent = bind_iv->dom->extent;
      }

      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmtNode::make(bind_iv, tir::attr::thread_extent, extent, no_op));
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

  UninterpFun VisitUninterpFun(UninterpFun ufun) {
    PrimExpr old_body = ufun->body;
    bool old_found = found;
    std::swap(found, old_found);
    found = false;
    PrimExpr new_body = old_body.defined() ? this->VisitExpr(old_body) : old_body;
    UninterpFun new_ufun = ufun;
    if (found) {
      new_ufun = UninterpFunNode::make(ufun->fname, ufun->range, ufun->dimensions, ufun->parameters,
                                       new_body, ufun->type);
      found = true;
    } else {
      std::swap(found, old_found);
    }
    return new_ufun;
  }

  PrimExpr VisitExpr_(const tir::CallNode* op) final {
    if (auto ufun = op->func.as<UninterpFunNode>()) {
      UninterpFun new_ufun = VisitUninterpFun(Downcast<UninterpFun>(op->func));

      PrimExpr ret = tir::CallNode::make(op->dtype, op->name, op->args, op->call_type, op->arg_dims,
                                         new_ufun, op->value_index, op->custom_realize_bounds);
      return ret;
    } else if (auto op_node = op->func.as<OperationNode>()) {
      Tensor t = Downcast<Operation>(op->func).output(op->value_index);
      auto it = vmap_.find(t);
      if (it != vmap_.end()) {
        PrimExpr ret = tir::CallNode::make(op->dtype, it->second->op->name + ".r", op->args,
                                           op->call_type, op->arg_dims, it->second->op,
                                           it->second->value_index, op->custom_realize_bounds);
        found = true;
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

UninterpFun ReplaceTensor(UninterpFun ufun, const std::unordered_map<Tensor, Tensor>& replace) {
  TensorReplacer repl(replace);
  UninterpFun ret = repl.VisitUninterpFun(ufun);
  return repl.found ? ret : ufun;
}

Modes ReplaceTensor(Modes mode, const std::unordered_map<Tensor, Tensor>& replace) {
  bool changed = false;

  auto handle_uf_array = [&changed, &replace](Array<UninterpFun> arr) {
    Array<UninterpFun> new_arr;
    for (auto uf : arr) {
      if (uf.defined() && uf->body.defined()) {
        auto new_uf = ReplaceTensor(uf, replace);
        if (new_uf != uf) {
          changed = true;
        }
        new_arr.push_back(new_uf);
      } else {
        new_arr.push_back(uf);
      }
    }
    return new_arr;
  };

  auto new_l_funs = handle_uf_array(mode->l_funs);
  auto new_l_fun_mins = handle_uf_array(mode->l_fun_mins);
  auto new_a_funs = handle_uf_array(mode->a_funs);

  if (changed) {
    return ModesNode::make(mode->dimensions, mode->l_maxes, new_l_fun_mins, new_l_funs, new_a_funs,
                           mode->loop_layout);
  } else {
    return mode;
  }
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
          // if (t->op->name == "b_d") std::cout << "[CT]   Found " << t->op << std::endl;
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

std::vector<std::vector<Stmt>> MergeWhileHoisting(const Stage& s,
                                                  const std::vector<std::vector<Stmt>>& defs,
                                                  const std::vector<Stmt>& preds) {
  std::vector<std::vector<Stmt>> ret;
  ret.resize(defs.size());
  std::unordered_set<const Object*> generated_preds;
  std::unordered_set<const Object*> generated_vars;
  std::unordered_set<const Object*> leaf_vars;
  for (auto lv : s->leaf_iter_vars) {
    leaf_vars.insert(lv->var.get());
  }

  VarCollector collector;

  auto generate_preds = [&](int idx) {
    for (auto pred : preds) {
      if (generated_preds.count(pred.get())) continue;
      auto var_nodes = collector.collect(pred);
      bool generate = true;
      for (auto var_node : var_nodes) {
        // std::cout << "[NEED_VAR] " << var_node->name_hint << " " << var_node << std::endl;
        if (!generated_vars.count(var_node) && leaf_vars.count(var_node)) {
          generate = false;
        }
      }
      if (generate) {
        ret[idx].push_back(pred);
        generated_preds.insert(pred.get());
      }
    }
  };

  generate_preds(0);

  for (size_t i = 0; i < defs.size(); ++i) {
    auto inner_def = defs[i];
    for (auto def : defs[i]) {
      if (auto let = def.as<LetStmtNode>()) {
        generated_vars.insert(let->var.get());
      } else if (auto for_stmt = def.as<ForNode>()) {
        generated_vars.insert(for_stmt->loop_var.get());
      } else if (auto attr_stmt = def.as<AttrStmtNode>()) {
        if (attr_stmt->attr_key == tir::attr::thread_extent) {
          // std::cout << "[GEN_VAR] " << Downcast<IterVar>(attr_stmt->node)->var << " "
          // << Downcast<IterVar>(attr_stmt->node)->var.get() << std::endl;
          generated_vars.insert(Downcast<IterVar>(attr_stmt->node)->var.get());
        }
      } else {
        CHECK(false);
      }

      ret[i].push_back(def);
      generate_preds(i);
    }
  }
  return ret;
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
