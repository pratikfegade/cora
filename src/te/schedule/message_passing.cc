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
 * \file message_passing.cc
 * \brief The message passing domain.
 */
#include "message_passing.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ir/attrs.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>

#include "../../arith/compute_expr.h"
#include "../../runtime/thread_storage_scope.h"
#include "../../tir/ir/var_replacer.h"
#include "schedule_utils.h"

namespace tvm {
namespace te {

using namespace tir;

void Update(std::unordered_map<IterVar, Range>* p_state, const IterVar& iv, Range r,
            arith::Analyzer* analyzer) {
  auto it = p_state->find(iv);
  if (it == p_state->end()) {
    (*p_state)[iv] = r;
    analyzer->Bind(iv->var, r);
  } else {
    // TODO (ppf): HACK HACK HACK. We're commenting out an error
    // condition that should ideally be checked and reported
    bool match = is_zero(it->second->min) &&
                 analyzer->CanProve(
                     UninterpFun::InlineUninterpFunCalls(r->extent - it->second->extent) == 0);
    CHECK(match) << iv << " domain already inferred,"
                 << " cannot prove their extents are the same " << it->second->extent << " vs "
                 << r->extent << " " << it->second;
  }
}

void UpdateShim(const Stage& stage, std::unordered_map<IterVar, Range>* p_state, const IterVar& iv,
                Range r, arith::Analyzer* analyzer) {
  bool print = false;  // stage->op->name == "O.local";
  if (print) std::cout << "Updating " << iv << " " << r << std::endl;
  Update(p_state, iv, r, analyzer);
}

PrimExpr zero_if_args_zero_ufun_call(DataType dtype, Array<PrimExpr> args, Array<Dimension> dims,
                                     UninterpFun func) {
  Array<PrimExpr> compressed_args;
  Array<Dimension> compressed_dims;

  for (size_t i = 0; i < dims.size(); ++i) {
    if (func->dimensions.Contains(dims[i])) {
      compressed_args.push_back(args[i]);
      compressed_dims.push_back(dims[i]);
    }
  }

  bool args_zero = true;
  for (auto arg : compressed_args) {
    if (!is_zero(arg)) {
      args_zero = false;
      break;
    }
  }

  if (args_zero) {
    return IntImm(dtype, 0);
  } else {
    return func.MakeCallTo(compressed_args, compressed_dims, dtype);
  }
}

void PassDownDomain(const Stage& stage, std::unordered_map<IterVar, Range>* p_state,
                    arith::Analyzer* actx, bool allow_missing) {
  bool print = false;  // stage->op->name == "S";
  auto ceil_div = [actx, print](PrimExpr a, PrimExpr b) {
    if (actx->CanProve(indexmod(a, b) == 0)) {
      return actx->Simplify(indexdiv(a, b));
    }
    return actx->Simplify(indexdiv(a + (b - 1), b));
  };

  auto& state = *p_state;
  // forward iteration on relations
  if (print) std::cout << "[PDD] Stage " << stage << std::endl;
  for (IterVarRelation rel : stage->relations) {
    if (const SplitNode* r = rel.as<SplitNode>()) {
      if (!state.count(r->parent)) {
        CHECK(allow_missing) << stage << " " << r->parent;
        continue;
      }
      CHECK(!state.count(r->inner));
      const Range& range_parent = state.at(r->parent);
      if (print) {
        std::cout << "[SPL] P " << r->parent->var << " " << range_parent << std::endl;
      }

      if (r->factor.defined()) {
        if (print) {
          std::cout << "[SPL]    When" << std::endl;
          auto outer_extent = ceil_div(range_parent->extent, r->factor);
          std::cout << "[SPL]    What" << std::endl;
          std::cout << "[SPL]    FAC " << r->outer->var << " " << outer_extent << std::endl;
          std::cout << "[SPL]    FAC " << r->inner->var << " "
                    << Range::make_by_min_extent(0, r->factor) << std::endl;
        }
        UpdateShim(stage, p_state, r->inner, Range::make_by_min_extent(0, r->factor), actx);
        if (print)
          std::cout << "[SPL]    FAC " << r->outer->var << " "
                    << ceil_div(range_parent->extent, r->factor) << std::endl;
        UpdateShim(stage, p_state, r->outer,
                   Range::make_by_min_extent(0, ceil_div(range_parent->extent, r->factor)), actx);
      } else {
        UpdateShim(stage, p_state, r->outer, Range::make_by_min_extent(0, r->nparts), actx);
        UpdateShim(stage, p_state, r->inner,
                   Range::make_by_min_extent(0, ceil_div(range_parent->extent, r->nparts)), actx);
      }
    } else if (const RaggedFuseNode* r = rel.as<RaggedFuseNode>()) {
      if (!state.count(r->outer) || !state.count(r->inner)) {
        CHECK(allow_missing);
        continue;
      }
      if (print) std::cout << "[FPL]  FREL for " << r->fused->var << std::endl;
      const Range& range_outer = state.at(r->outer);
      const Range& range_inner_unreplaced = state.at(r->inner);

      std::unordered_map<const VarNode*, PrimExpr> vsub_min;
      std::unordered_map<const VarNode*, PrimExpr> vsub_max;
      {
        for (auto iv : stage->all_iter_vars) {
          if (state.count(iv)) {
            auto range = state.at(iv);
            vsub_min[iv->var.as<VarNode>()] = range->min;
            vsub_max[iv->var.as<VarNode>()] = range->max_inclusive();
          }
        }
      }

      Range range_inner = Range::make_by_min_max_inclusive(
          VarReplacer(vsub_min)(range_inner_unreplaced->min),
          VarReplacer(vsub_max)(range_inner_unreplaced->max_inclusive()));
      if (print)
        std::cout << "[RFPL] O/I " << range_outer->extent << " " << range_inner->extent
                  << std::endl;
      if (print) std::cout << "[FPL]   Outer " << range_outer << std::endl;
      if (print) std::cout << "[FPL]   Inner " << range_inner << std::endl;
      auto fused_min = zero_if_args_zero_ufun_call(
          r->fused->var.dtype(), {range_outer->min, range_inner->min},
          r->outer_inner_to_fused_uf->dimensions, r->outer_inner_to_fused_uf);
      auto fused_max_inclusive = Simplify(zero_if_args_zero_ufun_call(
          r->fused->var.dtype(), {range_outer->max_inclusive(), range_inner->max_inclusive()},
          r->outer_inner_to_fused_uf->dimensions, r->outer_inner_to_fused_uf));
      state[r->fused] = Range::make_by_min_max_inclusive(fused_min, fused_max_inclusive);
      if (print) {
        if (print) std::cout << "[FPL]    Fused " << state[r->fused] << std::endl;
        std::cout << "[RFPL]    Vars " << r->outer->var << " " << r->inner->var << " "
                  << r->fused->var << std::endl;
        std::cout << "[RFPL]    F " << fused_max_inclusive << std::endl;
      }
    } else if (const FuseNode* r = rel.as<FuseNode>()) {
      if (!state.count(r->outer) || !state.count(r->inner)) {
        CHECK(allow_missing);
        continue;
      }
      const Range& range_outer = state.at(r->outer);
      const Range& range_inner = state.at(r->inner);
      state[r->fused] = Range::make_by_min_extent(0, range_outer->extent * range_inner->extent);
      if (print) {
        std::cout << "[FPL]    Vars " << r->outer->var << " " << r->inner->var << " "
                  << r->fused->var << std::endl;
        std::cout << "[FPL]    F " << range_outer->extent << " " << range_inner->extent
                  << std::endl;
      }
    } else if (const RebaseNode* r = rel.as<RebaseNode>()) {
      if (!state.count(r->parent)) {
        // std::cout << "[PDD] Op " << stage->op << " " << r->parent << std::endl;
        CHECK(allow_missing) << stage->op << " " << r->parent;
        continue;
      }
      // std::cout << "[PDD] Rebasing " << stage << " " << r->rebased << " "
      // << Range::make_by_min_extent(0, state.at(r->parent)->extent) << std::endl;
      UpdateShim(stage, p_state, r->rebased,
                 Range::make_by_min_extent(0, state.at(r->parent)->extent), actx);
    } else if (const SingletonNode* s = rel.as<SingletonNode>()) {
      UpdateShim(stage, p_state, s->iter, Range::make_by_min_extent(0, 1), actx);
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
  // update the extents of bound threads.
  for (auto kv : stage->iter_var_attrs) {
    if (kv.second->bind_thread.defined()) {
      CHECK(state.count(kv.first)) << kv.first;
      Range r = state.at(kv.first);
      IterVar b_iv = kv.second->bind_thread;

      if (!is_zero(r->min)) {
        LOG(INFO) << "Inferred range for CUDA thread has a non-zero min when passing down " << stage
                  << " " << kv.first << " " << b_iv->var->name_hint << " " << r;
      }
      CHECK(is_zero(r->min))
          << "Inferred range for CUDA thread has a non-zero min when passing down " << stage << " "
          << kv.first << " " << b_iv->var->name_hint << " " << r;
      UpdateShim(stage, p_state, b_iv, r, actx);
    }
  }
}

void PassUpIndex(const Stage& stage, const Map<IterVar, Range>& dom_map,
                 std::unordered_map<IterVar, PrimExpr>* p_state, bool allow_missing) {
  bool print = false;  // stage->op->name == "O";
  auto& state = *p_state;
  for (size_t i = stage->relations.size(); i != 0; --i) {
    IterVarRelation rel = stage->relations[i - 1];
    if (const SplitNode* s = rel.as<SplitNode>()) {
      if (!state.count(s->outer) || !state.count(s->inner)) {
        CHECK(allow_missing);
        continue;
      }
      PrimExpr outer = state.at(s->outer);
      PrimExpr inner = state.at(s->inner);
      PrimExpr factor = dom_map.at(s->inner)->extent;
      PrimExpr parent_min = dom_map.at(s->parent)->min;
      state[s->parent] = inner + outer * factor;
      // add min if they exist
      if (!is_zero(parent_min)) {
        state[s->parent] = state[s->parent] + parent_min;
      }
      if (print) {
        std::cout << "[PUI] Split O/I/P " << outer << " " << inner << " " << state[s->parent]
                  << std::endl;
      }
    } else if (const RaggedFuseNode* s = rel.as<RaggedFuseNode>()) {
      if (!state.count(s->fused)) {
        CHECK(allow_missing);
        continue;
      }
      PrimExpr fused_value = state.at(s->fused);
      state[s->outer] =
          zero_if_args_zero_ufun_call(s->outer->var.dtype(), {fused_value},
                                      s->fused_to_outer_uf->dimensions, s->fused_to_outer_uf);
      state[s->inner] =
          zero_if_args_zero_ufun_call(s->inner->var.dtype(), {fused_value},
                                      s->fused_to_inner_uf->dimensions, s->fused_to_inner_uf);
      if (print) {
        std::cout << "[PUI] RFuse P/O/I " << state[s->fused] << " " << state[s->outer] << " "
                  << state[s->inner] << std::endl;
      }
    } else if (const FuseNode* s = rel.as<FuseNode>()) {
      if (!state.count(s->fused)) {
        CHECK(allow_missing);
        continue;
      }
      PrimExpr value = state.at(s->fused);
      PrimExpr factor = dom_map.at(s->inner)->extent;
      PrimExpr outer_min = dom_map.at(s->outer)->min;
      PrimExpr inner_min = dom_map.at(s->inner)->min;
      state[s->outer] = indexdiv(value, factor);
      state[s->inner] = indexmod(value, factor);
      // add min if they exist
      if (!is_zero(outer_min)) {
        state[s->outer] = state[s->outer] + outer_min;
      }
      if (!is_zero(inner_min)) {
        state[s->inner] = state[s->inner] + inner_min;
      }
      if (print) {
        std::cout << "[PUI] Fuse P/O/I " << state[s->fused] << " " << state[s->outer] << " "
                  << state[s->inner] << std::endl;
      }
    } else if (const RebaseNode* s = rel.as<RebaseNode>()) {
      if (!state.count(s->rebased)) {
        CHECK(allow_missing);
        continue;
      }
      PrimExpr value = state.at(s->rebased);
      PrimExpr parent_min = dom_map.at(s->parent)->min;
      // add min if they exist
      if (!is_zero(parent_min)) {
        state[s->parent] = value + parent_min;
      } else {
        state[s->parent] = value;
      }
      if (print) {
        std::cout << "[PUI] Rebase C/P " << state[s->rebased] << " " << state[s->parent]
                  << std::endl;
      }
    } else if (rel.as<SingletonNode>()) {
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }

  std::unordered_map<const VarNode*, PrimExpr> vsub;
  for (auto it : state) {
    vsub[it.first->var.operator->()] = it.second;
  }
  VarReplacer replacer(vsub);
  for (auto it : state) {
    state[it.first] = replacer(it.second);
  }
}

void PassDownIndex(const Stage& stage, const Map<IterVar, Range>& dom_map,
                   std::unordered_map<IterVar, PrimExpr>* p_state, bool allow_missing) {
  auto& state = *p_state;
  for (IterVarRelation rel : stage->relations) {
    if (const SplitNode* s = rel.as<SplitNode>()) {
      if (!state.count(s->parent)) {
        CHECK(allow_missing);
        continue;
      }
      Range r = dom_map.at(s->inner);
      CHECK(is_zero(r->min));
      PrimExpr parent = state.at(s->parent);
      PrimExpr factor = r->extent;
      state[s->outer] = indexdiv(parent, factor);
      state[s->inner] = indexmod(parent, factor);
    } else if (const RaggedFuseNode* s = rel.as<RaggedFuseNode>()) {
      if (!state.count(s->inner) && !state.count(s->outer)) {
        CHECK(allow_missing);
        continue;
      }
      PrimExpr inner_value = state.at(s->inner);
      PrimExpr outer_value = state.at(s->outer);
      state[s->fused] = zero_if_args_zero_ufun_call(
          s->fused->var.dtype(), {outer_value, inner_value}, s->outer_inner_to_fused_uf->dimensions,
          s->outer_inner_to_fused_uf);
    } else if (const FuseNode* s = rel.as<FuseNode>()) {
      if (!state.count(s->inner) && !state.count(s->outer)) {
        CHECK(allow_missing);
        continue;
      }
      PrimExpr factor = dom_map.at(s->inner)->extent;
      PrimExpr outer_min = dom_map.at(s->outer)->min;
      PrimExpr inner_min = dom_map.at(s->inner)->min;
      PrimExpr inner = state.at(s->inner);
      PrimExpr outer = state.at(s->outer);
      CHECK(is_zero(outer_min)) << s->outer << " " << dom_map.at(s->outer);
      CHECK(is_zero(inner_min)) << s->inner << " " << dom_map.at(s->inner);
      state[s->fused] = outer * factor + inner;
    } else if (const RebaseNode* s = rel.as<RebaseNode>()) {
      if (!state.count(s->rebased)) {
        CHECK(allow_missing);
        continue;
      }
      PrimExpr value = state.at(s->parent);
      PrimExpr parent_min = dom_map.at(s->parent)->min;
      CHECK(is_zero(parent_min));
      state[s->rebased] = value;
    } else if (const SingletonNode* s = rel.as<SingletonNode>()) {
      state[s->iter] = make_zero(s->iter->var.dtype());
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

// Domain message passing.
void PassUpDomain(const SplitNode* s, const std::unordered_map<IterVar, Range>& dom_map,
                  const IntSet& outer, const IntSet& inner, IntSet* parent) {
  if (dom_map.count(s->outer) && dom_map.count(s->inner) && dom_map.count(s->parent) &&
      outer.match_range(dom_map.at(s->outer)) && inner.match_range(dom_map.at(s->inner))) {
    // std::cout << "[PUD] SplitResult " << dom_map.at(s->parent) << std::endl;
    *parent = IntSet::range(dom_map.at(s->parent));
    return;
  }
  PrimExpr factor = dom_map.at(s->inner)->extent;
  PrimExpr parent_min = dom_map.at(s->parent)->min;
  CHECK(outer.defined());
  CHECK(inner.defined());
  CHECK(factor.defined());
  // std::cout << "[PUD] SplitEvaling " << (s->outer->var * factor + s->inner->var + parent_min)
  // << std::endl;
  *parent = arith::EvalSet(s->outer->var * factor + s->inner->var + parent_min,
                           {{s->outer, outer}, {s->inner, inner}}, &dom_map);
}

void PassUpDomain(const FuseNode* s, const std::unordered_map<IterVar, Range>& dom_map,
                  const IntSet& fused, IntSet* outer, IntSet* inner) {
  CHECK(dom_map.count(s->outer));
  CHECK(dom_map.count(s->inner));
  CHECK(dom_map.count(s->fused));

  if (fused.match_range(dom_map.at(s->fused))) {
    *outer = IntSet::range(dom_map.at(s->outer));
    *inner = IntSet::range(dom_map.at(s->inner));
    return;
  }
  PrimExpr outer_min = dom_map.at(s->outer)->min;
  PrimExpr inner_min = dom_map.at(s->inner)->min;

  if (fused.is_single_point()) {
    PrimExpr value = fused.point_value();
    PrimExpr factor = dom_map.at(s->inner)->extent;
    PrimExpr v_outer = indexdiv(value, factor);
    PrimExpr v_inner = indexmod(value, factor);
    if (!is_zero(outer_min)) v_outer = v_outer + outer_min;
    if (!is_zero(inner_min)) v_inner = v_inner + inner_min;
    *outer = IntSet::single_point(v_outer);
    *inner = IntSet::single_point(v_inner);
  } else {
    PrimExpr fused_extent = (fused.max() - fused.min() + 1);
    PrimExpr inner_extent = dom_map.at(s->inner)->extent;
    *outer = IntSet::interval(outer_min + indexdiv(fused.min(), inner_extent),
                              outer_min + indexdiv(fused.max(), inner_extent));
    if (is_zero(
            Simplify(UninterpFun::InlineUninterpFunCalls(indexmod(inner_extent, fused_extent)))) &&
        is_zero(
            Simplify(UninterpFun::InlineUninterpFunCalls(indexmod(fused.min(), fused_extent))))) {
      // fused never spans multiple rows, make a tight bounding box
      // there may be other cases when bounding box could be tightened
      *inner = IntSet::interval(inner_min + indexmod(fused.min(), inner_extent),
                                inner_min + indexmod(fused.max(), inner_extent));
    } else {  // fused may span multiple rows, use full row widths
      if (!is_zero(Simplify(indexmod(fused_extent, inner_extent))) ||
          !is_zero(Simplify(indexmod(fused.min(), inner_extent)))) {
        LOG(WARNING) << "fused and original axes are not aligned, this may cause redundant "
                        "computations for "
                     << s->fused << " " << fused_extent << " " << inner_extent;
      }
      *inner = IntSet::range(dom_map.at(s->inner));
    }
    return;
  }
}

void PassUpDomain(const RaggedFuseNode* s, const std::unordered_map<IterVar, Range>& dom_map,
                  const IntSet& fused, IntSet* outer, IntSet* inner) {
  CHECK(dom_map.count(s->outer));
  CHECK(dom_map.count(s->inner));
  CHECK(dom_map.count(s->fused));

  if (fused.is_single_point()) {
    PrimExpr fused_val = fused.point_value();
    PrimExpr outer_val = zero_if_args_zero_ufun_call(
        s->outer->var.dtype(), {fused_val}, s->fused_to_outer_uf->dimensions, s->fused_to_outer_uf);
    *outer = IntSet::single_point(outer_val);
    PrimExpr inner_val = zero_if_args_zero_ufun_call(
        s->inner->var.dtype(), {fused_val}, s->fused_to_inner_uf->dimensions, s->fused_to_inner_uf);
    *inner = IntSet::single_point(inner_val);
  } else {
    PrimExpr fused_min = fused.min();
    PrimExpr fused_max_inclusive = fused.max();

    // std::cout << "[PUD] " << fused << " " << s->outer->dom << " " << s->inner->dom << std::endl;

    PrimExpr outer_min = zero_if_args_zero_ufun_call(
        s->outer->var.dtype(), {fused_min}, s->fused_to_outer_uf->dimensions, s->fused_to_outer_uf);
    PrimExpr outer_max_inclusive =
        zero_if_args_zero_ufun_call(s->outer->var.dtype(), {fused_max_inclusive},
                                    s->fused_to_outer_uf->dimensions, s->fused_to_outer_uf);
    *outer = IntSet::range(Range::make_by_min_max_inclusive(outer_min, outer_max_inclusive));

    PrimExpr inner_min_boundary = zero_if_args_zero_ufun_call(
        s->inner->var.dtype(), {fused_min}, s->fused_to_inner_uf->dimensions, s->fused_to_inner_uf);

    PrimExpr inner_max_inclusive_boundary =
        zero_if_args_zero_ufun_call(s->inner->var.dtype(), {fused_max_inclusive},
                                    s->fused_to_inner_uf->dimensions, s->fused_to_inner_uf);

    Range inner_range = dom_map.at(s->inner);

    *inner = IntSet::interval(
        FuseSelectNode::make(EQNode::make(s->outer, outer_min), inner_min_boundary,
                             inner_range->min, s->fused_to_inner_uf, fused_min),
        FuseSelectNode::make(EQNode::make(s->outer, outer_max_inclusive),
                             inner_max_inclusive_boundary, inner_range->max_inclusive(),
                             s->fused_to_inner_uf, fused_max_inclusive));
  }
}

void PassUpDomain(const RebaseNode* s, const std::unordered_map<IterVar, Range>& dom_map,
                  const IntSet& rebased, IntSet* parent) {
  CHECK(dom_map.count(s->parent));
  if (rebased.match_range(dom_map.at(s->rebased))) {
    *parent = IntSet::range(dom_map.at(s->parent));
    return;
  }
  PrimExpr parent_min = dom_map.at(s->parent)->min;
  *parent = arith::EvalSet(s->rebased->var + parent_min, {{s->rebased, rebased}});
}

void PassUpDomain(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                  std::unordered_map<IterVar, IntSet>* p_state) {
  bool print = false;  //(stage->op->name == "O.local");
  auto& state = *p_state;
  for (size_t i = stage->relations.size(); i != 0; --i) {
    IterVarRelation rel = stage->relations[i - 1];
    if (const SplitNode* r = rel.as<SplitNode>()) {
      IntSet parent;
      PassUpDomain(r, dom_map, state.at(r->outer), state.at(r->inner), &parent);
      state[r->parent] = parent;
      if (print) {
        std::cout << "[PUD] Outer " << state.at(r->outer) << std::endl;
        std::cout << "[PUD] Inner " << state.at(r->inner) << std::endl;
        std::cout << "[PUD] Parent " << parent << std::endl;
      }
    } else if (const RaggedFuseNode* r = rel.as<RaggedFuseNode>()) {
      IntSet outer, inner;
      PassUpDomain(r, dom_map, state.at(r->fused), &outer, &inner);
      state[r->outer] = outer;
      state[r->inner] = inner;
      // if (print) {
      // std::cout << "[PUD] RFused " << state.at(r->fused) << std::endl;
      // std::cout << "[PUD] RInner " << inner << std::endl;
      // std::cout << "[PUD] ROuter " << outer << std::endl;
      // }
    } else if (const FuseNode* r = rel.as<FuseNode>()) {
      IntSet outer, inner;
      PassUpDomain(r, dom_map, state.at(r->fused), &outer, &inner);
      state[r->outer] = outer;
      state[r->inner] = inner;
      // if (print) {
      // std::cout << "[PUD] Fused " << state.at(r->fused) << std::endl;
      // std::cout << "[PUD] Inner " << inner << std::endl;
      // std::cout << "[PUD] Outer " << outer << std::endl;
      // }
    } else if (const RebaseNode* r = rel.as<RebaseNode>()) {
      IntSet parent;
      PassUpDomain(r, dom_map, state.at(r->rebased), &parent);
      // if (print) std::cout << "[PUD] 3" << parent << " " << state.at(r->rebased) << std::endl;
      state[r->parent] = parent;
    } else if (rel.as<SingletonNode>()) {
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

// Pass up bit mask with or relation.
void PassUpBitMaskOr(const Stage& stage, std::unordered_map<IterVar, int>* p_state,
                     bool allow_missing) {
  auto& state = *p_state;
  for (size_t i = stage->relations.size(); i != 0; --i) {
    IterVarRelation rel = stage->relations[i - 1];
    if (const SplitNode* s = rel.as<SplitNode>()) {
      if (!state.count(s->inner) && !state.count(s->outer)) {
        CHECK(allow_missing) << s->inner->var << " " << s->outer->var;
        continue;
      }
      int res = 0;
      if (state.count(s->parent)) res |= state[s->parent];
      if (state.count(s->inner)) res |= state[s->inner];
      if (state.count(s->outer)) res |= state[s->outer];
      state[s->parent] = res;
    } else if (const RaggedFuseNode* s = rel.as<RaggedFuseNode>()) {
      if (!state.count(s->fused)) {
        CHECK(allow_missing);
        continue;
      }
      if (!state.count(s->outer)) {
        state[s->outer] = state[s->fused];
      } else {
        state[s->outer] |= state[s->fused];
      }
      if (!state.count(s->inner)) {
        state[s->inner] = state[s->fused];
      } else {
        state[s->inner] |= state[s->fused];
      }
    } else if (const FuseNode* s = rel.as<FuseNode>()) {
      if (!state.count(s->fused)) {
        CHECK(allow_missing);
        continue;
      }
      if (!state.count(s->outer)) {
        state[s->outer] = state[s->fused];
      } else {
        state[s->outer] |= state[s->fused];
      }
      if (!state.count(s->inner)) {
        state[s->inner] = state[s->fused];
      } else {
        state[s->inner] |= state[s->fused];
      }
    } else if (const RebaseNode* s = rel.as<RebaseNode>()) {
      if (!state.count(s->rebased)) {
        CHECK(allow_missing);
        continue;
      }
      if (!state.count(s->parent)) {
        state[s->parent] = state[s->rebased];
      } else {
        state[s->parent] |= state[s->rebased];
      }
    } else if (rel.as<SingletonNode>()) {
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

void PassDownBitMaskOr(const Stage& stage, std::unordered_map<IterVar, int>* p_state,
                       bool allow_missing) {
  auto& state = *p_state;
  for (IterVarRelation rel : stage->relations) {
    if (const SplitNode* s = rel.as<SplitNode>()) {
      if (!state.count(s->parent)) {
        CHECK(allow_missing);
        continue;
      }
      if (!state.count(s->outer)) {
        state[s->outer] = state.at(s->parent);
      } else {
        state[s->outer] |= state.at(s->parent);
      }
      if (!state.count(s->inner)) {
        state[s->inner] = state.at(s->parent);
      } else {
        state[s->inner] |= state.at(s->parent);
      }
    } else if (const RaggedFuseNode* s = rel.as<RaggedFuseNode>()) {
      if (!state.count(s->outer) && !state.count(s->inner)) {
        CHECK(allow_missing);
        continue;
      }
      int res = 0;
      if (state.count(s->outer)) res |= state.at(s->outer);
      if (state.count(s->inner)) res |= state.at(s->inner);
      if (state.count(s->fused)) res |= state.at(s->fused);
      state[s->fused] = res;
    } else if (const FuseNode* s = rel.as<FuseNode>()) {
      if (!state.count(s->outer) && !state.count(s->inner)) {
        CHECK(allow_missing);
        continue;
      }
      int res = 0;
      if (state.count(s->outer)) res |= state.at(s->outer);
      if (state.count(s->inner)) res |= state.at(s->inner);
      if (state.count(s->fused)) res |= state.at(s->fused);
      state[s->fused] = res;
    } else if (const RebaseNode* s = rel.as<RebaseNode>()) {
      if (!state.count(s->parent)) {
        CHECK(allow_missing);
        continue;
      }
      if (!state.count(s->rebased)) {
        state[s->rebased] = state.at(s->parent);
      } else {
        state[s->rebased] |= state.at(s->parent);
      }
    } else if (const SingletonNode* s = rel.as<SingletonNode>()) {
      state[s->iter] = 0;
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

/*!
 * \brief message passing to find if boundary checking on IterVar is needed.
 * \param s The stage to be used.
 * \param p_state The message passing state
 *     IterVar->flag
 */
void PassUpBoundCheck(const Stage& s, const Map<IterVar, Range>& dom_map,
                      std::unordered_map<IterVar, bool>* p_state, arith::Analyzer* analyzer) {
  std::unordered_map<const Object*, int> assumed_padding;
  for (auto rel : s->relations) {
    if (const RaggedFuseNode* s = rel.as<RaggedFuseNode>()) {
      if (s->assumed_fused_padding > 0) {
        assumed_padding[s->fused.get()] = s->assumed_fused_padding;
      }
    }
  }

  auto& state = *p_state;
  for (size_t i = s->relations.size(); i != 0; --i) {
    IterVarRelation rel = s->relations[i - 1];
    if (const SplitNode* s = rel.as<SplitNode>()) {
      bool outer = state.at(s->outer);
      bool inner = state.at(s->inner);

      if (dom_map.count(s->inner) && dom_map.count(s->outer)) {
        PrimExpr factor = dom_map.at(s->inner)->extent;
        PrimExpr step = dom_map.at(s->outer)->extent;
        if (outer || inner) {
          state[s->parent] = true;
        } else {
          // Very bad way of letting the analyzer know that sequence
          // lengths are padded to multiple of some factor
          PrimExpr to_prove1 = dom_map.at(s->parent)->extent == factor * step;
          PrimExpr to_prove2 = Simplify(UninterpFun::InlineUninterpFunCalls(to_prove1));
          if (analyzer->CanProve(to_prove1) || analyzer->CanProve(to_prove2)) {
            state[s->parent] = false;
          } else {
            auto it = assumed_padding.find(s->parent.get());
            if (it != assumed_padding.end()) {
              std::cout << "[PBC] Found assumed padding and proved bound unnecessary for " << s
                        << " " << s->parent << std::endl;
              state[s->parent] = !analyzer->CanProve(floormod(it->second, factor) == 0);
            } else {
              state[s->parent] = true;
            }
          }
        }
      } else {
        state[s->parent] = true;
      }
    } else if (const RaggedFuseNode* s = rel.as<RaggedFuseNode>()) {
      bool fused = state.at(s->fused);
      state[s->outer] = fused;
      state[s->inner] = fused;
    } else if (const FuseNode* s = rel.as<FuseNode>()) {
      bool fused = state.at(s->fused);
      state[s->outer] = fused;
      state[s->inner] = fused;
    } else if (const RebaseNode* s = rel.as<RebaseNode>()) {
      state[s->parent] = state.at(s->rebased);
    } else if (rel.as<SingletonNode>()) {
      // nop
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

std::unordered_set<std::string> CollectDependentCudaVars(
    const Stage& stage, const Map<IterVar, Range>& dom_map,
    const std::unordered_map<const VarNode*, std::string>& bind_map,
    const std::unordered_map<IterVar, PrimExpr>& value_map) {
  VarCollector collector;
  for (auto riv : stage->op->root_iter_vars()) {
    if (dom_map.count(riv) && value_map.count(riv)) {
      Range r = dom_map.at(riv);
      collector.collect(r);
    }
  }
  auto vars = collector.getCollected();
  for (auto riv : stage->op->root_iter_vars()) {
    vars.insert(riv->var.as<VarNode>());
  }
  std::unordered_set<std::string> ret;
  for (auto var : vars) {
    if (bind_map.count(var)) ret.insert(bind_map.at(var));
  }
  return ret;
}

PrimExpr implies(PrimExpr a, PrimExpr b) { return OrNode::make(NotNode::make(a), b); }

void AddConstraintsToAnalyzer(const Stage& stage, const Map<IterVar, Range>& dom_map,
                              const std::unordered_map<std::string, Range>& env_dom_map,
                              const std::unordered_map<std::string, IterVar>& env_var_map,
                              const std::unordered_map<const VarNode*, std::string>& bind_map,
                              const std::unordered_map<IterVar, PrimExpr>& value_map,
                              const std::unordered_map<IterVar, bool>& bound_state,
                              const Map<Stage, Array<Stage>>& attach_stages,
                              const Map<Stage, Array<IterVar>>& attach_vars, bool attach_stage,
                              arith::Analyzer* p_analyzer, bool print) {
  // print = false;
  arith::Analyzer& analyzer = *p_analyzer;
  if (print) std::cout << "[MBC] Adding constraints for stage " << stage << std::endl;

  // For all itervars in the stage, add their inferred ranges
  if (print) std::cout << "[MBC] Adding itervar range constraints" << std::endl;
  auto add_range_constraint = [&](Var v, Range r) {
    analyzer.AddConstraint(v >= r->min);
    analyzer.AddConstraint(v <= r->max_inclusive());
  };
  for (auto iv : stage->all_iter_vars) {
    if (!bound_state.at(iv)) {
      if (dom_map.count(iv)) {
        // if (print) std::cout << "[MBC]  IV1 " << iv->var << " " << dom_map.at(iv) << std::endl;
        add_range_constraint(iv->var, dom_map.at(iv));
      } else {
        if (iv->dom.defined()) {
          // if (print) std::cout << "[MBC]  IV2 " << iv->var << " " << iv->dom << std::endl;
          add_range_constraint(iv->var, iv->dom);
        }
      }
    }
  }

  // Add the relations between the different itervars associated
  // with this stage
  if (print) std::cout << "[MBC] Adding itervar relation constraints" << std::endl;
  std::unordered_map<IterVar, PrimExpr> state;
  for (auto iv : stage->leaf_iter_vars) {
    state[iv] = iv->var;
  }
  PassUpIndex(stage, dom_map, &state, false);
  for (auto it : state) {
    if (it.first->var.same_as(it.second)) continue;
    analyzer.AddConstraint(it.first->var == it.second);
  }

  // For all fi and associated l_fun l, fi(f) < l(o), fi >= 0, fo >= 0, oif >= 0
  if (print) std::cout << "[MBC] Adding ragged fusion relation constraints" << std::endl;
  auto add_non_neg_constraint = [&](UninterpFun uf) {
    analyzer.AddForallConstraint(uf->parameters,
                                 uf.MakeCallTo(uf->parameters, uf->dimensions) >= 0);
  };
  for (auto rel : stage->relations) {
    if (auto frel = rel.as<RaggedFuseNode>()) {
      if (stage.is_ancestor_attached_at_root()) {
        if (dom_map.count(frel->inner)) {
          analyzer.AddConstraint(
              dom_map.at(frel->inner)->max_exclusive() >
              frel->fused_to_inner_uf.MakeCallTo(Array<PrimExpr>({frel->fused->var}),
                                                 frel->fused_to_inner_uf->dimensions));
        }
      }
      if (frel->inner->dom.defined()) {
        analyzer.AddConstraint(
            frel->inner->dom->max_exclusive() >
            frel->fused_to_inner_uf.MakeCallTo(Array<PrimExpr>({frel->fused->var}),
                                               frel->fused_to_inner_uf->dimensions));
      }
      add_non_neg_constraint(frel->fused_to_outer_uf);
      add_non_neg_constraint(frel->fused_to_inner_uf);
      add_non_neg_constraint(frel->outer_inner_to_fused_uf);

      // Monotonicity constraints for fo/fi functions
      {
        Var f1 = Var("f1", DataType::Int(32));
        Var f2 = Var("f2", DataType::Int(32));
        auto fo1 = frel->fused_to_outer_uf.MakeCallTo(Array<Var>({f1}),
                                                      frel->fused_to_outer_uf->dimensions);
        auto fo2 = frel->fused_to_outer_uf.MakeCallTo(Array<Var>({f2}),
                                                      frel->fused_to_outer_uf->dimensions);
        analyzer.AddForallConstraint({f1, f2}, implies(f1 <= f2, fo1 <= fo2));

        auto fi1 = frel->fused_to_inner_uf.MakeCallTo(Array<Var>({f1}),
                                                      frel->fused_to_inner_uf->dimensions);
        auto fi2 = frel->fused_to_inner_uf.MakeCallTo(Array<Var>({f2}),
                                                      frel->fused_to_inner_uf->dimensions);
        analyzer.AddForallConstraint(
            {f1, f2},
            implies(f1 >= 0 && f2 >= 0 && f1 <= f2 && EQNode::make(fo1, fo2), fi1 <= fi2));
      }

      // f = oif(fo(f), fi(f))
      {
        Var f = Var("f", DataType::Int(32));
        auto fo = frel->fused_to_outer_uf.MakeCallTo(Array<Var>({f}),
                                                     frel->fused_to_outer_uf->dimensions);
        auto fi = frel->fused_to_inner_uf.MakeCallTo(Array<Var>({f}),
                                                     frel->fused_to_inner_uf->dimensions);
        auto oif = frel->outer_inner_to_fused_uf.MakeCallTo(
            Array<PrimExpr>({fo, fi}), frel->outer_inner_to_fused_uf->dimensions);
        analyzer.AddForallConstraint({f}, implies(f >= 0, EQNode::make(f, oif)));
      }

      // Padding constraints, if specified
      if (dom_map.count(frel->fused) && frel->assumed_fused_padding > 0) {
        analyzer.AddConstraint(EQNode::make(
            FloorModNode::make(dom_map.at(frel->fused)->extent, frel->assumed_fused_padding), 0));
      }
    }
  }

  // For all l_funs in the stage, add positivity and padding
  // constraints
  if (print) std::cout << "[MBC] Adding l_fun constraints" << std::endl;
  auto add_l_fun_constraints = [&](UninterpFun lf) {
    if (lf->arity() > 0) {
      Array<PrimExpr> args;
      PrimExpr args_positive = IntImm(DataType::Bool(), 1);
      for (auto param : lf->parameters) {
        args.push_back(param);
        args_positive = args_positive && (param >= 0);
      }
      auto call = lf.MakeCallTo(args, lf->dimensions);
      analyzer.AddForallConstraint(lf->parameters, (call == lf->body));
      // analyzer.AddForallConstraint(lf->parameters, implies(args_positive, call > 0));
      analyzer.AddForallConstraint(lf->parameters, implies(args_positive, call >= lf->range->min));
    } else {
      auto call = lf.MakeCallTo(Array<Var>({}), {});
      analyzer.AddConstraint(call == lf->body);
      // analyzer.AddConstraint(call > 0);
      analyzer.AddConstraint(call >= lf->range->min);
    }
  };
  if (stage->op->loop_layout().defined()) {
    for (auto lf : stage->op->loop_layout()->l_funs) {
      add_l_fun_constraints(lf);
    }
    for (auto lf : stage->op->loop_layout()->l_fun_mins) {
      add_l_fun_constraints(lf);
    }
  } else {
    for (auto iv : stage->op->root_iter_vars()) {
      const CallNode* call;
      if (iv->dom.defined() && (call = iv->dom->extent.as<CallNode>())) {
        if (call->func.defined() && call->func.as<UninterpFunNode>()) {
          add_l_fun_constraints(Downcast<UninterpFun>(call->func));
        }
      }
    }
  }

  if (!attach_stage) {
    struct FuncTriple {
      UninterpFun fused_to_outer_uf;
      UninterpFun fused_to_inner_uf;
      UninterpFun outer_inner_to_fused_uf;
    };

    struct pair_hash {
      std::size_t operator()(const std::pair<const Object*, const Object*>& pair) const {
        return std::hash<const Object*>()(pair.first) ^ std::hash<const Object*>()(pair.second);
      }
    };

    std::unordered_set<std::pair<const Object*, const Object*>, pair_hash> already_inserted;
    auto add_equality_constraint = [&](UninterpFun uf1, UninterpFun uf2) {
      if (uf1 == uf2) return;
      auto key = std::make_pair(uf1.get(), uf2.get());
      if (already_inserted.count(key)) {
        return;
      }
      already_inserted.insert(key);
      CHECK_EQ(uf1->arity(), uf2->arity());
      analyzer.AddForallConstraint(uf1->parameters,
                                   EQNode::make(uf1.MakeCallTo(uf1->parameters, uf1->dimensions),
                                                uf2.MakeCallTo(uf1->parameters, uf2->dimensions)));
    };

    std::unordered_map<const Object*, FuncTriple> fused_fun_map;
    auto handle_rel = [&](Dimension fused_dim, UninterpFun fused_to_outer_uf,
                          UninterpFun fused_to_inner_uf, UninterpFun outer_inner_to_fused_uf) {
      auto it = fused_fun_map.find(fused_dim.get());
      if (it != fused_fun_map.end()) {
        auto& funs = it->second;
        add_equality_constraint(fused_to_outer_uf, funs.fused_to_outer_uf);
        add_equality_constraint(fused_to_inner_uf, funs.fused_to_inner_uf);
        add_equality_constraint(outer_inner_to_fused_uf, funs.outer_inner_to_fused_uf);
      } else {
        fused_fun_map[fused_dim.get()] = {fused_to_outer_uf, fused_to_inner_uf,
                                          outer_inner_to_fused_uf};
      }
    };

    auto handle_stage = [&](Stage s) {
      for (auto rel : s->relations) {
        if (auto frel = rel.as<RaggedFuseNode>()) {
          handle_rel(frel->fused_to_outer_uf->dimensions[0], frel->fused_to_outer_uf,
                     frel->fused_to_inner_uf, frel->outer_inner_to_fused_uf);
        }
      }

      if (s->dim_relation_graph.defined()) {
        for (auto rel : s->dim_relation_graph->relations) {
          if (auto frel = rel.as<RaggedDimensionFuseNode>()) {
            handle_rel(frel->fused_to_outer_uf->dimensions[0], frel->fused_to_outer_uf,
                       frel->fused_to_inner_uf, frel->outer_inner_to_fused_uf);
          }
        }
      }
    };

    for (auto it : attach_stages) {
      handle_stage(it.first);
      for (auto it : it.second) {
        handle_stage(it);
      }
    }

    // Add constraints stating tha a leaf iv and the thread iv it is
    // bound to, if any, are equal
    if (print) std::cout << "[MBC] Adding itervar binding constraints" << std::endl;
    for (auto it : attach_stages) {
      for (auto kv : it.first->iter_var_attrs) {
        if (kv.second->bind_thread.defined()) {
          analyzer.AddConstraint(EQNode::make(kv.first->var, kv.second->bind_thread->var));
        }
      }
    }

    // Add ranges for env_vars
    if (print) std::cout << "[MBC] Adding env var constraints" << std::endl;
    for (auto it : env_var_map) {
      add_range_constraint(it.second->var, env_dom_map.at(it.first));
    }

    // Add value map constraints to include the new .init variables
    // generated during the initialization phase of ops containing
    // reductions
    if (print) std::cout << "[MBC] Adding value map constraints" << std::endl;
    for (auto it : value_map) {
      analyzer.AddConstraint(it.first->var == it.second);
    }
  }

  if (!attach_stage && attach_stages.count(stage) && attach_vars.count(stage)) {
    auto this_attach_stages = attach_stages.at(stage);
    auto this_attach_iters = attach_vars.at(stage);

    Stage previous_stage = NullValue<Stage>();
    for (size_t i = 0; i < this_attach_stages.size(); ++i) {
      if (previous_stage == this_attach_stages[i]) continue;

      std::unordered_map<IterVar, bool> attach_stage_bound_state;
      for (IterVar iv : this_attach_stages[i]->leaf_iter_vars) {
        attach_stage_bound_state[iv] = false;
      }
      PassUpBoundCheck(this_attach_stages[i], dom_map, &attach_stage_bound_state, &analyzer);

      AddConstraintsToAnalyzer(this_attach_stages[i], dom_map, env_dom_map, env_var_map, bind_map,
                               value_map, attach_stage_bound_state, attach_stages, attach_vars,
                               true, p_analyzer, print);
      previous_stage = this_attach_stages[i];
    }
  }
}

std::vector<PrimExpr> MakeBoundCheck(
    const Stage& stage, const Map<IterVar, Range>& dom_map,
    const std::unordered_map<std::string, Range>& env_dom_map,
    const std::unordered_map<std::string, IterVar>& env_var_map,
    const std::unordered_map<const VarNode*, std::string>& bind_map,
    const std::unordered_map<IterVar, PrimExpr>& value_map, bool skip_ivar_domain,
    const std::unordered_set<IterVar>& skip_iter, const Map<Stage, Array<Stage>>& attach_stages,
    const Map<Stage, Array<IterVar>>& attach_vars) {
  arith::Analyzer analyzer;

  bool print = false;
  // bool print = (stage->op->name == "A.shared");
  if (print) {
    std::cout << "[MBC] Genning bounds check for " << stage->op << std::endl;
  }
  if (stage->no_bounds_check) {
    // std::cout << "[BOUNDS] Skipping bounds check for " << stage->op << std::endl;
    return {};
  }
  std::unordered_map<const VarNode*, PrimExpr> vsub_map;
  // if (print)
  //   std::cout << "[CHECK] Op " << stage->op << " " << stage->storage_scope_rank << " "
  //             << bind_map.size() << std::endl;
  for (auto it : value_map) {
    vsub_map[it.first->var.as<VarNode>()] = it.second;
  }

  for (auto it : bind_map) {
    if (env_var_map.count(it.second)) {
      vsub_map[it.first] = env_var_map.at(it.second)->var;
    }
  }

  VarReplacer replacer(vsub_map);
  auto process_pred = [&](PrimExpr pred) {
    return tir::Simplify(replacer(UninterpFun::InlineUninterpFunCalls(pred)));
  };

  std::unordered_map<IterVar, bool> bound_state;
  for (IterVar iv : stage->leaf_iter_vars) {
    bound_state[iv] = false;
  }
  PassUpBoundCheck(stage, dom_map, &bound_state, &analyzer);

  // Add the necessary constraints to the analyzers
  AddConstraintsToAnalyzer(stage, dom_map, env_dom_map, env_var_map, bind_map, value_map,
                           bound_state, attach_stages, attach_vars, false, &analyzer, print);

  std::vector<PrimExpr> preds;
  std::unordered_map<const VarNode*, IntSet> iset_dmap;

  std::unordered_map<const VarNode*, PrimExpr> value_vsub_map;
  // setup domain map for set analysis
  for (const auto& kv : dom_map) {
    if (isCudaThread(kv.first) && env_dom_map.count(kv.first->thread_tag)) {
      CHECK(env_var_map.count(kv.first->thread_tag)) << kv.first->var->name_hint;
      iset_dmap[env_var_map.at(kv.first->thread_tag)->var.get()] =
          IntSet::range(env_dom_map.at(kv.first->thread_tag));
      value_vsub_map[kv.first->var.get()] = env_var_map.at(kv.first->thread_tag);
    } else {
      iset_dmap[kv.first->var.get()] = IntSet::range(kv.second);
    }
  }

  // PPF: Now that the domains of bound thread vars may be larger than
  // those of the original ones, we need to add conditionals to skip
  // computation when the thread var bounds exceed the original var
  // bounds.
  std::unordered_set<std::string> generated_env_checks;
  for (auto kv : stage->iter_var_attrs) {
    if (kv.second->bind_thread.defined()) {
      IterVar original_var = kv.first;
      IterVar bound_thread_var = kv.second->bind_thread;
      Range original_range = dom_map[original_var];
      Range bound_thread_range = NullValue<Range>();
      if (print)
        std::cout << "[CHECK] Visiting " << original_var << " " << bound_thread_var << std::endl;
      if (env_dom_map.count(bound_thread_var->thread_tag)) {
        bound_thread_range = env_dom_map.at(bound_thread_var->thread_tag);
      } else {
        bound_thread_range = dom_map[bound_thread_var];
      }
      generated_env_checks.insert(bound_thread_var->thread_tag);
      bool can_avoid_check =
          analyzer.CanProve(bound_thread_range->extent == original_range->extent);
      if (print) std::cout << "[CHECK]    " << original_range << std::endl;
      if (print) std::cout << "[CHECK]    " << bound_thread_range << std::endl;
      if (print) std::cout << "[CHECK]    " << can_avoid_check << std::endl;
      if (!can_avoid_check) {
        auto check = process_pred(bound_thread_var->var < original_range->extent);
        if (print) std::cout << "[CHECK]   Adding check " << check << std::endl;
        preds.emplace_back(check);
      }
    }
  }

  if (stage->op.as<ComputeOpNode>()) {
    auto cudaVars = CollectDependentCudaVars(stage, dom_map, bind_map, value_map);
    for (auto it : env_var_map) {
      if (!generated_env_checks.count(it.first) && !cudaVars.count(it.first)) {
        tvm::runtime::ThreadScope ts = tvm::runtime::ThreadScope::make(it.first);
        if (stage->storage_scope_rank <= ts.rank) {
          if (print) std::cout << "[CHECK2]   " << (it.second->var < 1) << std::endl;
          preds.emplace_back(process_pred(it.second->var < 1));
        }
      }
    }
  }

  for (const IterVar& iv : stage->all_iter_vars) {
    if (skip_iter.count(iv) || iv->iter_type == kOpaque || iv->iter_type == kLoopNestOpaque)
      continue;
    if (bound_state.at(iv)) {
      Range dom = dom_map.at(iv);
      PrimExpr value = Simplify(value_map.at(iv) - dom->min);
      PrimExpr extent = Simplify(dom->extent);
      PrimExpr vmax = EvalSet(value, iset_dmap).max();
      bool print2 = print;  // && (iv->var->name_hint == "iO_13_f");

      if (print2) {
        std::cout << "[CHECK3]   IV: " << iv->var << std::endl;
        std::cout << "[CHECK3]    Extent: " << extent << std::endl;
        std::cout << "[CHECK3]    Value:  " << value << std::endl;
        // std::cout << "[CHECK3]    VMax:   " << vmax << std::endl;
      }
      bool can_avoid_check = analyzer.CanProve(value < extent);
      if (print2) {
        std::cout << "[CHECK3]     Result: " << can_avoid_check << std::endl;
      }
      if (!can_avoid_check) {
        preds.emplace_back(process_pred(value < extent));
      }
    }
  }

  VarReplacer value_replacer(value_vsub_map);
  for (const IterVar& iv : stage->op->root_iter_vars()) {
    if (skip_iter.count(iv) || iv->iter_type == kOpaque || iv->iter_type == kLoopNestOpaque)
      continue;
    Range dom = dom_map.at(iv);
    CHECK(iv->dom.defined());

    bool ivar_domain_same = iv->dom->min.same_as(dom->min) && iv->dom->extent.same_as(dom->extent);

    if (!skip_ivar_domain && !ivar_domain_same) {
      PrimExpr value = replacer(value_map.at(iv) - iv->dom->min);
      IntSet s = EvalSet(value, iset_dmap);
      PrimExpr vmin = s.min();
      PrimExpr vmax = s.max();

      if (print) {
        std::cout << "[CHECK6]   IV: " << iv << std::endl;
        std::cout << "[CHECK6]    Extent: " << iv->dom->extent << std::endl;
        std::cout << "[CHECK6]    Value:  " << value << std::endl;
      }

      // The range of `value` resides in [vmin, vmax]
      bool can_avoid_check_min = analyzer.CanProveGreaterEqual(value, 0);
      if (print) {
        std::cout << "[CHECK6]     MinResult:   " << can_avoid_check_min << std::endl;
      }
      if (!can_avoid_check_min) {
        preds.emplace_back(process_pred(value >= 0));
      }
      bool can_avoid_check_max =
          analyzer.CanProveGreaterEqual(Simplify(iv->dom->extent - value - 1), 0);
      if (print) {
        std::cout << "[CHECK6]     MaxResult:   " << can_avoid_check_max << std::endl;
      }
      if (!can_avoid_check_max) {
        preds.emplace_back(process_pred(value < iv->dom->extent));
      }
    }
  }

  // Dedup for readibility
  std::vector<PrimExpr> ret;
  for (const auto& pred : preds) {
    bool repeated = false;
    for (const auto& fin : ret) {
      if (analyzer.CanProve(UninterpFun::InlineUninterpFunCalls(EQNode::make(pred, fin)))) {
        repeated = true;
      }
    }
    if (!repeated) {
      if (print) std::cout << "[PUSHING] " << pred << std::endl;
      ret.push_back(pred);
    } else {
      if (print) std::cout << "[REDUNDANT] " << pred << std::endl;
    }
  }
  return ret;
}

/* Dimensions */
void DimensionPassDownValues(Stage s, const BaseVarDimOpNode* op,
                             const std::unordered_map<const DimensionNode*, Range>& dom_map,
                             std::unordered_map<const DimensionNode*, PrimExpr>* p_state,
                             bool allow_missing) {
  bool print = false;  // s->op->name == "is_h2h.ila";
  const DimensionRelationGraph& graph = s->dim_relation_graph;
  // std::cout << "[PDD] Passing down values " << graph->relations.size() << std::endl;
  auto& state = *p_state;
  for (DimensionRelation rel : graph->relations) {
    if (const DimensionSplitNode* s = rel.as<DimensionSplitNode>()) {
      if (!state.count(s->parent.operator->())) {
        CHECK(allow_missing);
        continue;
      }
      PrimExpr parent = state.at(s->parent.operator->());
      PrimExpr factor = s->factor;
      if (print) {
        std::cout << "[PDD]  Parent " << s->parent->name << " " << parent << std::endl;
        std::cout << "[PDD]    Inner " << s->inner->name << " " << indexdiv(parent, factor)
                  << std::endl;
        std::cout << "[PDD]    Outer " << s->outer->name << " " << indexmod(parent, factor)
                  << std::endl;
      }
      state[s->outer.operator->()] = indexdiv(parent, factor);
      state[s->inner.operator->()] = indexmod(parent, factor);
    } else if (const RaggedDimensionFuseNode* s = rel.as<RaggedDimensionFuseNode>()) {
      if (!state.count(s->inner.operator->()) && !state.count(s->outer.operator->())) {
        CHECK(allow_missing);
        continue;
      };

      PrimExpr inner_value = state.at(s->inner.operator->());
      PrimExpr outer_value = state.at(s->outer.operator->());
      state[s->fused.operator->()] = zero_if_args_zero_ufun_call(
          DataType::Int(32), {outer_value, inner_value}, s->outer_inner_to_fused_uf->dimensions,
          s->outer_inner_to_fused_uf);
    } else if (const DimensionFuseNode* s = rel.as<DimensionFuseNode>()) {
      if (!state.count(s->inner.operator->()) && !state.count(s->outer.operator->())) {
        CHECK(allow_missing);
        continue;
      };
      // std::cout << "[DPDV] IV " << s->inner << std::endl;
      PrimExpr factor = dom_map.at(s->inner.operator->())->extent;
      // PrimExpr outer_min = dom_map.at(s->outer.operator->())->min;
      // PrimExpr inner_min = dom_map.at(s->inner.operator->())->min;

      // PrimExpr factor = s->factor;
      PrimExpr inner = state.at(s->inner.operator->());
      PrimExpr outer = state.at(s->outer.operator->());
      state[s->fused.operator->()] = outer * factor + inner;
    } else {
      LOG(FATAL) << "unknown dimension relation type";
    }
  }
}

void Update(std::unordered_map<const DimensionNode*, Range>* p_state, const Dimension& iv, Range r,
            arith::Analyzer& analyzer) {
  auto it = p_state->find(iv.operator->());
  if (it == p_state->end()) {
    (*p_state)[iv.operator->()] = r;
  } else {
    bool match = is_zero(it->second->min) && analyzer.CanProve(r->extent - it->second->extent == 0);
    CHECK(match) << iv << " domain already inferred,"
                 << " cannot prove their extents are the same " << it->second->extent << " vs "
                 << r->extent;
  }
}

void DimensionPassDownDomain(Stage s, const BaseVarDimOpNode* op,
                             std::unordered_map<const DimensionNode*, Range>* p_state,
                             bool allow_missing) {
  // std::cout << "[DPDD] Stage " << s << std::endl;
  const DimensionRelationGraph& graph = s->dim_relation_graph;
  arith::Analyzer analyzer;
  auto ceil_div = [&analyzer](PrimExpr a, PrimExpr b) {
    if (analyzer.CanProve(indexmod(a, b) == 0)) {
      return analyzer.Simplify(indexdiv(a, b));
    }
    return analyzer.Simplify(indexdiv(a + (b - 1), b));
  };

  auto& state = *p_state;
  // forwar iteration on relations
  for (DimensionRelation rel : graph->relations) {
    if (const DimensionSplitNode* r = rel.as<DimensionSplitNode>()) {
      if (!state.count(r->parent.operator->())) {
        CHECK(allow_missing);
        continue;
      }
      CHECK(!state.count(r->inner.operator->()));
      const Range& range_parent = state.at(r->parent.operator->());
      if (r->factor.defined()) {
        Update(p_state, r->inner, Range::make_by_min_extent(0, r->factor), analyzer);
        Update(p_state, r->outer,
               Range::make_by_min_extent(ceil_div(range_parent->min, r->factor),
                                         ceil_div(range_parent->extent, r->factor)),
               analyzer);
      } else {
        CHECK(false);
        Update(p_state, r->outer, Range::make_by_min_extent(0, r->nparts), analyzer);

        Update(p_state, r->inner,
               Range::make_by_min_extent(ceil_div(range_parent->min, r->nparts),
                                         ceil_div(range_parent->extent, r->nparts)),
               analyzer);
      }
    } else if (const RaggedDimensionFuseNode* r = rel.as<RaggedDimensionFuseNode>()) {
      // std::cout << "[DPDD]  Ragged" << std::endl;
      if (!state.count(r->outer.operator->()) || !state.count(r->inner.operator->())) {
        CHECK(allow_missing);
        continue;
      }
      const Range& range_outer = state.at(r->outer.operator->());
      const Range& range_inner_unreplaced = state.at(r->inner.operator->());
      // std::cout << "[DPDD] Outer " << range_outer << std::endl;
      // std::cout << "[DPDD] Inner " << range_inner_unreplaced << std::endl;

      std::unordered_map<const VarNode*, PrimExpr> vsub_min;
      std::unordered_map<const VarNode*, PrimExpr> vsub_max;
      {
        for (auto di : op->GetAllDimensions()) {
          auto dim = di->dim;
          if (state.count(dim.operator->())) {
            auto range = state.at(dim.operator->());
            vsub_min[op->GetIterVarFromDim(0, dim)->var.as<VarNode>()] = range->min;
            vsub_max[op->GetIterVarFromDim(0, dim)->var.as<VarNode>()] = range->max_inclusive();
          }
        }
      }

      Range range_inner = Range::make_by_min_max_inclusive(
          VarReplacer(vsub_min)(range_inner_unreplaced->min),
          VarReplacer(vsub_max)(range_inner_unreplaced->max_inclusive()));

      // std::cout << "[DPDD]   UF " << r->outer_inner_to_fused_uf.defined() << std::endl;
      auto fused_min = zero_if_args_zero_ufun_call(
          DataType::Int(32), {range_outer->min, range_inner->min},
          r->outer_inner_to_fused_uf->dimensions, r->outer_inner_to_fused_uf);
      auto fused_max_inclusive = Simplify(zero_if_args_zero_ufun_call(
          DataType::Int(32), {range_outer->max_inclusive(), range_inner->max_inclusive()},
          r->outer_inner_to_fused_uf->dimensions, r->outer_inner_to_fused_uf));

      state[r->fused.operator->()] =
          Range::make_by_min_max_inclusive(fused_min, fused_max_inclusive);
    } else if (const DimensionFuseNode* r = rel.as<DimensionFuseNode>()) {
      std::cout << "[DPDD]  Dense" << std::endl;
      if (!state.count(r->outer.operator->()) || !state.count(r->inner.operator->())) {
        CHECK(allow_missing);
        continue;
      }
      const Range& range_outer = state.at(r->outer.operator->());
      const Range& range_inner = state.at(r->inner.operator->());
      state[r->fused.operator->()] =
          Range::make_by_min_extent(0, range_outer->extent * range_inner->extent);
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

// Pass up bit mask with or relation.
void DimensionPassUpBitMaskOr(const Stage& stage,
                              std::unordered_map<const DimensionNode*, int>* p_state,
                              bool allow_missing) {
  auto& state = *p_state;
  for (size_t i = stage->dim_relation_graph->relations.size(); i != 0; --i) {
    DimensionRelation rel = stage->dim_relation_graph->relations[i - 1];
    if (const DimensionSplitNode* s = rel.as<DimensionSplitNode>()) {
      if (!state.count(s->inner.operator->()) && !state.count(s->outer.operator->())) {
        CHECK(allow_missing) << s->inner.operator->() << " " << s->outer;
        continue;
      }
      int res = 0;
      if (state.count(s->parent.operator->())) res |= state[s->parent.operator->()];
      if (state.count(s->inner.operator->())) res |= state[s->inner.operator->()];
      if (state.count(s->outer.operator->())) res |= state[s->outer.operator->()];
      state[s->parent.operator->()] = res;
    } else if (const DimensionFuseNode* s = rel.as<DimensionFuseNode>()) {
      if (!state.count(s->fused.operator->())) {
        CHECK(allow_missing);
        continue;
      }
      if (!state.count(s->outer.operator->())) {
        state[s->outer.operator->()] = state[s->fused.operator->()];
      } else {
        state[s->outer.operator->()] |= state[s->fused.operator->()];
      }
      if (!state.count(s->inner.operator->())) {
        state[s->inner.operator->()] = state[s->fused.operator->()];
      } else {
        state[s->inner.operator->()] |= state[s->fused.operator->()];
      }
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

void DimensionPassUpBitMaskExact(const Stage& stage,
                                 std::unordered_set<const DimensionNode*>* p_state,
                                 bool* p_exact_possible) {
  auto& state = *p_state;
  auto& exact_possible = *p_exact_possible;

  for (size_t i = stage->dim_relation_graph->relations.size(); i != 0; --i) {
    DimensionRelation rel = stage->dim_relation_graph->relations[i - 1];
    if (const DimensionSplitNode* s = rel.as<DimensionSplitNode>()) {
      bool inner_present = state.count(s->inner.operator->());
      bool outer_present = state.count(s->outer.operator->());
      if (inner_present && outer_present) {
        state.insert(s->parent.operator->());
      } else if ((inner_present && !outer_present) || (!inner_present && outer_present)) {
        exact_possible = false;
        return;
      }
    } else if (const DimensionFuseNode* s = rel.as<DimensionFuseNode>()) {
      if (state.count(s->fused.operator->())) {
        state.insert(s->outer.operator->());
        state.insert(s->inner.operator->());
      }
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

Modes DimensionPassDownModes(Stage& stage, const BaseVarDimOpNode* compute_op,
                             // const std::unordered_map<const DimensionNode*, Range>& dom_map,
                             const Modes& root_layout) {
  if (stage->dim_relation_graph->relations.size() == 0) {
    return root_layout;
  }
  std::unordered_map<const DimensionNode*, UninterpFun> l_funs;
  for (size_t i = 0; i < root_layout->ndim(); ++i) {
    l_funs[root_layout->dimensions[i].operator->()] = root_layout->l_funs[i];
  }

  for (size_t i = stage->dim_relation_graph->relations.size(); i != 0; --i) {
    DimensionRelation rel = stage->dim_relation_graph->relations[i - 1];
    if (const DimensionSplitNode* s = rel.as<DimensionSplitNode>()) {
      CHECK(l_funs.count(s->parent.operator->()));
      UninterpFun parent_fun = l_funs.at(s->parent.operator->());
      UninterpFun inner_fun = UninterpFunNode::from_constant(s->inner->name + "_luf", s->factor);
      UninterpFun outer_fun =
          UninterpFunNode::make(s->outer->name + "_luf",
                                Range::make_by_min_extent(parent_fun->range->min / s->factor,
                                                          parent_fun->range->extent / s->factor),
                                parent_fun->dimensions, parent_fun->parameters,
                                parent_fun->body / s->factor, UninterpFunNode::kLFun);
      l_funs[s->inner.operator->()] = inner_fun;
      l_funs[s->outer.operator->()] = outer_fun;
    } else if (const RaggedDimensionFuseNode* s = rel.as<RaggedDimensionFuseNode>()) {
      CHECK(l_funs.count(s->outer.operator->()) || l_funs.count(s->inner.operator->()));
      UninterpFun outer_fun = l_funs.at(s->outer.operator->());
      UninterpFun inner_fun = l_funs.at(s->inner.operator->());

      CHECK_EQ(outer_fun->arity(), 0)
          << "Currently, we only allow dimension fusion where the outer dimension in dense.";

      PrimExpr outer_max_inclusive = outer_fun->body - 1;
      PrimExpr inner_max_inclusive = inner_fun.MakeCallTo({outer_max_inclusive}, {s->outer});
      PrimExpr max_inclusive = s->outer_inner_to_fused_uf.MakeCallTo(
          Array<PrimExpr>({outer_max_inclusive, inner_max_inclusive}), {s->outer, s->inner});
      l_funs[s->fused.operator->()] =
          UninterpFunNode::from_constant(s->fused->name + "_luf", max_inclusive + 1);

    } else if (const DimensionFuseNode* s = rel.as<DimensionFuseNode>()) {
      CHECK(l_funs.count(s->outer.operator->()) || l_funs.count(s->inner.operator->()));
      UninterpFun outer_fun = l_funs.at(s->outer.operator->());
      UninterpFun inner_fun = l_funs.at(s->inner.operator->());

      PrimExpr range_min = outer_fun->range->min * inner_fun->range->min;
      PrimExpr range_extent = outer_fun->range->extent * inner_fun->range->min +
                              outer_fun->range->min * inner_fun->range->extent +
                              outer_fun->range->extent * inner_fun->range->extent;

      Array<Dimension> dimensions;
      Array<Var> parameters;
      std::unordered_map<const VarNode*, PrimExpr> outer_vsub;
      std::unordered_map<const VarNode*, PrimExpr> inner_vsub;

      auto handle_fun = [&dimensions, &parameters](
                            const UninterpFun& fun,
                            std::unordered_map<const VarNode*, PrimExpr>& vsub) {
        for (size_t i = 0; i < fun->arity(); ++i) {
          if (!dimensions.Contains(fun->dimensions[i])) {
            dimensions.push_back(fun->dimensions[i]);
            auto param = Var(fun->dimensions[i]->name + "_fp", DataType::Int(32));
            parameters.push_back(param);
            vsub[fun->parameters[i].operator->()] = param;
          }
        }
      };

      handle_fun(outer_fun, outer_vsub);
      handle_fun(inner_fun, inner_vsub);

      PrimExpr body =
          VarReplacer(outer_vsub)(outer_fun->body) * VarReplacer(inner_vsub)(inner_fun->body);
      UninterpFun fused_fun = UninterpFunNode::make(
          s->fused->name + "_luf", Range::make_by_min_extent(range_min, range_extent), dimensions,
          parameters, body, UninterpFunNode::kLFun);
      l_funs[s->fused.operator->()] = fused_fun;
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }

  Array<PrimExpr> leaf_l_fun_maxs;
  Array<UninterpFun> leaf_l_funs;
  Map<Dimension, UninterpFun> leaf_a_funs;
  for (auto dim : stage->dim_relation_graph->leaf_dimensions) {
    CHECK(l_funs.count(dim.operator->()));
    auto l_fun = l_funs.at(dim.operator->());
    leaf_l_funs.push_back(l_fun);
    leaf_l_fun_maxs.push_back(l_fun->range->max_inclusive());
    leaf_a_funs.Set(dim, UninterpFunNode::make("a_fi", NullValue<Range>(), {}, {},
                                               NullValue<PrimExpr>(), UninterpFunNode::kAFun));
  }

  if (root_layout->loop_layout) {
    CHECK(false);
    return {};
  } else {
    return ModesNode::make_storage_layout(stage->dim_relation_graph->leaf_dimensions,
                                          leaf_l_fun_maxs, leaf_l_funs, leaf_a_funs);
  }
}

void DimensionPassUpDomain(Stage s, std::unordered_map<const DimensionNode*, Range>* p_state,
                           bool allow_missing) {
  const DimensionRelationGraph& graph = s->dim_relation_graph;
  arith::Analyzer analyzer;
  auto ceil_div = [&analyzer](PrimExpr a, PrimExpr b) {
    if (analyzer.CanProve(indexmod(a, b) == 0)) {
      return analyzer.Simplify(indexdiv(a, b));
    }
    return analyzer.Simplify(indexdiv(a + (b - 1), b));
  };

  auto& state = *p_state;
  // forwar iteration on relations
  for (size_t i = s->dim_relation_graph->relations.size(); i != 0; --i) {
    DimensionRelation rel = s->dim_relation_graph->relations[i - 1];
    if (const DimensionSplitNode* r = rel.as<DimensionSplitNode>()) {
      if (!state.count(r->inner.operator->()) || !state.count(r->outer.operator->())) {
        CHECK(allow_missing);
        continue;
      }

      const Range& range_inner = state.at(r->inner.operator->());
      const Range& range_outer = state.at(r->outer.operator->());
      CHECK(r->factor.defined());
      PrimExpr parent_min = range_outer->min * r->factor + range_inner->min;
      PrimExpr parent_max_inclusive =
          range_outer->max_inclusive() * r->factor + range_inner->max_inclusive();
      state[r->parent.operator->()] =
          Range::make_by_min_max_inclusive(parent_min, parent_max_inclusive);
    } else if (const RaggedDimensionFuseNode* r = rel.as<RaggedDimensionFuseNode>()) {
      if (!state.count(r->fused.operator->())) {
        CHECK(allow_missing);
        continue;
      }

      Range fused = state.at(r->fused.operator->());
      auto int_type = DataType::Int(32);
      Range outer;
      Range inner;
      if (is_one(fused->extent)) {
        PrimExpr fused_val = fused->min;
        PrimExpr outer_val = zero_if_args_zero_ufun_call(
            int_type, {fused_val}, r->fused_to_outer_uf->dimensions, r->fused_to_outer_uf);
        outer = Range::make_by_min_extent(outer_val, 1);
        PrimExpr inner_val = zero_if_args_zero_ufun_call(
            int_type, {fused_val}, r->fused_to_inner_uf->dimensions, r->fused_to_inner_uf);
        inner = Range::make_by_min_extent(inner_val, 1);
      } else {
        CHECK(false);
        PrimExpr fused_min = fused->min;
        PrimExpr fused_max_inclusive = fused->max_inclusive();

        PrimExpr outer_min = zero_if_args_zero_ufun_call(
            int_type, {fused_min}, r->fused_to_outer_uf->dimensions, r->fused_to_outer_uf);
        PrimExpr outer_max_inclusive =
            zero_if_args_zero_ufun_call(int_type, {fused_max_inclusive},
                                        r->fused_to_outer_uf->dimensions, r->fused_to_outer_uf);
        outer = Range::make_by_min_max_inclusive(outer_min, outer_max_inclusive);

        PrimExpr inner_min_boundary = zero_if_args_zero_ufun_call(
            int_type, {fused_min}, r->fused_to_inner_uf->dimensions, r->fused_to_inner_uf);

        PrimExpr inner_max_inclusive_boundary =
            zero_if_args_zero_ufun_call(int_type, {fused_max_inclusive},
                                        r->fused_to_inner_uf->dimensions, r->fused_to_inner_uf);

        // Range inner_range = dom_map.at(s->inner);

        // inner = Range::make_by_min_max_inclusive(
        //     SelectNode::make(EQNode::make(s->outer, outer_min), inner_min_boundary,
        //                      inner_range->min),
        //     SelectNode::make(EQNode::make(s->outer, outer_max_inclusive),
        //                      inner_max_inclusive_boundary, inner_range->max_inclusive()));
      }
      state[r->outer.operator->()] = outer;
      state[r->inner.operator->()] = inner;
    } else if (const DimensionFuseNode* r = rel.as<DimensionFuseNode>()) {
      if (!state.count(r->fused.operator->())) {
        CHECK(allow_missing);
        continue;
      }
      const Range& range_fused = state.at(r->fused.operator->());
      if (is_one(Simplify(range_fused->extent))) {
        state[r->outer.operator->()] =
            Range::make_by_min_extent(indexdiv(range_fused->min, r->factor), 1);
        state[r->inner.operator->()] =
            Range::make_by_min_extent(indexmod(range_fused->min, r->factor), 1);
      } else {
        auto outer_range =
            Range::make_by_min_max_inclusive(indexdiv(range_fused->min, r->factor),
                                             indexdiv(range_fused->max_inclusive(), r->factor));
        state[r->outer.operator->()] = outer_range;
        state[r->inner.operator->()] = Range::make_by_min_max_inclusive(
            if_then_else(outer_range->extent > 1, 0, indexmod(range_fused->min, r->factor)),
            if_then_else(outer_range->extent > 1, r->factor - 1,
                         indexmod(range_fused->max_inclusive(), r->factor)));
      }
    } else {
      LOG(FATAL) << "Unsupported relation type";
    }
  }

  for (auto it : state) {
    state[it.first] =
        Range::make_by_min_extent(Simplify(it.second->min), Simplify(it.second->extent));
  }
}

void LeafDimensionsDependenceInformation(Stage& stage, const Modes& root_layout,
                                         DimDepMap* p_outer_to_inner_deps,
                                         DimDepMap* p_inner_to_outer_deps) {
  bool print = (stage->op->name == "B");

  DimDepMap& outer_to_inner_deps = *p_outer_to_inner_deps;
  DimDepMap& inner_to_outer_deps = *p_inner_to_outer_deps;

  auto insert_map = [](DimDepMap& map, Dimension key, Dimension value) {
    auto it = map.find(key.operator->());
    it->second.insert(value.operator->());
  };

  for (size_t i = 0; i < root_layout->dimensions.size(); ++i) {
    inner_to_outer_deps[root_layout->dimensions[i].operator->()] = {};
    outer_to_inner_deps[root_layout->dimensions[i].operator->()] = {};
  }
  for (size_t i = 0; i < root_layout->dimensions.size(); ++i) {
    Dimension inner_dim = root_layout->dimensions[i];
    for (auto outer_dim : root_layout->l_funs[i]->dimensions) {
      insert_map(outer_to_inner_deps, outer_dim, inner_dim);
      insert_map(inner_to_outer_deps, inner_dim, outer_dim);
    }
  }

  for (size_t i = stage->dim_relation_graph->relations.size(); i != 0; --i) {
    DimensionRelation rel = stage->dim_relation_graph->relations[i - 1];
    if (const DimensionSplitNode* s = rel.as<DimensionSplitNode>()) {
      CHECK(s->factor.defined());
      auto outer = s->outer.operator->();
      auto inner = s->inner.operator->();
      auto parent = s->parent.operator->();

      outer_to_inner_deps[outer] = outer_to_inner_deps[parent];
      outer_to_inner_deps[inner] = outer_to_inner_deps[parent];

      inner_to_outer_deps[outer] = outer_to_inner_deps[parent];
      // inner_to_outer_deps[inner] = {};
      inner_to_outer_deps[inner] = outer_to_inner_deps[parent];
    } else if (const DimensionFuseNode* s = rel.as<DimensionFuseNode>()) {
      auto outer = s->outer.operator->();
      auto inner = s->inner.operator->();
      auto fused = s->fused.operator->();

      auto set_fused_set = [&](DimDepMap& parent_map) {
        DimNodeSet fused_deps;
        auto outer_deps = parent_map[outer];
        auto inner_deps = parent_map[inner];

        fused_deps.insert(outer_deps.begin(), outer_deps.end());
        fused_deps.insert(inner_deps.begin(), inner_deps.end());
        parent_map[fused] = fused_deps;
      };
      set_fused_set(outer_to_inner_deps);
      set_fused_set(inner_to_outer_deps);
    } else {
      LOG(FATAL) << "Unsupported relation type";
    }
  }
}

}  // namespace te
}  // namespace tvm
