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
    // } else if (isCudaThread(iv) || isCPUEnvThread(iv)) {
    //   // Range range = it->second;
    //   // PrimExpr to_prove =
    //   //     UninterpFun::InlineUninterpFunCalls(range->extent + range->min >= r->extent +
    //   r->min);
    //   // CHECK(is_zero(r->min) && analyzer->CanProve(to_prove))
    //   //     << iv->var << " " << r << " " << range << " " << to_prove;

    //   Range range = iv->dom;
    //   std::cout << iv->var << " " << r << " " << range << std::endl;
    //   PrimExpr to_prove =
    //       UninterpFun::InlineUninterpFunCalls(range->extent + range->min >= r->extent + r->min);
    //   CHECK(is_zero(r->min) && analyzer->CanProve(to_prove))
    //       << iv->var << " " << r << " " << range << " " << to_prove;
  } else {
    // TODO (ppf): HACK HACK HACK. We're commenting out an error condition that should ideally be
    // checked reported
    bool match = is_zero(it->second->min) &&
                 analyzer->CanProve(
                     UninterpFun::InlineUninterpFunCalls(r->extent - it->second->extent) == 0);
    // bool match = analyzer->CanProve(
    // UninterpFun::InlineUninterpFunCalls(r->extent - it->second->extent) == 0);
    CHECK(match) << iv << " domain already inferred,"
                 << " cannot prove their extents are the same " << it->second->extent << " vs "
                 << r->extent << " " << it->second;
  }
}

void UpdateShim(const Stage& stage, std::unordered_map<IterVar, Range>* p_state, const IterVar& iv,
                Range r, arith::Analyzer* analyzer) {
  Update(p_state, iv, r, analyzer);
}

void PassDownDomain(const Stage& stage, std::unordered_map<IterVar, Range>* p_state,
                    arith::Analyzer* actx, bool allow_missing) {
  bool print = false;  // stage->op->name == "O";
  auto ceil_div = [actx, print](PrimExpr a, PrimExpr b) {
    if (actx->CanProve(indexmod(a, b) == 0)) {
      if (print) std::cout << "[SPL]    Simpl Bwgin 1" << std::endl;
      auto ret = actx->Simplify(indexdiv(a, b));
      if (print) std::cout << "[SPL]    Simpl End 1" << std::endl;
      return ret;
    }
    if (print) std::cout << "[SPL] When**********************************************" << std::endl;
    auto ret = actx->Simplify(indexdiv(a + (b - 1), b));
    if (print) std::cout << "[SPL] What**********************************************" << std::endl;
    return ret;
  };

  auto& state = *p_state;
  // forward iteration on relations
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
          // std::cout << "[SPL]    When" << std::endl;
          auto outer_extent = ceil_div(range_parent->extent, r->factor);
          // std::cout << "[SPL]    What" << std::endl;
          std::cout << "[SPL]    FAC " << r->outer->var << " " << outer_extent << std::endl;
          std::cout << "[SPL]    FAC " << r->inner->var << " "
                    << Range::make_by_min_extent(0, r->factor) << std::endl;
        }
        UpdateShim(stage, p_state, r->inner, Range::make_by_min_extent(0, r->factor), actx);
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
      const Range& range_outer = state.at(r->outer);
      const Range& range_inner_unreplaced = state.at(r->inner);
      ;
      Range range_inner = Range(
          VarReplacer({{r->outer->var.as<VarNode>(), range_outer->min}})(
              range_inner_unreplaced->min),
          VarReplacer({{r->outer->var.as<VarNode>(), range_outer->extent + range_outer->min - 1}})(
              range_inner_unreplaced->min + range_inner_unreplaced->extent - 1));
      if (print)
        std::cout << "[FPL]   Outer/Inner " << range_outer << " " << range_inner << std::endl;
      state[r->fused] = Range(
          CallNode::make(r->fused->var.dtype(), r->outer_inner_to_fused_uf->fname,
                         {range_outer->min, range_inner->min}, CallNode::CallType::UninterpFunCall,
                         r->outer_inner_to_fused_uf->dimensions, r->outer_inner_to_fused_uf, 0),
          CallNode::make(r->fused->var.dtype(), r->outer_inner_to_fused_uf->fname,
                         {range_outer->min + range_outer->extent - 1,
                          range_inner->min + range_inner->extent - 1},
                         CallNode::CallType::UninterpFunCall,
                         r->outer_inner_to_fused_uf->dimensions, r->outer_inner_to_fused_uf, 0));
      if (print) std::cout << "[FPL]    Fused " << state[r->fused] << std::endl;
    } else if (const FuseNode* r = rel.as<FuseNode>()) {
      if (!state.count(r->outer) || !state.count(r->inner)) {
        CHECK(allow_missing);
        continue;
      }
      const Range& range_outer = state.at(r->outer);
      const Range& range_inner = state.at(r->inner);
      state[r->fused] = Range::make_by_min_extent(0, range_outer->extent * range_inner->extent);
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
    } else if (const RaggedFuseNode* s = rel.as<RaggedFuseNode>()) {
      if (!state.count(s->fused)) {
        CHECK(allow_missing);
        continue;
      }
      PrimExpr fused_value = state.at(s->fused);
      state[s->outer] = CallNode::make(s->outer->var.dtype(), s->fused_to_outer_uf->fname,
                                       {fused_value}, CallNode::CallType::UninterpFunCall,
                                       s->fused_to_outer_uf->dimensions, s->fused_to_outer_uf, 0);
      state[s->inner] = CallNode::make(s->inner->var.dtype(), s->fused_to_inner_uf->fname,
                                       {fused_value}, CallNode::CallType::UninterpFunCall,
                                       s->fused_to_inner_uf->dimensions, s->fused_to_inner_uf, 0);
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
    } else if (rel.as<SingletonNode>()) {
    } else {
      LOG(FATAL) << "unknown relation type";
    }
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
      state[s->fused] =
          CallNode::make(s->fused->var.dtype(), s->outer_inner_to_fused_uf->fname,
                         {outer_value, inner_value}, CallNode::CallType::UninterpFunCall,
                         s->outer_inner_to_fused_uf->dimensions, s->outer_inner_to_fused_uf, 0);
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
      CHECK(is_zero(outer_min));
      CHECK(is_zero(inner_min));
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
    *parent = IntSet::range(dom_map.at(s->parent));
    return;
  }
  PrimExpr factor = dom_map.at(s->inner)->extent;
  PrimExpr parent_min = dom_map.at(s->parent)->min;
  CHECK(outer.defined());
  CHECK(inner.defined());
  CHECK(factor.defined());
  *parent = arith::EvalSet(s->outer->var * factor + s->inner->var + parent_min,
                           {{s->outer, outer}, {s->inner, inner}});
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

  PrimExpr fused_min = fused.min();
  PrimExpr fused_max = fused.max();

  PrimExpr outer_min = CallNode::make(s->outer->var.dtype(), s->fused_to_outer_uf->fname,
                                      {fused_min}, CallNode::CallType::UninterpFunCall,
                                      s->fused_to_outer_uf->dimensions, s->fused_to_outer_uf, 0);
  PrimExpr outer_max = CallNode::make(s->outer->var.dtype(), s->fused_to_outer_uf->fname,
                                      {fused_max}, CallNode::CallType::UninterpFunCall,
                                      s->fused_to_outer_uf->dimensions, s->fused_to_outer_uf, 0);
  *outer = IntSet::range(Range(outer_min, outer_max));

  PrimExpr inner_min_boundary =
      CallNode::make(s->inner->var.dtype(), s->fused_to_inner_uf->fname, {fused_min},
                     CallNode::CallType::UninterpFunCall, s->fused_to_inner_uf->dimensions,
                     s->fused_to_inner_uf, 0);

  PrimExpr inner_max_boundary =
      CallNode::make(s->inner->var.dtype(), s->fused_to_inner_uf->fname, {fused_max},
                     CallNode::CallType::UninterpFunCall, s->fused_to_inner_uf->dimensions,
                     s->fused_to_inner_uf, 0);

  Range inner_range = dom_map.at(s->inner);

  *inner = IntSet::range(
      Range(if_then_else(EQNode::make(s->outer, outer_min), inner_min_boundary, inner_range->min),
            if_then_else(EQNode::make(s->outer, outer_max), inner_max_boundary,
                         inner_range->min + inner_range->extent - 1)));
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
  bool print = false;  //(stage->op->name == "imv.ila.repl");
  auto& state = *p_state;
  for (size_t i = stage->relations.size(); i != 0; --i) {
    IterVarRelation rel = stage->relations[i - 1];
    if (const SplitNode* r = rel.as<SplitNode>()) {
      IntSet parent;
      PassUpDomain(r, dom_map, state.at(r->outer), state.at(r->inner), &parent);
      state[r->parent] = parent;
      if (print)
        std::cout << "[PUD] Split " << state.at(r->outer) << " " << state.at(r->inner) << " "
                  << parent << std::endl;
    } else if (const RaggedFuseNode* r = rel.as<RaggedFuseNode>()) {
      IntSet outer, inner;
      PassUpDomain(r, dom_map, state.at(r->fused), &outer, &inner);
      state[r->outer] = outer;
      state[r->inner] = inner;
      if (print)
        std::cout << "[PUD] Fuse "
                  << " " << state.at(r->fused) << inner << " " << outer << std::endl;
    } else if (const FuseNode* r = rel.as<FuseNode>()) {
      IntSet outer, inner;
      PassUpDomain(r, dom_map, state.at(r->fused), &outer, &inner);
      state[r->outer] = outer;
      state[r->inner] = inner;
      if (print)
        std::cout << "[PUD] Fuse "
                  << " " << state.at(r->fused) << inner << " " << outer << std::endl;
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
          if (analyzer->CanProve(dom_map.at(s->parent)->extent == factor * step)) {
            state[s->parent] = false;
          } else {
            state[s->parent] = true;
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
    // if (stage->op->name == "next_v") {
    // std::cout << "[CUDA_BOUND] " << var->name_hint << " "
    // << (bind_map.count(var) ? bind_map.at(var) : "") << std::endl;
    // }
    if (bind_map.count(var)) ret.insert(bind_map.at(var));
  }
  return ret;
}

std::vector<PrimExpr> MakeBoundCheck(
    const Stage& stage, const Map<IterVar, Range>& dom_map,
    const std::unordered_map<std::string, Range>& env_dom_map,
    const std::unordered_map<std::string, IterVar>& env_var_map,
    const std::unordered_map<const VarNode*, std::string>& bind_map,
    const std::unordered_map<IterVar, PrimExpr>& value_map, bool skip_ivar_domain,
    const std::unordered_set<IterVar>& skip_iter) {
  arith::Analyzer analyzer;

  bool print = false;  //(stage->op->name == "O");
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
    // std::cout << "PROCESSING PRED " << pred << std::endl;
    PrimExpr inlined = UninterpFun::InlineUninterpFunCalls(pred);
    PrimExpr replaced = replacer(inlined);
    PrimExpr ret = tir::Simplify(replaced);
    if (analyzer.CanProve(!ret)) {
      std::cout << "FALSE PRED " << pred << " " << inlined << " " << replaced << " " << ret
                << std::endl;
      std::cout << " " << std::endl;
    }
    return ret;
  };

  std::unordered_map<IterVar, bool> bound_state;
  for (IterVar iv : stage->leaf_iter_vars) {
    bound_state[iv] = false;
  }
  PassUpBoundCheck(stage, dom_map, &bound_state, &analyzer);

  std::vector<PrimExpr> preds;
  std::unordered_map<const VarNode*, IntSet> iset_dmap;

  std::unordered_map<const VarNode*, PrimExpr> value_vsub_map;
  // setup domain map for set analysis
  for (const auto& kv : dom_map) {
    if (isCudaThread(kv.first) && env_dom_map.count(kv.first->var->name_hint)) {
      CHECK(env_var_map.count(kv.first->var->name_hint)) << kv.first->var->name_hint;
      iset_dmap[env_var_map.at(kv.first->var->name_hint)->var.get()] =
          IntSet::range(env_dom_map.at(kv.first->var->name_hint));
      value_vsub_map[kv.first->var.get()] = env_var_map.at(kv.first->var->name_hint);
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
      if (env_dom_map.count(bound_thread_var->var->name_hint)) {
        bound_thread_range =
            env_dom_map.at(bound_thread_var->var->name_hint);  // dom_map[bound_thread_var];
      } else {
        bound_thread_range = dom_map[bound_thread_var];
      }
      generated_env_checks.insert(bound_thread_var->var->name_hint);
      if (!analyzer.CanProve(bound_thread_range->extent == original_range->extent)) {
        preds.emplace_back(process_pred(bound_thread_var->var < original_range->extent));
      }
    }
  }

  if (stage->op.as<ComputeOpNode>()) {
    auto cudaVars = CollectDependentCudaVars(stage, dom_map, bind_map, value_map);
    for (auto it : env_var_map) {
      if (!generated_env_checks.count(it.first) && !cudaVars.count(it.first)) {
        tvm::runtime::ThreadScope ts = tvm::runtime::ThreadScope::make(it.first);
        if (stage->storage_scope_rank <= ts.rank) {
          // if (print) std::cout << "[CHECK2] " << stage << " " << (it.second->var < 1) <<
          // std::endl;
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
      PrimExpr value = value_map.at(iv) - dom->min;
      PrimExpr vmax = EvalSet(value, iset_dmap).max();
      if (vmax.dtype() != value.dtype() || !analyzer.CanProve(vmax < dom->extent)) {
        // if (print) {
        //   std::cout << "[CHECK3]   " << iv << " " << process_pred(value < dom->extent) <<
        //   std::endl;
        // }
        preds.emplace_back(process_pred(value < dom->extent));
      }
    }
  }

  VarReplacer value_replacer(value_vsub_map);
  for (const IterVar& iv : stage->op->root_iter_vars()) {
    if (skip_iter.count(iv) || iv->iter_type == kOpaque || iv->iter_type == kLoopNestOpaque)
      continue;
    Range dom = dom_map.at(iv);
    CHECK(iv->dom.defined());
    if (!skip_ivar_domain && !iv->dom.same_as(dom)) {
      if (print) {
        std::cout << "[CHECK]   " << iv << " " << iv->dom << " " << value_map.at(iv) << std::endl;
      }

      PrimExpr value = replacer(value_map.at(iv) - iv->dom->min);
      IntSet s = EvalSet(value, iset_dmap);
      PrimExpr vmin = s.min();
      PrimExpr vmax = s.max();
      // if (print) {
      //   std::cout << "[CHECK0]   " << value_map.at(iv) << " " << s << " "
      //             << analyzer.CanProve(process_pred1(value >= 0)) << std::endl;
      // }
      // The range of `value` resides in [vmin, vmax]
      // std::cout << "[TPT] " << (vmin >= 0) << std::endl;
      if (vmin.dtype() != value.dtype() || !analyzer.CanProve(vmin >= 0)) {
        // if (print) {
        //   std::cout << "[CHECK5]   " << process_pred(value >= 0) << std::endl;
        // }
        preds.emplace_back(process_pred(value >= 0));
      }
      // if (print) {
      //   std::cout << "[CHECK_MAX]   " << value << " " << vmax << " " << iv->dom->extent
      //             << std::endl;
      // }
      if (vmax.dtype() != value.dtype() || !analyzer.CanProve(vmax < iv->dom->extent)) {
        if (print) {
          std::cout << "[CHECK6]   " << iv << " " << process_pred(value < iv->dom->extent)
                    << std::endl;
        }
        preds.emplace_back(process_pred(value < iv->dom->extent));
      }
    }
  }

  // Dedup for readibility
  std::vector<PrimExpr> ret;
  for (const auto& pred : preds) {
    bool repeated = false;
    for (const auto& fin : ret) {
      if (analyzer.CanProve(UninterpFun::InlineUninterpFunCalls(pred == fin))) {
        repeated = true;
      }
    }
    if (!repeated) {
      if (print) std::cout << "[PUSHING] " << pred << std::endl;
      ret.push_back(pred);
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
        // Update(p_state, r->inner, Range::make_by_min_extent(0, r->factor), analyzer);

        // Update(p_state, r->inner, Range::make_by_min_extent(range_parent->min, r->factor),
        //        analyzer);
        // Update(p_state, r->outer,
        //        Range::make_by_min_extent(0, ceil_div(range_parent->extent, r->factor)),
        //        analyzer);

        Update(p_state, r->inner, Range::make_by_min_extent(0, r->factor), analyzer);
        Update(p_state, r->outer,
               Range::make_by_min_extent(ceil_div(range_parent->min, r->factor),
                                         ceil_div(range_parent->extent, r->factor)),
               analyzer);
      } else {
        CHECK(false);
        // Update(p_state, r->outer, Range::make_by_min_extent(0, r->nparts), analyzer);
        // // Update(p_state, r->inner,
        // // Range::make_by_min_extent(0, ceil_div(range_parent->extent, r->nparts)), analyzer);

        // Update(
        //     p_state, r->inner,
        //     Range::make_by_min_extent(range_parent->min, ceil_div(range_parent->extent,
        //     r->nparts)), analyzer);

        Update(p_state, r->outer, Range::make_by_min_extent(0, r->nparts), analyzer);

        Update(p_state, r->inner,
               Range::make_by_min_extent(ceil_div(range_parent->min, r->nparts),
                                         ceil_div(range_parent->extent, r->nparts)),
               analyzer);
      }
    } else if (const DimensionFuseNode* r = rel.as<DimensionFuseNode>()) {
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
      UninterpFun outer_fun = UninterpFunNode::make(
          s->outer->name + "_luf",
          Range::make_by_min_extent(parent_fun->range->min / s->factor,
                                    parent_fun->range->extent / s->factor),
          parent_fun->dimensions, parent_fun->parameters, parent_fun->body / s->factor);
      l_funs[s->inner.operator->()] = inner_fun;
      l_funs[s->outer.operator->()] = outer_fun;
    } else if (const DimensionFuseNode* s = rel.as<DimensionFuseNode>()) {
      CHECK(l_funs.count(s->outer.operator->()) || l_funs.count(s->inner.operator->()));
      UninterpFun outer_fun = l_funs.at(s->outer.operator->());
      UninterpFun inner_fun = l_funs.at(s->inner.operator->());

      PrimExpr range_min = outer_fun->range->min * inner_fun->range->min;
      PrimExpr range_extent = outer_fun->range->extent * inner_fun->range->min +
                              outer_fun->range->min * inner_fun->range->extent +
                              outer_fun->range->extent * inner_fun->range->extent;

      Map<Dimension, Var> outer_param_mapping;
      Map<Dimension, Var> inner_param_mapping;
      Array<Dimension> dimensions;
      for (size_t i = 0; i < outer_fun->arity(); ++i) {
        outer_param_mapping.Set(outer_fun->dimensions[i], outer_fun->parameters[i]);
        if (!dimensions.Contains(outer_fun->dimensions[i])) {
          dimensions.push_back(outer_fun->dimensions[i]);
        }
      }

      for (size_t i = 0; i < inner_fun->arity(); ++i) {
        inner_param_mapping.Set(inner_fun->dimensions[i], inner_fun->parameters[i]);
        if (!dimensions.Contains(inner_fun->dimensions[i])) {
          dimensions.push_back(inner_fun->dimensions[i]);
        }
      }

      std::unordered_map<const VarNode*, PrimExpr> outer_vsub;
      std::unordered_map<const VarNode*, PrimExpr> inner_vsub;

      Array<Var> parameters;
      for (auto dim : dimensions) {
        Var param = Var(dim->name + "_fp", DataType::Int(32));
        parameters.push_back(param);
        if (s->dependent_ragged_dims) {
          if (outer_param_mapping.count(dim)) {
            outer_vsub[outer_param_mapping.at(dim).as<VarNode>()] = param;
          }
          if (inner_param_mapping.count(dim)) {
            inner_vsub[inner_param_mapping.at(dim).as<VarNode>()] = param;
          }
        }
      }

      if (s->dependent_ragged_dims) {
        PrimExpr body =
            VarReplacer(outer_vsub)(outer_fun->body) * VarReplacer(inner_vsub)(inner_fun->body);

        UninterpFun fused_fun = UninterpFunNode::make(
            s->fused->name + "_luf", Range::make_by_min_extent(range_min, range_extent), dimensions,
            parameters, body);
        l_funs[s->fused.operator->()] = fused_fun;
      } else {
        UninterpFun fused_fun = UninterpFunNode::make(
            s->fused->name + "_oif", Range::make_by_min_extent(range_min, range_extent), dimensions,
            parameters, NullValue<PrimExpr>());
        l_funs[s->fused.operator->()] = fused_fun;
      }
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
    leaf_l_fun_maxs.push_back(l_fun->range->min + l_fun->range->extent - 1);
    leaf_a_funs.Set(
        dim, UninterpFunNode::make("a_fi", NullValue<Range>(), {}, {}, NullValue<PrimExpr>()));
  }

  return ModesNode::make(stage->dim_relation_graph->leaf_dimensions, leaf_l_fun_maxs, leaf_l_funs,
                         leaf_a_funs, root_layout->loop_layout);
}

}  // namespace te

}  // namespace tvm
