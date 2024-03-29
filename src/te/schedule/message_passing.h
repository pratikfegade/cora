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
 * \file message_passing.h
 * \brief Common utilities to do message passing
 *  on the schedule hyper graph.
 */
#ifndef TVM_TE_SCHEDULE_MESSAGE_PASSING_H_
#define TVM_TE_SCHEDULE_MESSAGE_PASSING_H_

#include <tvm/arith/analyzer.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/expr.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace te {
/*!
 * \brief Downward inference of domain of each IterVar.
 *  Caller set the range of the root, then the function
 *  propagates it towards the leaves.
 *
 * \param stage The stage to operate on.
 * \param p_state The state of the message passing.
 * \param analyzer Analyzer context, storing information about bounds in p_state.
 * \param allow_missing Whether allow missing value.
 */
void PassDownDomain(const Stage& stage, std::unordered_map<IterVar, Range>* p_state,
                    arith::Analyzer* analyzer, bool allow_missing = false);

/*!
 * \param Upward inference of index of each IterVar.
 *  given index assignement of the leaves,
 *
 * \param stage The stage to operate on.
 * \param dom_map The domain map of each iteration variable's domain.
 * \param p_state The index state of each IterVar.
 * \param allow_missing Whether allow missing value.
 */
void PassUpIndex(const Stage& stage, const Map<IterVar, Range>& dom_map,
                 std::unordered_map<IterVar, PrimExpr>* p_state, bool allow_missing = false);

/*!
 * \param Downward inference of index of each IterVar.
 *  given index assignement of roots.
 *
 * \param stage The stage to operate on.
 * \param dom_map The domain map of each iteration variable's domain.
 * \param p_state The index state of each IterVar.
 * \param allow_missing Whether allow missing value.
 */
void PassDownIndex(const Stage& stage, const Map<IterVar, Range>& dom_map,
                   std::unordered_map<IterVar, PrimExpr>* p_state, bool allow_missing = false);

/*!
 * \param Upward inference of domain set of each IterVar.
 *  given domain assignment of the leaves,
 *
 * \param stage The stage to operate on.
 * \param dom_map The domain map of each iteration variable's maximum domain.
 * \param p_state The index state of each IterVar.
 */
void PassUpDomain(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                  std::unordered_map<IterVar, IntSet>* p_state);

/*!
 * \brief Upward message passing of bitmask with or relation.
 * \param stage The stage to operate on.
 * \param p_state The index state of each IterVar.
 * \param allow_missing Whether allow missing value.
 */
void PassUpBitMaskOr(const Stage& stage, std::unordered_map<IterVar, int>* p_state,
                     bool allow_missing = false);

/*!
 * \brief Downward message passing of bitmask with or relation.
 * \param stage The stage to operate on.
 * \param p_state The index state of each IterVar.
 * \param allow_missing Whether allow missing value.
 */
void PassDownBitMaskOr(const Stage& stage, std::unordered_map<IterVar, int>* p_state,
                       bool allow_missing = false);

/*!
 * \brief Create boundary check predicates given remapped value of root
 * \param stage The stage we operate on
 * \param dom_map The domain map of each value.
 * \param value_map The value map of the root iter var.
 * \param skip_ivar_domain Whether we skip check for IterVar's original domain.
 * \param skip_iter The set of variables to skip bound condition.
 * \return List of predicates that we need to check.
 */
std::vector<PrimExpr> MakeBoundCheck(
    const Stage& stage, const Map<IterVar, Range>& dom_map,
    const std::unordered_map<std::string, Range>& env_dom_map,
    const std::unordered_map<std::string, IterVar>& env_var_map,
    const std::unordered_map<const VarNode*, std::string>& bind_map,
    const std::unordered_map<IterVar, PrimExpr>& value_map, bool skip_ivar_domain,
    const std::unordered_set<IterVar>& skip_iter, const Map<Stage, Array<Stage>>& attach_stages,
    const Map<Stage, Array<IterVar>>& attach_vars);

/* Pass values down the dimension relations */
void DimensionPassDownValues(Stage s, const BaseVarDimOpNode* compute_op,
                             const std::unordered_map<const DimensionNode*, Range>& dom_map,
                             std::unordered_map<const DimensionNode*, PrimExpr>* p_state,
                             bool allow_missing);

Modes DimensionPassDownModes(Stage& s, const BaseVarDimOpNode* compute_op,
                             // const std::unordered_map<const DimensionNode*, Range>& dom_map,
                             const Modes& root_layout);

void DimensionPassDownDomain(Stage s, const BaseVarDimOpNode* op,
                             std::unordered_map<const DimensionNode*, Range>* p_state,
                             bool allow_missing);

void DimensionPassUpBitMaskOr(const Stage& stage,
                              std::unordered_map<const DimensionNode*, int>* p_state,
                              bool allow_missing = false);

void DimensionPassUpBitMaskExact(const Stage& stage,
                                 std::unordered_set<const DimensionNode*>* p_state,
                                 bool* p_exact_possible);

using DimNodeSet = std::unordered_set<const DimensionNode*>;
using DimDepMap = std::unordered_map<const DimensionNode*, DimNodeSet>;

void LeafDimensionsDependenceInformation(Stage& stage, const Modes& root_layout,
                                         DimDepMap* p_outer_to_inner_deps,
                                         DimDepMap* p_inner_to_outer_deps);

void DimensionPassUpDomain(Stage s, std::unordered_map<const DimensionNode*, Range>* p_state,
                           bool allow_missing);

}  // namespace te
}  // namespace tvm
#endif  // TVM_TE_SCHEDULE_MESSAGE_PASSING_H_
