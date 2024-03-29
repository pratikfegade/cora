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
 * \file op_util.h
 * \brief Common utility used in operator construction.
 */
#ifndef TVM_TE_OPERATION_OP_UTIL_H_
#define TVM_TE_OPERATION_OP_UTIL_H_

#include <tvm/te/schedule.h>
#include <tvm/tir/expr.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../tir/pass/arg_binder.h"
#include "../../tir/pass/ir_util.h"
#include "../schedule/message_passing.h"

namespace tvm {
namespace te {

using tir::MergeNest;

const BaseVarDimOpNode* GetBaseVarDimOp(Operation op);

std::vector<std::vector<Stmt>> MergeWhileHoisting(const Stage& s,
                                                  const std::vector<std::vector<Stmt>>& defs,
                                                  const std::vector<Stmt>& preds);

/*!
 * \brief During PropBoundsToInputs, if any of the intsets refer to
 * any iter vars of the consumer, we must translate them accordingly
 * to the itervars of the corresponding producer. This is needed to be
 * done as now allow loop extents to be a function of other loop iter
 * vars.
 *
 * \param set The set to be translated.
 * \param producer The producer.
 * \param tensor The consumed tensor.
 */
IntSet TranslateIterVarsFromConsumerToProducer(IntSet set, Operation consumer, Tensor tensor);

/*!
 * \brief Build loop nest for stage.
 *
 * \param stage The stage to create a loop nest.
 * \param dom_map The range of each iter var.
 * \param begin_iter_pos The beginning position of leaf_iter_vars to generate loop.
 * \param new_loop_var Whether create new loop variable.
 * \param skip_iter Whether skip certain iteration.
 * \param p_value_map The result value of each IterVar.
 * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1
 */
std::vector<std::vector<Stmt>> MakeLoopNest(const Stage& stage,
                                            const std::unordered_map<IterVar, Range>& dom_map,
                                            size_t begin_iter_pos, bool new_loop_var,
                                            const std::unordered_set<IterVar>& skip_iter,
                                            std::unordered_map<IterVar, PrimExpr>* p_value_map,
                                            bool debug_keep_trivial_loop);

std::vector<std::vector<Stmt>> MakeComputeOpLoopNest(
    const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map, size_t begin_iter_pos,
    bool new_loop_var, const std::unordered_set<IterVar>& skip_iter,
    std::unordered_map<IterVar, PrimExpr>* p_value_map, bool debug_keep_trivial_loop,
    Array<DimInfo> all_dimensions);

std::vector<std::vector<Stmt>> MakeScanOpLoopNest(
    const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map, size_t begin_iter_pos,
    bool new_loop_var, const std::unordered_set<IterVar>& skip_iter,
    std::unordered_map<IterVar, PrimExpr>* p_value_map, bool debug_keep_trivial_loop,
    Array<Dimension> explicit_dims);
/*!
 * \brief Create a nest of if checking the predicates.
 *
 * \param predicates The predicates to be checked.
 * \return List of If nest that checks the predicates.
 */
std::vector<Stmt> MakeIfNest(const std::vector<PrimExpr>& predicates);

/*!
 * \brief Replace the tensor reference (especially in Call's) in stmt by the replace map.
 * \param stmt The statement to be processed.
 * \param replace The replacement rule.
 */
Stmt ReplaceTensor(Stmt stmt, const std::unordered_map<Tensor, Tensor>& replace);
/*!
 * \brief Replace the tensor reference (especially in Call's) in stmt by the replace map.
 * \param expr The expression to be processed.
 * \param replace The replacement rule.
 */
PrimExpr ReplaceTensor(PrimExpr expr, const std::unordered_map<Tensor, Tensor>& replace);

UninterpFun ReplaceTensor(UninterpFun ufun, const std::unordered_map<Tensor, Tensor>& replace);
Modes ReplaceTensor(Modes mode, const std::unordered_map<Tensor, Tensor>& replace);

/*!
 * \brief Collect all tensors referenced in all expressions in the given array
 * \param collected_tensors The collected tensors.
 * \param exprs The expressions to look in.
 */
void CollectTensors(Array<Tensor>& collected_tensors, Array<PrimExpr> exprs);

/*!
 * \brief Substitute the variables of stmt by value map.
 * \param stmt the statment
 * \param value_map The value map.
 * \return Substituted result.
 */
Stmt Substitute(Stmt stmt, const std::unordered_map<IterVar, PrimExpr>& value_map);

/*!
 * \brief Converts Halide ForType to its corresponding IterVarType
 * \param for_type The ForType to be converted
 */
IterVarType ForTypeToIterVarType(tir::ForType for_type);

/*!
 * \brief Converts IterVarType to its corresponding Halide ForType
 * \param iter_type The IterVarType to be converted
 */
tir::ForType IterVarTypeToForType(IterVarType iter_type);

}  // namespace te
}  // namespace tvm
#endif  // TVM_TE_OPERATION_OP_UTIL_H_
