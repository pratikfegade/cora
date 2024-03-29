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
 * \file tvm/tir/ir_pass.h
 * \brief Collection of IR pass functions
 *
 *  When the pass functions in this file are for Stmt,
 *  we can use PassFunction(Evaluate(expr)) to apply it to Expr
 */
#ifndef TVM_TIR_IR_PASS_H_
#define TVM_TIR_IR_PASS_H_

#include <tvm/te/schedule.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/lowered_func.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tir {

/*!
 * \brief Simplify the expression.
 * \param expr The expression to be simplifed.
 * \param vrange The range information about the variable.
 * \return Canonicalized statement.
 */
TVM_DLL PrimExpr Simplify(PrimExpr expr, Map<Var, Range> vrange = Map<Var, Range>());

/*!
 * \brief Simplify the statement.
 * \param stmt The statement to be simplifed.
 * \param vrange The range information about the variable.
 * \return Canonicalized statement.
 */
Stmt Simplify(Stmt stmt, Map<Var, Range> vrange = Map<Var, Range>());

/*!
 * \brief Simplify by applying canonical form.
 * \param stmt The statement to be canonically simplifed.
 * \param vrange The range information about the variable.
 * \return Canonicalized statement.
 */
Stmt CanonicalSimplify(Stmt stmt, Map<Var, Range> vrange = Map<Var, Range>());

/*!
 * \brief Simplify by applying canonical form.
 * \param expr The statement to be canonically simplifed.
 * \param vrange The range information about the variable.
 * \return Canonicalized expression.
 */
TVM_DLL PrimExpr CanonicalSimplify(PrimExpr expr, Map<Var, Range> vrange = Map<Var, Range>());

/*!
 * \brief Deep compare lhs and rhs
 * \param lhs The left operand
 * \param rhs The right operand
 * \return The comparison result.
 */
TVM_DLL bool Equal(const PrimExpr& lhs, const PrimExpr& rhs);

/*!
 * \brief Deep compare lhs and rhs
 * \param lhs The left operand
 * \param rhs The right operand
 * \return The comparison result.
 */
bool Equal(const Stmt& lhs, const Stmt& rhs);

/*!
 * \brief Deep compare lhs and rhs.
 *
 *  If you only want equality comparison, use Equal
 *  which will also tie definitions. The compare mode
 *  will give order of expression in total order.
 *
 * \param lhs The left operand
 * \param rhs The right operand
 * \return The comparison result.
 */
int Compare(const PrimExpr& lhs, const PrimExpr& rhs);

/*!
 * \brief verifies whether the IR stmt or Expr is in SSA form.
 *  That is: each VarExpr is defined and assigned once(in Let/For)
 *
 * \param ir The root of the IR DAG.
 * \return Whether IR is in SSA form.
 * \note All the passes in this file uses SSA form and outputs SSA form.
 */
TVM_DLL bool VerifySSA(const Stmt& ir);

/*!
 * \brief Whether the expression have side effect.
 * \return whether expression have side effect
 */
TVM_DLL bool HasSideEffect(const PrimExpr& e);

/*!
 * \brief Whether e expression used var.
 * \param e The expression to be checked.
 * \param v The variable.
 * \return Whether e uses v.
 */
bool ExprUseVar(const PrimExpr& e, const Var& v);

/*!
 * \brief Whether e expression used any var in variable set..
 * \param e The expression to be checked.
 * \param vset The variable set.
 * \return Whether e uses vset.
 */
bool ExprUseVar(const PrimExpr& e, const std::unordered_set<const VarNode*>& vset);

/*!
 * \brief Convert a IR node to be SSA form.
 * \param stmt The source statement to be converted.
 * \return The converted form.
 */
TVM_DLL Stmt ConvertSSA(Stmt stmt);

/*!
 * \brief Perform hfusion based on the group ids in the stmt.
 * \param stmt The source statement to be converted.
 * \return The converted form.
 */
TVM_DLL LoweredFunc HorizontalFuse(LoweredFunc stmt);

/*!
 * \brief Substitute the var specified in key->var to be value.
 * \param stmt The source statement to be substituted
 * \param value_map The map of new values.
 * \return The converted form.
 */
Stmt Substitute(Stmt stmt, const std::unordered_map<const VarNode*, PrimExpr>& value_map);

/*!
 * \brief Substitute the var specified in key->var to be value.
 * \param expr The source expression to be substituted
 * \param value_map The map of new values.
 * \return The converted expression.
 */
PrimExpr Substitute(PrimExpr expr, const std::unordered_map<const VarNode*, PrimExpr>& value_map);

/*!
 * \brief Substitute the var specified in key->var to be value.
 * \param stmt The source statement to be substituted
 * \param value_map The map of new values.
 * \return The converted form.
 */
Stmt Substitute(Stmt stmt, const Map<Var, PrimExpr>& value_map);

/*!
 * \brief Substitute the var specified in key->var to be value.
 * \param expr The source expression to be substituted
 * \param value_map The map of new values.
 * \return The converted expression.
 */
PrimExpr Substitute(PrimExpr expr, const Map<Var, PrimExpr>& value_map);

/*!
 * \brief inline all calls of f in stmt.
 *
 * \param stmt The statement to apply inline optimization.
 * \param f The function reference to be inlined
 * \param args The arguments variable of the function.
 * \param body The definition body of the function.
 * \return The result stmt
 *
 * \note All the passes in this file uses SSA form and outputs SSA form.
 */
Stmt Inline(Stmt stmt, FunctionRef f, Array<Var> args, PrimExpr body, Map<Var, PrimExpr> vmap = {});

/*!
 * \brief inline lets with no side effects.
 *
 * \param stmt The statement to apply inline optimization.
 * \return The result stmt
 *
 */
Stmt InlineLets(Stmt stmt);

/*!
 * \brief Flatten the multi-dimensional read/write
 *  to single dimensional Load/Store
 *
 * \param stmt The stmt to be trasnformed.
 * \param extern_buffer Map specifies external
 *    buffer assignment of input and outputs.
 * \param cache_line_size The size of CPU cache line.
 * \param create_bound_attribute Whether to create bound attributes.
 * \return Transformed stmt.
 */
Stmt StorageFlatten(Stmt stmt, Map<te::Tensor, Buffer> extern_buffer, int cache_line_size,
                    bool create_bound_attribute = false);

/*!
 * \brief Try to modify the AST to support TensorCore
 *
 * \param stmt The stmt to be trasnformed.
 * \param schedule The original schedule.
 * \param extern_buffer Map specifies external
 *    buffer assignment of input and outputs.
 * \return Transformed stmt.
 */
Stmt RewriteForTensorCore(Stmt stmt, te::Schedule schedule, Map<te::Tensor, Buffer> extern_buffer);

/*!
 * \brief Verify if there is any argument bound to compact buffer.
 *
 * \param stmt The stmt to be verified.
 * \return true if there is any buffer_bind_scope attribute found,
 *        otherwise, false.
 */
bool VerifyCompactBuffer(Stmt stmt);

/*!
 * \brief Remove No Op from the Stmt.
 * \param stmt The stmt to be trasnformed
 * \return Transformed stmt.
 */
Stmt RemoveNoOp(Stmt stmt);

/*!
 * \brief unroll the constant loop marked by unroll.
 * This pass also automatically attaches pragma unroll tag to loops which meets the standard.
 *
 * \param stmt The statment to be unrolled.
 * \param auto_max_step The maximum step before stop attach automatic unroll
 * \param auto_max_depth The maximum depth before stop attach automatic unroll
 * \param auto_max_extent The maximum extent of the loop we can unroll,
 *                     this is an legacy option that do not take the loop total steps into account.
 * \param explicit_unroll Whether explicitly unroll the loop, or leave unroll annotation to codegen.
 * \return Transformed stmt.
 */
Stmt UnrollLoop(Stmt stmt, int auto_max_step, int auto_max_depth, int auto_max_extent,
                bool explicit_unroll);

/*!
 * \brief peel the loop marked by unroll.
 *
 * \param stmt The statment to be peeled.
 * \return Transformed stmt.
 */
// Stmt PeelLoop(Stmt stmt);
LoweredFunc PeelLoop(LoweredFunc stmt);

/*!
 * \brief Add env loops for CPU (c and llvm, for now) targets.
 *
 * \param stmt The statment to be transformed.
 * \return Transformed stmt.
 */
Stmt CreateEnvLoopsForStmt(Stmt stmt, std::string target);
LoweredFunc CreateEnvLoopsForFunc(LoweredFunc f, std::string target);

/*!
 * \brief vectorize the constant loops
 * \param stmt The statement to be vectorized.
 * \return Transformed stmt.
 */
Stmt VectorizeLoop(Stmt stmt);

/*!
 * \brief convert vectorized loops into serialized loops
 * \param stmt The statement to skip vectorization on.
 * \return Transformed stmt.
 */
Stmt SkipVectorize(Stmt stmt);

/*!
 * \brief instruments bound checkers.
 * \param stmt The statement to be instrumented.
 * \return Instrumented stmt.
 */
Stmt InstrumentBoundCheckers(Stmt stmt);

/*!
 * \brief Inject virtual thread loops into stmt.
 * \param stmt The statement to be transformed.
 * \return Transformed stmt.
 */
Stmt InjectVirtualThread(Stmt stmt);

/*!
 * \brief Inject prefetch instructions into stmt.
 * \param stmt The statement to be transformed.
 * \return Transformed stmt.
 */
Stmt InjectPrefetch(Stmt stmt);

/*!
 * \brief Inject double buffer into stmt.
 * \param stmt The statement to be transformed.
 * \param split_loop Loop splitting factor.
 * \return Transformed stmt.
 */
Stmt InjectDoubleBuffer(Stmt stmt, int split_loop);

/*!
 * \brief Inject copy intrinsics with optional pad.
 *
 * \param stmt The statement to be transformed.
 * \param pragma_key The pragma key for hint of copy.
 * \param fintrin The function with signature
 *
 *   Stmt fintrin(Buffer src,
 *                Buffer dst,
 *                Array<Expr> pad_before,
 *                Array<Expr> pad_after,
 *                Expr pad_value)
 * \return Transformed stmt.
 */
Stmt InjectCopyIntrin(Stmt stmt, const std::string& pragma_key, const runtime::PackedFunc& fintrin);

/*!
 * \brief Rewrite storage allocation pattern.
 *  Moves the allocation to outer most possible scope.
 *  Trying to share space between allocations to make
 *  a static allocation plan when possible.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt StorageRewrite(Stmt stmt);

/*!
 * \brief partition loops in the stmt
 * \param stmt The stmt to do loop partition
 * \param split_const_loop flag to enable partition for const loop
 * \return Transformed stmt.
 */
Stmt LoopPartition(Stmt stmt, bool split_const_loop);
Stmt RemoveLikelyTags(Stmt stmt);

/*!
 * \brief Detect and insert sync points to co-processor.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt CoProcSync(Stmt stmt);

/*!
 * \brief Lift common attrs with attr_key to outer scope.
 *
 * \param stmt The stmt to be transformed
 * \param attr_key The attribute key to be checked.
 * \return Transformed stmt.
 */
Stmt LiftAttrScope(Stmt stmt, std::string attr_key);

/*!
 * \brief Detect and rewrite unsafe select that contains memory access.
 * \param stmt The statement to be rewritten.
 * \return Transformed stmt.
 */
Stmt RewriteUnsafeSelect(Stmt stmt);

/*!
 * \brief Lower attached storage access information.
 * Do this pass after all storage access analysis finish.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt LowerStorageAccessInfo(Stmt stmt);

/*!
 * \brief Decorate the stmt with a device scope, this is helpful for
 * hardware accelerator without thread blocks.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt DecorateDeviceScope(Stmt stmt);

/*!
 * \brief Loop invariant code motion which locates and hoists if statements.
 * \param stmt The stmt to do if statement hoisting.
 * \return Transformed stmt.
 */
Stmt HoistIfThenElse(Stmt stmt);

/*!
 * \brief More aggresive loop invariant code motion which locates and hoists if statements.
 * \param stmt The stmt to do if statement hoisting.
 * \return Transformed stmt.
 */
LoweredFunc BetterHoistIfThenElse(LoweredFunc f, std::string target, Array<PrimExpr> constraints);
Stmt BetterHoistIfThenElseStmt(Stmt f, std::string target, Array<PrimExpr> constraints);

/*!
 * \brief Hoist loop invariant buffer loads.
 * \param f The func to work on.
 * \return Transformed func.
 */
LoweredFunc HoistLoads(LoweredFunc f);

/*!
 * \brief Remove redundant if conditions
 * \param func The func to optimize.
 * \param target The target.
 * \return Transformed stmt.
 */
LoweredFunc RemoveRedundantIfsFromFunc(LoweredFunc f, std::string target,
                                       Array<PrimExpr> constraints);

/*!
 * \brief Remove redundant if conditions
 * \param stmt The stmt to optimize.
 * \return Transformed stmt.
 */
Stmt RemoveRedundantIfs(Stmt stmt, Array<PrimExpr> constraints);

/*!
 * \brief Expand intrisic if then else expressions.
 * \param stmt The stmt.
 * \return Transformed stmt.
 */
Stmt ExpandIntrinsicITE(Stmt stmt);

class MakeAPIResult;

class MakeAPIResultNode : public runtime::Object {
 public:
  LoweredFunc function;
  Array<Buffer> host_intermediate_buffers;
  Array<Buffer> device_intermediate_buffers;

  TVM_DLL static MakeAPIResult make(LoweredFunc function, Array<Buffer> host_intermediate_buffers,
                                    Array<Buffer> device_intermediate_buffers);

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("function", &function);
    v->Visit("host_intermediate_buffers", &host_intermediate_buffers);
    v->Visit("device_intermediate_buffers", &device_intermediate_buffers);
  }

  static constexpr const char* _type_key = "tir.MakeAPIResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(MakeAPIResultNode, Object);
};

class MakeAPIResult : public runtime::ObjectRef {
 public:
  MakeAPIResult() {}
  // construct from shared ptr.
  explicit MakeAPIResult(runtime::ObjectPtr<runtime::Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const MakeAPIResultNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = MakeAPIResultNode;
};

inline const MakeAPIResultNode* MakeAPIResult::operator->() const {
  return static_cast<const MakeAPIResultNode*>(data_.get());
}

/*!
 * \brief Make an user callable API LoweredFunc.
 *
 *  The main task of this function is to create code to :
 *   - Map the values in the api_args to Var that is required by body.
 *   - Insert assertions to check type/value of the passed arguments.
 *
 * \param body The body of the function.
 * \param name The name of the function.
 * \param api_args Arguments to the function, can be either Var, or Buffer
 * \param num_unpacked_args Number of arguments that
 *         are processed in plain form instead of packed form.
 * \param is_restricted Whether the caller can guarantee that each buffer argument do not overlap.
 *  It is recommended to set to true for optimized code if such invariant holds.
 *
 * \return a LoweredFunc with the specified signiture.
 *
 * \note
 *  The function signature have two cases
 *
 *  let num_packed_args = len(api_args) - num_unpacked_args;
 *
 *  if num_packed_args is zero:
 *     f(api_arg_0, api_arg_1, .., api_arg_n) where n == len(api_args)
 *
 *  if num_packed_args is not zero:
 *       f(TVMArg* packed_args, int* packed_arg_type_ids, int num_packed_args,
 *         api_arg_k, api_arg_k+1, ... api_arg_n,
 *         TVMValue* out_ret_val, int* out_ret_tcode)
 *
 *       where n == len(api_args), k == num_packed_args
 *
 *  There is no thread_axis in generated function.
 */

enum class PrepCodeMode {
  kWithPrepCode = 1,
  kNoPrepCode = 2,
  kOnlyPrepCode = 3,
};
MakeAPIResult MakeAPI(Stmt body, std::string name, Array<ObjectRef> length_api_args,
                      Array<ObjectRef> tensor_api_args, int num_unpacked_args, bool is_restricted,
                      PrepCodeMode prep_code_mode);

/*!
 * \brief Bind the device type of host function to be device_type.
 * \param func The function to be binded.
 * \param device_type The device type to be binded.
 * \return The binded function.
 */
LoweredFunc BindDeviceType(LoweredFunc func, int device_type);
/*!
 * \brief Find undefined vars in the statment.
 * \param stmt The function to be checked.
 * \param defs The vars that is defined.
 * \return Array of undefined vars.
 */
Array<Var> UndefinedVars(const Stmt& stmt, const Array<Var>& defs);

/*!
 * \brief Split the function into a host function and device functions.
 * \param func The function to be splitted.
 *
 * \return Array of functions, the first one is host function,
 *     the others are device functions.
 */
Array<LoweredFunc> SplitHostDevice(LoweredFunc func, std::string grid_sync_str = "");

/*!
 * \brief Insert sync between parallel read/write of shared buffers.
 *
 * \param stmt The stmt to be trasnformed.
 * \param storage_scope The storage scope considered.
 */
LoweredFunc ThreadSync(LoweredFunc stmt, std::string storage_scope, std::string target);

/*!
 * \brief Lower cross thread alleduce in the stmt.
 * \param f The device function to be lowered.
 * \param warp_size the size of warp where no sync is needed.
 * \return Transformed function.
 */
LoweredFunc LowerThreadAllreduce(LoweredFunc f, int warp_size, std::string target);

/*!
 * \brief Lower warp memory in stmt.
 * \param f The device function to be lowered.
 * \param warp_size the size of warp where no sync is needed.
 *        this function will only take in effect if warp_size is bigger than one.
 * \return Transformed function.
 */
LoweredFunc LowerWarpMemory(LoweredFunc f, int warp_size);

/*!
 * \brief Remap the thread axis
 *
 *  This can be used to get equivalent program which uses
 *  threadIdx.y in place of threadIdx.x by passing
 *  {"threadIdx.x": thread_axis("threadIdx.y")}
 *
 *
 * \param f The device function to be lowered.
 * \param axis_map The map from StringImm -> ItrVar
 * \return Transformed function.
 */
LoweredFunc RemapThreadAxis(LoweredFunc f, Map<PrimExpr, IterVar> axis_map);

/*!
 * \brief Lower packed function call.
 * \param f The function to be lowered.
 * \return Transformed function.
 */
LoweredFunc LowerTVMBuiltin(LoweredFunc f);

/*!
 * \brief Combine context function calls.
 * \param f The host function to be lowered.
 * \return Transformed function.
 */
LoweredFunc CombineContextCall(LoweredFunc f);

/*!
 * \brief Rewrite the pointer content type of arguments,
 *  as well as Alloc internal to the function to use
 *  the most frequently accessed type for load/store
 *  to avoid pointer casting in backend when possible.
 *
 * \note implemeneted in storage_rewrite.cc
 * \param f The function to be trasnformed
 * \return Transformed function.
 */
LoweredFunc PointerValueTypeRewrite(LoweredFunc f);

/*!
 * \brief Lower attached storage access information on device.
 * Do this pass after all storage access analysis finish.
 *
 * \param func The device function to be lowered.
 * \return Transformed function.
 */
LoweredFunc LowerDeviceStorageAccessInfo(LoweredFunc func);

/*!
 * \brief Lower intrinsic function calls.
 * \param f The device function to be lowered.
 * \param target The target device.
 * \return Transformed function.
 */
LoweredFunc LowerIntrin(LoweredFunc f, const std::string& target);

/*!
 * \brief Lower custom datatypes.
 *
 * See tvm::datatypes::Registry for more information on adding custom datatypes.
 *
 * \param f The device function to be lowered.
 * \param target The target device.
 * \return Transformed function.
 */
LoweredFunc LowerCustomDatatypes(LoweredFunc f, const std::string& target);

/*!
 * \brief Infer the TensorCore fragment infomation using tensor intrinsics
 *
 * \param f The device function to be lowered.
 * \return Transformed function.
 */
LoweredFunc InferFragment(LoweredFunc f);

/*!
 * \brief skip assert stmt generation
 * \param f The function to be transformed.
 * \return Transformed function.
 */
LoweredFunc SkipAssert(LoweredFunc f);

/*!
 * \brief Verify if memory accesses are legal for a specific target device type.
 *
 *  In the case that tgt is cuda, if not all workload is bound with
 *  threads, CPU code is generated that tries to access GPU memory,
 *  which is illegal. This pass performs verification for this case.
 *
 * \param func The function to be verified.
 * \param device_type The target device type.
 * \return Success of memory verification.
 */
bool VerifyMemory(LoweredFunc func, int device_type);

/*!
 * \brief Verify the correctness of a GPU code
 *        It will check the whether the amount of memory usage or the number of threads
 *        in a block exceeds the limit
 * \param stmt The statement to be checked
 * \param constraints The dict to specify constraints to check.
 *        Possible keys are
 *
 *        "max_local_memory_per_block": Total amount of local memory per block (in bytes).
 *        "max_shared_memory_per_block": Total amount of shared memory per block (in bytes).
 *        "max_threads_per_block": Maximum number of threads per block.
 *        "max_thread_x": Maximum length of threadIdx.x.
 *        "max_thread_y": Maximum length of threadIdx.y.
 *        "max_thread_z": Maximum length of threadIdx.z.
 *
 *        If one key is missing in this argument, the pass won't check for that item.
 * \return valid Whether it is a valid GPU code
 *
 */
bool VerifyGPUCode(Stmt stmt, Map<std::string, PrimExpr> constraints);

/*!
 * \brief Remove ProducerConsumerNodes as they make if hoisting and
 * hfusion harder
 *
 * \param func The func to be processed.
 * \return return func.
 */
LoweredFunc RemoveProducerConsumerNodes(LoweredFunc func);

/*!
 * \brief Substitute thread vars for better load balancing
 *
 * \param to_substitute_in Substitute only in these functions
 * \param stmt The stmt to be processed.
 * \return return stmt.
 */
Stmt SubstituteThreadVars(Stmt stmt, Array<FunctionRef> to_substitute_in, Map<std::string, FunctionRef> vsub_map);
LoweredFunc SubstituteThreadVarsFunc(LoweredFunc func, Array<FunctionRef> to_substitute_in, Map<std::string, FunctionRef> vsub_map);

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_IR_PASS_H_
