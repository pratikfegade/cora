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
 *  Lower allreduce to device implementable ir.
 * \file lower_thread_allreduce.cc
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_set>

#include "../../arith/compute_expr.h"
#include "../../runtime/thread_storage_scope.h"
#include "ir_util.h"

namespace tvm {
namespace tir {

class ThreadAllreduceBuilder final : public StmtExprMutator {
 public:
  explicit ThreadAllreduceBuilder(int warp_size, std::string target)
      : warp_size_(warp_size), target_(target) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      thread_extents_.push_back(op);
      // std::cout << "[REDSET] EXTENT " << op->node << std::endl;
      Stmt ret = StmtExprMutator::VisitStmt_(op);
      // std::cout << "[REDSET] EXTENTOFF " << op->node << std::endl;
      thread_extents_.pop_back();
      return ret;
    } else if (op->attr_key == attr::storage_scope) {
      Stmt ret = StmtExprMutator::VisitStmt_(op);
      op = ret.as<AttrStmtNode>();
      const VarNode* v = op->node.as<VarNode>();
      if (alloc_remap_.count(v)) {
        return op->body;
      } else {
        return ret;
      }
    } else if (op->attr_key == attr::reduce_scope) {
      const CommReducerNode* combiner = op->node.as<CommReducerNode>();
      CHECK(combiner);
      reduce_combiner_.push_back(combiner);
      Stmt ret = StmtExprMutator::VisitStmt_(op);
      reduce_combiner_.pop_back();
      return ret;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const EvaluateNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<EvaluateNode>();
    const CallNode* call = op->value.as<CallNode>();
    if (call && call->is_intrinsic(intrinsic::tvm_thread_allreduce)) {
      return MakeAllreduce(call);
    } else {
      return stmt;
    }
  }
  Stmt VisitStmt_(const AllocateNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<AllocateNode>();
    auto it = alloc_remap_.find(op->buffer_var.get());
    if (it != alloc_remap_.end()) {
      const AllocateNode* repl = it->second.as<AllocateNode>();
      if (warp_allocs_.count(repl)) {
        stmt = AllocateNode::make(repl->buffer_var, repl->dtype, repl->extents, repl->condition,
                                  op->body);
        stmt = AttrStmtNode::make(repl->buffer_var, attr::storage_scope,
                                  StringImmNode::make("local"), stmt);
      } else {
        // use volatile access to shared buffer.
        stmt = AttrStmtNode::make(repl->buffer_var, attr::volatile_scope, 1, op->body);
        stmt =
            AllocateNode::make(repl->buffer_var, repl->dtype, repl->extents, repl->condition, stmt);
        stmt = AttrStmtNode::make(repl->buffer_var, attr::storage_scope,
                                  StringImmNode::make("shared"), stmt);
      }
      return stmt;
    } else {
      return stmt;
    }
  }
  PrimExpr VisitExpr_(const LoadNode* op) final {
    auto it = load_remap_.find(op->buffer_var.get());
    if (it != load_remap_.end()) {
      CHECK(is_zero(op->index));
      return it->second;
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

 private:
  // Thread entry
  struct ThreadEntry {
    runtime::ThreadScope scope;
    IterVar iv;
    int extent;
    // comparator
    bool operator<(const ThreadEntry& other) const {
      return scope.dim_index < other.scope.dim_index;
    }
  };

  // Emit warp shuffle  calls.
  PrimExpr WarpShuffle(const std::string intrin, Var mask_var, PrimExpr val,
                       PrimExpr delta_or_lane) {
    PrimExpr pred = const_true(1);
    PrimExpr index(0);
    PrimExpr mask = LoadNode::make(DataType::UInt(32), mask_var, index, pred, tir::kAll);
    Array<PrimExpr> args{mask, val, delta_or_lane};
    return tir::CallNode::make(val.dtype(), intrin, args, tir::CallNode::PureIntrinsic);
  }

  std::pair<bool, int> is_warp_reduction(const std::vector<DataType>& types) const {
    // Only cuda target supports warp reductions.
    if ((target_ != "cuda")) return std::make_pair(false, -1);

    // Supported types:
    // {u}int, {u}long, {u}long long, float, double, half/half2
    if (std::any_of(types.begin(), types.end(), [](DataType ty) {
          if (ty.is_float16()) return ty.lanes() > 2;
          if (ty.is_vector()) return true;
          return ty.bytes() < 4 || ty.bytes() > 8;
        })) {
      return std::make_pair(false, -1);
    }
    if (thread_extents_.empty()) {
      return std::make_pair(false, -1);
    }

    const AttrStmtNode* op = thread_extents_.back();
    DCHECK_EQ(op->attr_key, attr::thread_extent);

    IterVar iv = Downcast<IterVar>(op->node);
    ThreadEntry e;
    e.scope = runtime::ThreadScope::make(iv->thread_tag);
    e.extent = 0;
    if (auto ptr = op->value.as<IntImmNode>()) {
      e.extent = static_cast<int>(ptr->value);
    }
    // std::cout << "[LAR] " << e.extent << " " << warp_size_ << " " << e.scope.dim_index << " "
    //           << e.scope.rank << std::endl;

    if ((e.extent & (e.extent - 1)) != 0 || e.extent == 0 || e.extent > warp_size_)
      return std::make_pair(false, -1);

    // std::cout << "[LAR] " << e.extent << " " << warp_size_ << " " << e.scope.dim_index << " "
    //           << e.scope.rank << std::endl;

    if (e.scope.dim_index == 0 && e.scope.rank == 1)
      return std::make_pair(true, e.extent);
    else {
      std::cout << "[ALLREDUCE] Could not use warp shuffles for cross thread reduction as "
                << "threadIdx.x is not bound to the reduction axis." << std::endl;
      return std::make_pair(false, -1);
    }
  }

  // make allreduce.
  Stmt MakeAllreduce(const CallNode* call) {
    global_red_idx_++;
    // std::cout << "[M_RED] " << GetRef<PrimExpr>(call) << std::endl;
    CHECK(!reduce_combiner_.empty());
    const CommReducerNode* combiner = reduce_combiner_.back();
    size_t size = combiner->result.size();

    const IntImmNode* size_of_args = call->args[0].as<IntImmNode>();
    CHECK(size_of_args) << call->args[0]->GetTypeKey();
    CHECK_EQ(size, size_of_args->value);
    Array<PrimExpr> inits = combiner->identity_element;
    std::vector<PrimExpr> values(size);
    std::vector<DataType> types(size);
    PrimExpr cond = call->args[size + 1];
    for (size_t idx = 0; idx < size; ++idx) {
      values[idx] = call->args[1 + idx];
      if (!is_one(cond)) {
        values[idx] = SelectNode::make(cond, values[idx], inits[idx]);
      }
      types[idx] = values[idx].dtype();
    }
    std::vector<const VarNode*> buffers(size);
    for (size_t idx = 0; idx < size; ++idx) {
      const VarNode* buffer = call->args[2 + size + idx].as<VarNode>();
      CHECK(buffer);
      buffers[idx] = buffer;
    }

    std::unordered_set<const VarNode*> reduce_set;
    for (size_t i = 2 + 2 * size; i < call->args.size(); ++i) {
      const VarNode* v = call->args[i].as<VarNode>();
      CHECK(v);
      reduce_set.insert(v);
    }
    size_t nmatch = 0;
    std::vector<ThreadEntry> vred, vpar;
    for (const AttrStmtNode* attr : thread_extents_) {
      ThreadEntry e;
      IterVar iv = Downcast<IterVar>(attr->node);
      e.scope = runtime::ThreadScope::make(iv->thread_tag);
      e.iv = iv;
      CHECK_LE(e.scope.rank, 1);
      CHECK_GE(e.scope.dim_index, 0) << "vthread do not work with cross thread reduction";
      if (e.scope.rank == 1) {
        CHECK(arith::GetConstInt(attr->value, &(e.extent)))
            << "Need constant extent for reduce set " << iv;
        if (reduce_set.count(iv->var.get())) {
          // std::cout << "[REDSETCHECK] " << iv->var.get() << " " << iv->var->name_hint <<
          // std::endl;
          vred.push_back(e);
          ++nmatch;
        } else {
          vpar.push_back(e);
        }
      }
    }
    CHECK_EQ(nmatch, reduce_set.size()) << "Not all reduce index are presented in the context";
    std::sort(vred.begin(), vred.end());
    std::sort(vpar.begin(), vpar.end());
    // the size of each index.
    int reduce_extent, group_extent;
    PrimExpr reduce_index = FlattenThread(vred, &reduce_extent);
    PrimExpr group_index = FlattenThread(vpar, &group_extent);
    std::vector<Stmt> seq;
    std::vector<Var> shared_bufs(size);
    std::vector<Stmt> local_vars;
    std::vector<Stmt> local_masks;
    //
    // This is an optimization. For small reduction sizes, it may be beneficial
    // for a single warp to performance the entire reduction. No trips to shared
    // memory and no cross warp synchronizations are required.
    // The following code emits the reduction as follows:
    //
    // Allocate reduction vars v[i], i = 0..size-1
    //
    // for offset from WARP_SIZE to 1 by 2
    //
    //   a    <- load(v[i])
    //   b    <- shuffle_down(load(v[i], offset))
    //   v[i] <- reduction(a, b)
    //
    // broadcast results from lane 0 to all other lanes and store
    // the final reduction result to the proper location.
    //
    auto p = is_warp_reduction(types);
    if (p.first) {
      // std::cout << "[LAR] Creating warp shuffle " << size << std::endl;
      // TODO(tvm-team) sub-warp reduction support.
      CHECK(reduce_extent <= warp_size_) << "not a warp reduction";
      //
      // This is the index to the reduction variable, one reduction
      // variable per warp. Local scope seems easier to reason without
      // relying on a pattern match pass to fix it later.
      PrimExpr index(0);

      for (size_t idx = 0; idx < size; ++idx) {
        shared_bufs[idx] = Var("red_buf" + std::to_string(idx), DataType::Handle());
        PrimExpr pred = const_true(types[idx].lanes());
        // seq.emplace_back(StoreNode::make(shared_bufs[idx], values[idx], index, pred, tir::kAll));

        // Uses a local variable to store the shuffled data.
        // Later on, this allocation will be properly attached to this statement.
        Var var("t" + std::to_string(idx) + "_" + std::to_string(global_red_idx_), types[idx]);
        Stmt s = AllocateNode::make(var, var.dtype(), {PrimExpr(1)}, pred, EvaluateNode::make(0));
        local_vars.push_back(s);
        seq.emplace_back(StoreNode::make(var, values[idx], index, pred, tir::kAll));
      }

      // The mask for this reducer, as this reducer may sit inside
      // a divergent control flow. Here it uses a variable to cache the current
      // active channels.
      //
      Var mask_var("mask" + std::to_string(global_red_idx_), DataType::UInt(32));
      {
        PrimExpr pred = const_true(1);
        PrimExpr mask = CallNode::make(DataType::UInt(32), tir::intrinsic::tvm_warp_activemask, {},
                                       tir::CallNode::PureIntrinsic, {}, 0);
        seq.emplace_back(StoreNode::make(mask_var, mask, index, pred, tir::kAll));
        // Push allocation with an empty body. Later this will be fixed
        // when the entire body is ready.
        auto stmt = AllocateNode::make(mask_var, mask_var->dtype, {PrimExpr(1)}, pred,
                                       EvaluateNode::make(0));
        local_vars.push_back(stmt);
        local_masks.push_back(stmt);
      }

      // Emit reductions within a warp.
      for (int offset = p.second / 2; offset > 0; offset /= 2) {
        // Load reduction values, no synchronization needed.
        Array<PrimExpr> a, b;
        for (size_t i = 0; i < size; ++i) {
          Var var = shared_bufs[i];
          PrimExpr pred = const_true(types[i].lanes());
          // PrimExpr val = LoadNode::make(types[i], var, index, pred, tir::kAll);
          const AllocateNode* repl = local_vars[i].as<AllocateNode>();
          PrimExpr val = LoadNode::make(types[i], repl->buffer_var, index, pred, tir::kAll);
          a.push_back(val);

          // __shfl_*sync calls shall not appear in if_then_else expressions
          // as this is causing extra divergency. E.g.
          //
          // v1 = (v2 < v3) ? v3 : __shfl_sync(mask, v1, 0);
          //
          // behaves differently from
          //
          // int t = __shfl_sync(mask, v1, 0);
          // v1 = (v2 < v3) ? v3 : t;
          //
          // The former may cause dead lock as there is a divergent
          // branch with a warp sync call inside.
          //
          PrimExpr other =
              val + WarpShuffle(tir::intrinsic::tvm_warp_shuffle_down, mask_var, val, offset);
          Stmt s = StoreNode::make(repl->buffer_var, other, index, pred, tir::kAll);
          seq.push_back(s);

          PrimExpr load = LoadNode::make(types[i], repl->buffer_var, index, pred, tir::kAll);
          b.push_back(load);
        }

        // // Do reductions.
        // Array<PrimExpr> ret = (*combiner)(a, b);

        // // Store the reduction result to itself.
        // std::vector<Stmt> stores(size);
        // for (size_t i = 0; i < size; ++i) {
        //   Var var = shared_bufs[i];
        //   PrimExpr pred = const_true(types[i].lanes());
        //   stores[i] = StoreNode::make(var, ret[i], index, pred, tir::kAll);
        // }
        // seq.push_back(SeqStmt::Flatten(stores));
      }

      // Broadcast the reduction result from lane 0 to all other lanes.
      // This avoids to emit predicated stores, as all threads are
      // uniformly writting the same result.
      //
      for (size_t i = 0; i < size; ++i) {
        const AllocateNode* repl = local_vars[i].as<AllocateNode>();
        Var var = repl->buffer_var;
        PrimExpr pred = const_true(types[i].lanes());
        PrimExpr val = LoadNode::make(types[i], var, index, pred, tir::kAll);
        PrimExpr lane_id = indexdiv(indexmod(get_reduction_group_id(), 32), p.second) * p.second;
        PrimExpr splat = WarpShuffle(tir::intrinsic::tvm_warp_shuffle, mask_var, val, lane_id);
        seq.push_back(StoreNode::make(var, splat, index, pred, tir::kAll));
      }

      // Update existing allocations.
      for (size_t i = 0; i < size; ++i) {
        CHECK(!load_remap_.count(buffers[i]));
        PrimExpr pred = const_true(types[i].lanes());
        const AllocateNode* repl = local_vars[i].as<AllocateNode>();
        Var var = repl->buffer_var;
        load_remap_[buffers[i]] = LoadNode::make(types[i], var, index, pred, tir::kAll);
        Array<PrimExpr> extents{PrimExpr(1)};
        auto node = AllocateNode::make(var, types[i], extents, pred, EvaluateNode::make(0));
        alloc_remap_[buffers[i]] = node;
        warp_allocs_.insert(node.get());
      }
    } else {
      int threadx_extent = 1;
      if (reduce_extent == 1) {
        // special case, no reduction is needed.
        std::vector<Stmt> stores(size);
        for (size_t i = 0; i < size; ++i) {
          PrimExpr pred = const_true(types[i].lanes());
          Var buffer_var = Downcast<Var>(call->args[2 + size + i]);
          stores[i] = StoreNode::make(buffer_var, values[i], 0, pred, tir::kAll);
        }
        return SeqStmt::Flatten(stores);
      }
      // Whether the threadIdx.x is involved in reduction.
      // if (vred[0].scope.dim_index == 0) {
      if (vred[0].scope.dim_index == 0) {
        threadx_extent = vred[0].extent;
      }
      // This sync is necessary because there might be incomplete read of
      // previous iteration on the same buffer.
      seq.emplace_back(SyncThread("shared"));
      for (size_t idx = 0; idx < size; ++idx) {
        shared_bufs[idx] = Var("red_buf" + std::to_string(idx), DataType::Handle());
        PrimExpr pred = const_true(types[idx].lanes());
        seq.emplace_back(StoreNode::make(shared_bufs[idx], values[idx],
                                         BufIndex(reduce_index, group_index, reduce_extent), pred,
                                         tir::kAll));
      }
      seq.emplace_back(SyncThread("shared"));
      seq.emplace_back(MakeBufAllreduce(combiner, types, shared_bufs, reduce_index, group_index,
                                        reduce_extent, threadx_extent));
      for (size_t idx = 0; idx < size; ++idx) {
        CHECK(!load_remap_.count(buffers[idx])) << " " << buffers[idx]->name_hint;
        PrimExpr pred = const_true(types[idx].lanes());
        load_remap_[buffers[idx]] = LoadNode::make(
            types[idx], shared_bufs[idx],
            BufIndex(make_zero(reduce_index.dtype()), group_index, reduce_extent), pred, tir::kAll);
        alloc_remap_[buffers[idx]] = AllocateNode::make(
            shared_bufs[idx], types[idx], {PrimExpr(group_extent), PrimExpr(reduce_extent)}, pred,
            EvaluateNode::make(0));
      }
    }
    // Fix all local allocations as all statements are built.
    Stmt body = SeqStmt::Flatten(seq);
    for (auto var : local_masks) {
      const AllocateNode* repl = var.as<AllocateNode>();
      if (repl) {
        body =
            AllocateNode::make(repl->buffer_var, repl->dtype, repl->extents, repl->condition, body);
        body = AttrStmtNode::make(repl->buffer_var, attr::storage_scope,
                                  StringImmNode::make("local"), body);
      }
    }
    return body;
  }

  // make allreduce.
  Stmt MakeBufAllreduce(const CommReducerNode* combiner, const std::vector<DataType>& types,
                        const Array<Var>& shared_bufs, PrimExpr reduce_index, PrimExpr group_index,
                        int reduce_extent, int threadx_extent) {
    // Get next power of two
    int reduce_align = 1;
    while (reduce_extent > reduce_align) {
      reduce_align = reduce_align << 1;
    }
    CHECK_GT(reduce_align, 1);
    std::vector<Stmt> seq;

    size_t size = shared_bufs.size();
    PrimExpr buf_index = BufIndex(reduce_index, group_index, reduce_extent);
    // make reduction
    auto freduce = [&](int offset) {
      Array<PrimExpr> a, b;
      for (size_t i = 0; i < size; ++i) {
        b.push_back(LoadNode::make(types[i], shared_bufs[i],
                                   BufIndex(reduce_index + offset, group_index, reduce_extent),
                                   const_true(), tir::kAll));
        a.push_back(LoadNode::make(types[i], shared_bufs[i], buf_index, const_true(), tir::kAll));
      }
      Array<PrimExpr> ret = (*combiner)(a, b);
      std::vector<Stmt> stores(size);
      for (size_t i = 0; i < size; ++i) {
        stores[i] = StoreNode::make(shared_bufs[i], ret[i], buf_index, const_true(), tir::kAll);
      }
      return SeqStmt::Flatten(stores);
    };
    // Step one, check for
    if (reduce_align > reduce_extent) {
      // reduction with the boundary condition
      reduce_align = reduce_align >> 1;
      PrimExpr cond = reduce_index < (reduce_extent - reduce_align);
      seq.emplace_back(IfThenElseNode::make(cond, freduce(reduce_align)));
      seq.emplace_back(SyncThread("shared"));
    }
    CHECK(threadx_extent >= 1 && warp_size_ >= 1);
    // normal synchronization
    while (reduce_align > threadx_extent || reduce_align > warp_size_) {
      reduce_align = reduce_align >> 1;
      PrimExpr cond = reduce_index < reduce_align;
      seq.emplace_back(IfThenElseNode::make(cond, freduce(reduce_align)));
      seq.emplace_back(SyncThread("shared"));
    }
    // in warp synchronization.
    std::vector<Stmt> in_warp_seq;
    PrimExpr in_warp_cond = reduce_index < (reduce_align >> 1);
    while (reduce_align > 1) {
      reduce_align = reduce_align >> 1;
      in_warp_seq.emplace_back(freduce(reduce_align));
      seq.emplace_back(SyncThread("warp"));
    }
    if (in_warp_seq.size() != 0) {
      Stmt warp_body = SeqStmt::Flatten(in_warp_seq);
      seq.emplace_back(IfThenElseNode::make(in_warp_cond, warp_body));
      seq.emplace_back(SyncThread("shared"));
    }
    return SeqStmt::Flatten(seq);
  }
  // Flatten the thread index.
  // Also return a warp number,
  PrimExpr FlattenThread(const std::vector<ThreadEntry>& tvec, int* out_total_extent) {
    int& total_extent = *out_total_extent;
    total_extent = 1;
    if (tvec.size() == 0) {
      return make_zero(DataType::Int(32));
    }

    PrimExpr ret;
    for (const ThreadEntry& e : tvec) {
      if (ret.defined()) {
        ret = ret + e.iv->var * total_extent;
      } else {
        CHECK_EQ(total_extent, 1);
        ret = e.iv->var;
      }
      total_extent *= e.extent;
    }
    return ret;
  }
  // sync thread op.
  static Stmt SyncThread(const std::string& sync) {
    return EvaluateNode::make(CallNode::make(DataType::Int(32), intrinsic::tvm_storage_sync,
                                             {StringImmNode::make(sync)}, CallNode::Intrinsic));
  }
  // The local buffer index.
  static PrimExpr BufIndex(PrimExpr reduce_index, PrimExpr group_index, int reduce_extent) {
    if (!is_zero(group_index)) {
      return tir::Simplify(group_index * reduce_extent + reduce_index);
    } else {
      return reduce_index;
    }
  }
  // The warp size of the device.
  int warp_size_{1};
  // The target.
  std::string target_;
  int global_red_idx_{1};

  PrimExpr get_reduction_group_id() {
    PrimExpr threadx = 0;
    PrimExpr thready = 0;
    PrimExpr threadz = 0;

    PrimExpr y_extent = 0;
    PrimExpr x_extent = 0;
    for (auto op : thread_extents_) {
      DCHECK_EQ(op->attr_key, attr::thread_extent);

      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->var->name_hint == "threadIdx.x") {
        threadx = iv->var;
        if (auto ptr = op->value.as<IntImmNode>()) {
          x_extent = static_cast<int>(ptr->value);
        }
      } else if (iv->var->name_hint == "threadIdx.y") {
        thready = iv->var;
        if (auto ptr = op->value.as<IntImmNode>()) {
          y_extent = static_cast<int>(ptr->value);
        }
      } else if (iv->var->name_hint == "threadIdx.z") {
        threadz = iv->var;
      }
    }

    return threadz * y_extent * x_extent + thready * x_extent;
  }

  // surrounding scope of thread extent.
  std::vector<const AttrStmtNode*> thread_extents_;
  std::vector<const CommReducerNode*> reduce_combiner_;
  // The load remap
  std::unordered_map<const VarNode*, PrimExpr> load_remap_;
  // Allocate remap
  std::unordered_map<const VarNode*, Stmt> alloc_remap_;
  // Allocate from warp reductions
  std::unordered_set<const void*> warp_allocs_;
};

LoweredFunc LowerThreadAllreduce(LoweredFunc f, int warp_size, std::string target) {
  CHECK_NE(f->func_type, kHostFunc);
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  // std::cout << "[RED] Reduce input body\n " << n->body << std::endl;
  n->body = ThreadAllreduceBuilder(warp_size, target)(n->body);
  return LoweredFunc(n);
}
}  // namespace tir
}  // namespace tvm
