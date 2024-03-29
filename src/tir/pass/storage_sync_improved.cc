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
 * \file storage_sync.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <unordered_set>

#include "../../arith/const_fold.h"
#include "../../runtime/thread_storage_scope.h"
#include "../../tir/ir/var_replacer.h"
#include "ir_util.h"
#include "storage_access.h"

namespace tvm {
namespace tir {
class ThreadSyncPlanner : public StorageAccessVisitor {
 public:
  explicit ThreadSyncPlanner(StorageScope sync_scope) : sync_scope_(sync_scope) {}

  // The syncs inserted before each statement
  std::unordered_set<const Object*> syncs_inserted_;

 protected:
  bool Enabled(const VarNode* buf, const StorageScope& scope) const final {
    return in_device_env() && scope == sync_scope_;
  }
  // Plan the sync
  std::vector<AccessEntry> Summarize(std::vector<StmtEntry> seq, const ForNode* loop) final {
    // if (loop) std::cout << "[SYNC]   OLLA " << loop->loop_var << " " << seq.size() << std::endl;
    // Unsynced reads and writes
    std::vector<AccessEntry> reads;
    std::vector<AccessEntry> writes;
    // if it is a loop, rotate two times to consider effect of loop.
    // simulation based approach to find dependenceies
    for (size_t i = 0; i < seq.size(); ++i) {
      const StmtEntry& s = seq[i];
      // check if sync before statement is needed.
      bool sync_before_stmt = (syncs_inserted_.count(s.stmt) != 0);
      // Apply the syncs added already.
      if (sync_before_stmt) {
        reads.clear();
        writes.clear();
      }
      for (const AccessEntry& acc : s.access) {
        if (acc.type == kRead) {
          if (FindConflict(writes, acc, false)) {
            // std::cout << "[SYNC]   ALLO " << acc.buffer << std::endl;
            sync_before_stmt = true;
            break;
          }
        } else if (acc.type == kWrite) {
          if (FindConflict(reads, acc, false)) {
            // std::cout << "[SYNC]   ALLO " << acc.buffer << std::endl;
            sync_before_stmt = true;
            break;
          }
        } else if (acc.type == kSync) {
          reads.clear();
          writes.clear();
        }
      }
      // If sync is inserted. remove the irrelevant things.
      if (sync_before_stmt) {
        reads.clear();
        writes.clear();
      }
      // Add the read/write of current statement
      for (const AccessEntry& acc : s.access) {
        if (acc.type == kRead) {
          reads.push_back(acc);
        } else if (acc.type == kWrite) {
          writes.push_back(acc);
        } else if (acc.type == kSync) {
          reads.clear();
          writes.clear();
        }
      }
      if (sync_before_stmt) {
        // CHECK_EQ(condition_counter(), 0) << "Cannot insert syncs inside condition "
        //                                  << GetRef<Stmt>(static_cast<const StmtNode*>(s.stmt));
        if (sync_scope_.rank == StorageRank::kGlobal) {
          const StmtNode* stmt_node = static_cast<const StmtNode*>(s.stmt);
          Stmt stmt = GetRef<Stmt>(stmt_node);
          // std::cout << "[OSYNC] Inserting loop carried sync before " << stmt << std::endl;
          std::cout << "[OSYNC] Inserting loop carried sync" << std::endl;
        }
        syncs_inserted_.insert(s.stmt);
      }
    }
    if (loop != nullptr) {
      for (size_t i = 0; i < seq.size(); ++i) {
        const StmtEntry& s = seq[i];
        if (syncs_inserted_.count(s.stmt) != 0) break;
        if (reads.empty() && writes.empty()) break;
        bool sync_before_stmt = false;
        for (const AccessEntry& acc : s.access) {
          AccessEntry updatedForSequentialLoop;

          if (loop->for_type == tvm::tir::ForType::Serial) {
            updatedForSequentialLoop.threads = acc.threads;
            updatedForSequentialLoop.buffer = acc.buffer;
            updatedForSequentialLoop.dtype = acc.dtype;
            updatedForSequentialLoop.type = acc.type;
            updatedForSequentialLoop.scope = acc.scope;
            updatedForSequentialLoop.double_buffer_write = acc.double_buffer_write;

            PrimExpr oldMin = acc.touched.min();
            PrimExpr oldMax = acc.touched.max();

            std::unordered_map<const VarNode*, PrimExpr> vsub;
            vsub[loop->loop_var.get()] = loop->loop_var + 1;
            VarReplacer varReplacer(vsub);
            if (oldMin.same_as(oldMax)) {
              PrimExpr newValue = varReplacer(oldMin);
              updatedForSequentialLoop.touched = arith::IntSet::interval(newValue, newValue);
            } else {
              updatedForSequentialLoop.touched =
                  arith::IntSet::interval(varReplacer(oldMin), varReplacer(oldMax));
            }
          } else {
            updatedForSequentialLoop = acc;
          }

          if (updatedForSequentialLoop.type == kRead) {
            if (FindConflict(writes, updatedForSequentialLoop, true)) {
              // std::cout << "[SYNC]   ALLO1 " << acc.buffer << std::endl;
              sync_before_stmt = true;
              break;
            }
          } else if (updatedForSequentialLoop.type == kWrite) {
            if (FindConflict(reads, updatedForSequentialLoop, true)) {
              // std::cout << "[SYNC]   ALLO2 " << acc.buffer << std::endl;
              sync_before_stmt = true;
              break;
            }
          } else if (updatedForSequentialLoop.type == kSync) {
            reads.clear();
            writes.clear();
          }
        }
        if (sync_before_stmt) {
          // CHECK_EQ(condition_counter(), 0)
          //     << "Cannot insert syncs inside condition. Want to insert sync before "
          //     << GetRef<Stmt>(static_cast<const StmtNode*>(s.stmt));
          const StmtNode* stmt_node = static_cast<const StmtNode*>(s.stmt);
          Stmt stmt = GetRef<Stmt>(stmt_node);
          const ForNode* scan = nullptr;
          if (scan_scope_.size() > 0) scan = scan_scope_.front();
          if (scan && sync_scope_.rank == StorageRank::kGlobal) {
            std::cout << "[HSYNC] Inserting loop carried sync in loop " << loop->loop_var
                      << " in scope " << scan << std::endl;
            syncs_inserted_.insert(scan->body.get());
          } else {
            if (sync_scope_.rank == StorageRank::kGlobal) {
              // std::cout << "[NSYNC] Inserting loop carried sync in loop " << loop->loop_var
              // << " in scope " << scan << " before " << stmt << std::endl;
              std::cout << "[NSYNC] Inserting loop carried sync in loop " << loop->loop_var
                        << " in scope " << scan << std::endl;
            }
            syncs_inserted_.insert(s.stmt);
          }
          break;
        }
      }
    }
    // return the exposed entries, remove unecessary ones.
    int sync_count = 0;
    // head are before first sync, tail are after last sync
    std::vector<AccessEntry> head, tail;
    AccessEntry esync;
    esync.threads = this->env_threads();
    esync.type = kSync;
    esync.scope = sync_scope_;

    for (const StmtEntry& s : seq) {
      if (syncs_inserted_.count(s.stmt)) {
        if (sync_count != 0) {
          tail.clear();
        } else {
          head.push_back(esync);
        }
        ++sync_count;
      }
      for (const AccessEntry& acc : s.access) {
        if (acc.type == kSync) {
          if (sync_count != 0) {
            tail.clear();
          } else {
            head.push_back(esync);
          }
          ++sync_count;
        } else {
          if (sync_count != 0) {
            tail.push_back(acc);
          } else {
            head.push_back(acc);
          }
        }
      }
    }
    head.insert(head.end(), tail.begin(), tail.end());
    if (loop != nullptr) {
      // clear double buffer flag after a loop is finished.
      for (AccessEntry& e : head) {
        e.double_buffer_write = false;
      }
    }
    return head;
  }

  // IntervalSet[((cpu_par_thread.x*32) + (i.ila.i.o*16)), (((cpu_par_thread.x*32) + (i.ila.i.o*16))
  // + 15)] IntervalSet[((cpu_par_thread.x*32) + (i.ila.i.o*16)), (((cpu_par_thread.x*32) +
  // (i.ila.i.o*16)) + 15)]

 private:
  // find conflicting entry in vec.
  bool FindConflict(const std::vector<AccessEntry>& vec, const AccessEntry& e, bool loop_carry) {
    arith::Analyzer analyzer;

    for (IterVar iv : env_threads()) {
      Range r = get_thread_extent(iv);
      if (r.defined()) {
        analyzer.Bind(iv->var, r);
      }
    }

    for (const AccessEntry& x : vec) {
      if (x.buffer.same_as(e.buffer)) {
        // std::cout << "[SYNC] Sync for " << x.buffer << " " << (e.type == kWrite && x.type ==
        // kRead)
        //           << " " << e.sync_type << " " << x.sync_type << std::endl;
        if (e.sync_type == kNoWar && x.sync_type == kNoWar && e.type == kWrite && x.type == kRead) {
          // std::cout << "[SYNC]   Skipping WAR for " << x.buffer << std::endl;
          continue;
        }
        arith::IntSet set1 = x.touched;
        arith::IntSet set2 = e.touched;

        // std::cout << "[SYNC]   Conflict " << x.buffer << " " << set1 << " " << set2 << std::endl;
        // Assumes no race between threads
        // Same index value means no conflicts
        // TODO(tqchen) more standard set based testing.
        if (e.touched.is_single_point() && x.touched.is_single_point()) {
          if (Equal(analyzer.Simplify(e.touched.point_value()),
                    analyzer.Simplify(x.touched.point_value()))) {
            // std::cout << "[SYNC]     Continue 0" << std::endl;
            continue;
          }
        }

        bool set1_lt_set2 = false;
        if (!set1.max().same_as(arith::pos_inf()) && !set2.min().same_as(arith::neg_inf())) {
          set1_lt_set2 = analyzer.CanProve(set1.max() < set2.min());
        }

        bool set2_lt_set1 = false;
        if (!set2.max().same_as(arith::pos_inf()) && !set1.min().same_as(arith::neg_inf())) {
          set2_lt_set1 = analyzer.CanProve(set2.max() < set1.min());
        }

        if (set1_lt_set2 || set2_lt_set1) {
          // std::cout << "[SYNC]     Continue 1" << std::endl;
          continue;
        }

        if (x.double_buffer_write && e.type == kRead && !loop_carry) {
          // std::cout << "[SYNC]     Continue 2" << std::endl;
          continue;
        }
        return true;
      }
    }
    return false;
  }

 private:
  // synchronization scope
  StorageScope sync_scope_;
};

class ThreadSyncInserter : public StmtExprMutator {
 public:
  ThreadSyncInserter(StorageScope sync_scope, const std::unordered_set<const Object*>& syncs,
                     std::string target)
      : sync_scope_(sync_scope), syncs_(syncs), target_(target) {}

  Stmt VisitStmt(const Stmt& stmt) final {
    if (syncs_.size() == 0) return stmt;
    if (syncs_.count(stmt.get())) {
      Stmt barrier;
      if (sync_scope_.rank == StorageRank::kGlobal) {
        barrier = MakeGlobalBarrier();
      } else {
        barrier = EvaluateNode::make(CallNode::make(DataType::Int(32), intrinsic::tvm_storage_sync,
                                                    {StringImmNode::make(sync_scope_.to_string())},
                                                    CallNode::Intrinsic));
      }
      // Mutate after query, to avoid stmt change.
      return SeqStmt({barrier, StmtExprMutator::VisitStmt(stmt)});
    } else {
      return StmtExprMutator::VisitStmt(stmt);
    }
    return stmt;
  }
  PrimExpr VisitExpr_(const LoadNode* op) final {
    if (sync_scope_.rank == StorageRank::kGlobal &&
        GetScope(op->buffer_var.get()).rank == StorageRank::kGlobal) {
      ++rw_stats_[op->buffer_var].read_count;
    }
    return StmtExprMutator::VisitExpr_(op);
  }
  Stmt VisitStmt_(const StoreNode* op) final {
    if (sync_scope_.rank == StorageRank::kGlobal &&
        GetScope(op->buffer_var.get()).rank == StorageRank::kGlobal) {
      ++rw_stats_[op->buffer_var].write_count;
    }
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      bool temp = true;
      std::swap(temp, in_thread_env_);
      thread_extents_.push_back(op);
      Stmt ret = StmtExprMutator::VisitStmt_(op);
      thread_extents_.pop_back();
      std::swap(temp, in_thread_env_);
      // first thread scope.

      // This visits the nested statements on line 247, and then here
      // initializes the global barrier if it is not yet in a thread
      // environment. But what if the nested statement needs a
      // barrier, and we are in a thread environment?

      if (!in_thread_env_ && sync_scope_.rank == StorageRank::kGlobal) {
        ret = InitGlobalBarrier(ret.as<AttrStmtNode>());
        num_blocks_ = PrimExpr();
        is_lead_ = PrimExpr();
      }
      return ret;
    } else if (op->attr_key == attr::storage_scope) {
      const VarNode* buf = op->node.as<VarNode>();
      storage_scope_[buf] = StorageScope::make(op->value.as<StringImmNode>()->value);
      return StmtExprMutator::VisitStmt_(op);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->is_intrinsic(intrinsic::tvm_access_ptr)) {
      PrimExpr expr = StmtExprMutator::VisitExpr_(op);
      op = expr.as<CallNode>();
      CHECK_EQ(op->args.size(), 5U);
      const VarNode* buffer_var = op->args[1].as<VarNode>();
      Var var(GetRef<Var>(buffer_var));
      const IntImmNode* flag = op->args[4].as<IntImmNode>();
      if ((flag->value & 1) && sync_scope_.rank == StorageRank::kGlobal &&
          GetScope(buffer_var).rank == StorageRank::kGlobal) {
        ++rw_stats_[var].read_count;
      }
      if (flag->value & 2 && sync_scope_.rank == StorageRank::kGlobal &&
          GetScope(buffer_var).rank == StorageRank::kGlobal) {
        ++rw_stats_[var].write_count;
      }
      return expr;
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

 private:
  // RW statistics about data
  struct Entry {
    int read_count{0};
    int write_count{0};
  };
  // Get current storage scope.
  StorageScope GetScope(const VarNode* buf) const {
    auto it = storage_scope_.find(buf);
    StorageScope s;
    s.rank = StorageRank::kGlobal;
    if (it == storage_scope_.end()) return s;
    return it->second;
  }
  // private functions.
  Stmt InitGlobalBarrier(const AttrStmtNode* op) {
    if (target_ == "cuda") {
      CHECK(op != nullptr);
      Array<PrimExpr> pargs = {StringImmNode::make(runtime::symbol::tvm_prepare_global_barrier)};
      Stmt prep = EvaluateNode::make(CallNode::make(DataType::Int(32), intrinsic::tvm_call_packed,
                                                    pargs, CallNode::Intrinsic));
      Stmt body = op->body;
      for (const auto& kv : rw_stats_) {
        const auto& e = kv.second;
        if (e.read_count != 0 && e.write_count != 0) {
          body = AttrStmtNode::make(kv.first, attr::volatile_scope, 1, body);
        }
      }
      rw_stats_.clear();
      Stmt kinit = EvaluateNode::make(CallNode::make(
          DataType::Int(32), intrinsic::tvm_global_barrier_kinit, {}, CallNode::Intrinsic));
      body = SeqStmt({kinit, body});
      body = AttrStmtNode::make(op->node, op->attr_key, op->value, body);
      return SeqStmt({prep, body});
    } else {
      return GetRef<Stmt>(op);
    }
  }
  Stmt MakeGlobalBarrier() {
    CHECK(sync_scope_.rank == StorageRank::kGlobal);
    if (target_ == "cuda") {
      if (!num_blocks_.defined()) {
        CHECK(!is_lead_.defined());
        num_work_dim_ = thread_extents_.size();
        for (const AttrStmtNode* attr : thread_extents_) {
          IterVar iv = Downcast<IterVar>(attr->node);
          runtime::ThreadScope s = runtime::ThreadScope::make(iv->thread_tag);
          if (s.rank == 0) {
            num_blocks_ = (num_blocks_.defined() ? attr->value * num_blocks_ : attr->value);
          } else if (s.rank == 1) {
            PrimExpr cond = iv->var == make_zero(iv->var.dtype());
            is_lead_ = is_lead_.defined() ? (is_lead_ && cond) : cond;
          }
        }
      } else {
        CHECK_EQ(num_work_dim_, thread_extents_.size());
      }
      return EvaluateNode::make(
          CallNode::make(DataType::Int(32), intrinsic::tvm_storage_sync,
                         {StringImmNode::make(sync_scope_.to_string()), is_lead_, num_blocks_},
                         CallNode::Intrinsic));
    } else {
      return AttrStmtNode::make(EvaluateNode::make(0), "pragma_parallel_barrier_when_finish", 0,
                                EvaluateNode::make(0));
    }
  }
  // data structure.
  StorageScope sync_scope_;
  const std::unordered_set<const Object*>& syncs_;
  // The storage scope of each buffer
  std::unordered_map<const VarNode*, StorageScope> storage_scope_;
  // The read write statistics of storage
  std::unordered_map<Var, Entry, ObjectHash, ObjectEqual> rw_stats_;
  // The statistics for global barrier
  bool in_thread_env_{false};
  // memorized results
  std::vector<const AttrStmtNode*> thread_extents_;
  size_t num_work_dim_{0};
  PrimExpr num_blocks_;
  PrimExpr is_lead_;
  std::string target_;
};

Stmt ThreadSync(Stmt stmt, std::string storage_scope, std::string target) {
  if (storage_scope != "global" && target == "llvm") return stmt;
  // std::cout << "[SYNC] for " << storage_scope << std::endl;
  StorageScope sync_scope = StorageScope::make(storage_scope);
  ThreadSyncPlanner planner(sync_scope);
  planner(stmt);
  Stmt stmt2 = ThreadSyncInserter(sync_scope, planner.syncs_inserted_, target)(std::move(stmt));
  // if (storage_scope == "global") std::cout << "[AFTER_SYNC] " << stmt2 << std::endl;
  return stmt2;
}

LoweredFunc ThreadSync(LoweredFunc f, std::string storage_scope, std::string target) {
  // std::cout << "[SYNC] for " << storage_scope << std::endl;
  CHECK_NE(f->func_type, kHostFunc);
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  n->body = ThreadSync(f->body, storage_scope, target);
  return LoweredFunc(n);
}

}  // namespace tir
}  // namespace tvm
