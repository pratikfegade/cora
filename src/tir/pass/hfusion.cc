#include <tvm/arith/analyzer.h>
#include <tvm/arith/z3_analyzer.h>
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_equality.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/lowered_func.h>
#include <tvm/tir/stmt_functor.h>

#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "../ir/var_replacer.h"

#define COUT std::cout << "[RIfR] "
namespace tvm {
namespace tir {
class HFuser : public StmtMutator {
  void FuseGroupCommon(const Array<Stmt>& bodies, const Array<Var>& vars,
                       const Array<PrimExpr>& extents, const Var& common_var,
                       Array<Stmt>* p_new_bodies, Array<PrimExpr>* p_cumulative_extents) {
    Array<Stmt>& new_bodies = *p_new_bodies;
    Array<PrimExpr>& cumulative_extents = *p_cumulative_extents;

    PrimExpr cumulative_extent = 0;
    cumulative_extents.push_back(cumulative_extent);
    for (size_t i = 0; i < bodies.size(); ++i) {
      std::unordered_map<const VarNode*, PrimExpr> vsub;
      vsub[vars[i].operator->()] = common_var - cumulative_extent;
      VarReplacer replacer(vsub);
      // std::cout << "[HFUSE] Replacing " << vars[i] << " in " << bodies[i] << std::endl;
      Stmt new_body = replacer(bodies[i]);
      new_bodies.push_back(new_body);
      cumulative_extent = cumulative_extent + extents[i];
      cumulative_extents.push_back(cumulative_extent);
      // std::cout << "[HFUSE]   Cum Extent " << cumulative_extent << std::endl;
    }
  }

  Stmt FuseGroupAttrStmt(std::vector<const AttrStmtNode*> group) {
    Array<Stmt> bodies;
    Array<Var> vars;
    Array<PrimExpr> extents;
    IterVar common_iv = Downcast<IterVar>(group[0]->node);
    for (size_t i = 0; i < group.size(); ++i) {
      auto attr = group[i];
      bodies.push_back(attr->body);
      vars.push_back(Downcast<IterVar>(attr->node)->var);
      extents.push_back(attr->value);
    }
    Array<Stmt> new_bodies;
    Array<PrimExpr> cumulative_extents;
    FuseGroupCommon(bodies, vars, extents, common_iv->var, &new_bodies, &cumulative_extents);

    Stmt new_stmt = new_bodies[group.size() - 1];
    for (int i = group.size() - 2; i >= 0; --i) {
      new_stmt = IfThenElseNode::make(
          common_iv->var >= cumulative_extents[i] && common_iv->var < cumulative_extents[i + 1],
          new_bodies[i], new_stmt);
    }

    CHECK(!fused_attr_iv.defined());
    fused_attr_iv = common_iv;
    fused_iv_extent = cumulative_extents[group.size()];

    // std::cout << "[FUSE]  Set fused_attr " << fused_attr_iv << std::endl;
    return new_stmt;
  }

  Stmt FuseGroupFor(std::vector<const ForNode*> group) {
    Array<Stmt> bodies;
    Array<Var> vars;
    Array<PrimExpr> extents;
    Var new_loop_var = Var("fused", group[0]->loop_var->dtype);
    for (size_t i = 0; i < group.size(); ++i) {
      auto loop = group[i];
      bodies.push_back(loop->body);
      vars.push_back(loop->loop_var);
      CHECK(is_zero(loop->min));
      extents.push_back(loop->extent);
    }
    Array<Stmt> new_bodies;
    Array<PrimExpr> cumulative_extents;
    FuseGroupCommon(bodies, vars, extents, new_loop_var, &new_bodies, &cumulative_extents);

    Stmt new_stmt = EvaluateNode::make(0);

    for (int i = group.size() - 1; i >= 0; --i) {
      new_stmt = IfThenElseNode::make(
          new_loop_var >= cumulative_extents[i] && new_loop_var < cumulative_extents[i + 1],
          new_bodies[i], new_stmt);
    }
    return ForNode::make(new_loop_var, 0, cumulative_extents[group.size()], group[0]->for_type,
                         group[0]->device_api, new_stmt, -1);
  }

  Stmt FuseGroup(Array<Stmt> group) {
    // std::cout << "[FUSE] Group " << group.size() << std::endl;
    if (auto fn = group[0].as<ForNode>()) {
      std::vector<const ForNode*> for_group;
      auto last_type = fn->for_type;
      auto last_device_api = fn->device_api;
      for (auto stmt : group) {
        auto for_node = stmt.as<ForNode>();
        CHECK(for_node) << "Mixed groups not allowed";
        CHECK(last_type == for_node->for_type);
        CHECK(last_device_api == for_node->device_api);
        for_group.push_back(for_node);
      }
      return FuseGroupFor(for_group);
    } else {
      std::vector<const AttrStmtNode*> attr_group;
      ObjectRef last_iv = NullValue<ObjectRef>();
      for (auto stmt : group) {
        auto attr_node = stmt.as<AttrStmtNode>();
        CHECK(attr_node) << "Mixed groups not allowed";
        CHECK(attr_node->attr_key == attr::thread_extent);
        CHECK(!last_iv.defined() || last_iv == attr_node->node);
        attr_group.push_back(attr_node);
      }
      return FuseGroupAttrStmt(attr_group);
    }
    CHECK(false);
    return SeqStmt(group);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    Array<Stmt> current_group;
    int current_group_id = -1;

    Array<Stmt> new_seq;
    auto handle_stmt = [&](int stmt_group_id, Stmt stmt) {
      if (stmt_group_id >= 0) {
        if (stmt_group_id == current_group_id) {
          // Continue the current group
        } else {
          // We start a new group here
          if (current_group_id < 0) {
            // There was no previous group
            current_group_id = stmt_group_id;
            CHECK_EQ(current_group.size(), 0);
          } else {
            // There was a group we now need to end
            CHECK(current_group.size() > 0) << "Single stmt hfuse group";
            new_seq.push_back(FuseGroup(current_group));
            current_group.resize(0);
            current_group_id = stmt_group_id;
          }
        }
        current_group.push_back(StmtMutator::VisitStmt(stmt));
      } else if (current_group_id >= 0) {
        // We end a group here
        CHECK(current_group.size() > 0) << "Single stmt hfuse group";
        new_seq.push_back(FuseGroup(current_group));
        current_group.resize(0);
        current_group_id = -1;
      } else if (current_group_id >= 0) {
        new_seq.push_back(StmtMutator::VisitStmt(stmt));
      }
    };

    // std::cout << "[FUSE] Seq " << std::endl;
    for (auto stmt : op->seq) {
      if (auto fornode = stmt.as<ForNode>()) {
        // std::cout << "[FUSE]   For " << fornode->loop_var << std::endl;
        if (fornode->hfuse_group_id >= 0) {
          handle_stmt(fornode->hfuse_group_id, stmt);
        } else {
          new_seq.push_back(StmtMutator::VisitStmt(stmt));
        }
      } else if (auto attrnode = stmt.as<AttrStmtNode>()) {
        // std::cout << "[FUSE]   Attr " << attrnode->node << std::endl;
        if (attrnode->hfuse_group_id >= 0) {
          handle_stmt(attrnode->hfuse_group_id, stmt);
        } else {
          new_seq.push_back(StmtMutator::VisitStmt(stmt));
        }
      } else {
        new_seq.push_back(StmtMutator::VisitStmt(stmt));
      }
    }

    // Clean up last group
    if (current_group.size() > 0) {
      new_seq.push_back(FuseGroup(current_group));
    }

    return SeqStmt(new_seq);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::hfuse_group) {
      CHECK(!fused_attr_iv.defined());
      // std::cout << "[FUSE] Visiting hfuse body\n" << op->body << std::endl;
      Stmt body = this->VisitStmt(op->body);

      std::unordered_set<const VarNode*> to_remove;
      for (auto it : scope_map) {
        auto var = it.first;
        auto scope = it.second.as<StringImmNode>()->value;
        if (scope == "local" || scope == "shared") {
          auto alloc = alloc_map[var];

          // std::cout << "[FUSE]  Pushing Alloc " << var->name_hint << " " << scope << std::endl;
          body = AttrStmtNode::make(
              alloc->buffer_var, attr::storage_scope, it.second,
              AllocateNode::make(alloc->buffer_var, alloc->dtype, alloc->extents, alloc->layout,
                                 alloc->condition, body, alloc->new_expr, alloc->free_function),
              -1);
          to_remove.insert(var);
        }
      }

      for (auto var : to_remove) {
        // std::cout << "[FUSE]   Removing " << var->name_hint << " " << var << std::endl;
        alloc_map.erase(var);
        scope_map.erase(var);
      }

      // for (auto it : alloc_map) {
      // std::cout << "[FUSE]   Present " << it.first->name_hint << " " << it.first << std::endl;
      // }

      if (fused_attr_iv.defined()) {
        auto ret =
            AttrStmtNode::make(fused_attr_iv, attr::thread_extent, fused_iv_extent, body, -1);
        fused_attr_iv = NullValue<IterVar>();
        fused_iv_extent = NullValue<PrimExpr>();
        // std::cout << "[FUSE]  Returning\n" << ret << std::endl;
        return ret;
      }
    } else if (op->attr_key == attr::storage_scope) {
      const VarNode* buf = op->node.as<VarNode>();
      scope_map[buf] = op->value;
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    // std::cout << "[FUSE] Alloc " << op->buffer_var << std::endl;
    alloc_map[op->buffer_var.operator->()] = op;
    Stmt body = this->VisitStmt(op->body);

    // std::cout << "[FUSE] Back to alok " << op->buffer_var << std::endl;
    // for (auto it : alloc_map) {
    // std::cout << "[FUSE]   Present " << it.first->name_hint << " " << it.first << std::endl;
    // }
    if (alloc_map.count(op->buffer_var.operator->())) {
      alloc_map.erase(op->buffer_var.operator->());
      scope_map.erase(op->buffer_var.operator->());
      return AllocateNode::make(op->buffer_var, op->dtype, op->extents, op->layout, op->condition,
                                body, op->new_expr, op->free_function);
    } else {
      // std::cout << "[FUSE]   Skipping " << op->buffer_var << std::endl;
      return body;
    }
  }

  std::unordered_map<const VarNode*, PrimExpr> scope_map;
  std::unordered_map<const VarNode*, const AllocateNode*> alloc_map;

  IterVar fused_attr_iv = NullValue<IterVar>();
  PrimExpr fused_iv_extent = NullValue<PrimExpr>();
};

LoweredFunc HorizontalFuse(LoweredFunc f) {
  // std::cout << "[HFUSE] Fusing" << std::endl;
  HFuser fuser;
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  Stmt body = f->body;
  n->body = fuser(body);
  return LoweredFunc(n);
}

}  // namespace tir
}  // namespace tvm
#undef COUT
