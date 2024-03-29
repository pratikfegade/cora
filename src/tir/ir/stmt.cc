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
 * \file tvm/tir/stmt.cc
 */
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt.h>

#include "../../arith/compute_expr.h"
#include "../pass/ir_util.h"

namespace tvm {
namespace tir {

Stmt LetStmtNode::make(Var var, PrimExpr value, Stmt body) {
  CHECK(value.defined());
  CHECK(body.defined());
  CHECK_EQ(value.dtype(), var.dtype());

  ObjectPtr<LetStmtNode> node = make_object<LetStmtNode>();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.LetStmt").set_body_typed(LetStmtNode::make);

Stmt AttrStmtNode::make(ObjectRef node, std::string attr_key, PrimExpr value, Stmt body,
                        int hfuse_group_id) {
  CHECK(hfuse_group_id < 0 || attr_key == attr::thread_extent);
  auto n = make_object<AttrStmtNode>();
  n->node = node;
  n->attr_key = std::move(attr_key);
  n->value = std::move(value);
  n->body = std::move(body);
  n->hfuse_group_id = std::move(hfuse_group_id);
  return Stmt(n);
}

TVM_REGISTER_GLOBAL("tir.AttrStmt").set_body_typed(AttrStmtNode::make);

Stmt AssertStmtNode::make(PrimExpr condition, PrimExpr message, Stmt body) {
  CHECK(condition.defined());
  CHECK(message.dtype() == DataType::Int(32) || message.as<StringImmNode>())
      << "TypeError: AssertStmt message must be an int or string:" << message << "\n";

  ObjectPtr<AssertStmtNode> node = make_object<AssertStmtNode>();
  node->condition = std::move(condition);
  node->message = std::move(message);
  node->body = std::move(body);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.AssertStmt").set_body_typed(AssertStmtNode::make);

Stmt ProducerConsumerNode::make(FunctionRef func, bool is_producer, Stmt body) {
  CHECK(body.defined());

  ObjectPtr<ProducerConsumerNode> node = make_object<ProducerConsumerNode>();
  node->func = std::move(func);
  node->is_producer = is_producer;
  node->body = std::move(body);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.ProducerConsumer").set_body_typed(ProducerConsumerNode::make);

Stmt ForNode::make(Var loop_var, PrimExpr min, PrimExpr extent, ForType for_type,
                   DeviceAPI device_api, Stmt body, int hfuse_group_id) {
  CHECK(min.defined());
  CHECK(extent.defined());
  CHECK(min.dtype().is_scalar());
  CHECK(extent.dtype().is_scalar());
  CHECK(loop_var.dtype().is_scalar());
  CHECK(body.defined());

  ObjectPtr<ForNode> node = make_object<ForNode>();
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->for_type = for_type;
  node->device_api = device_api;
  node->body = std::move(body);
  node->hfuse_group_id = std::move(hfuse_group_id);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.For").set_body_typed([](Var loop_var, PrimExpr min, PrimExpr extent,
                                                 int for_type, int device_api, Stmt body) {
  return ForNode::make(loop_var, min, extent, static_cast<ForType>(for_type),
                       static_cast<DeviceAPI>(device_api), body);
});

Stmt StoreNode::make(Var buffer_var, PrimExpr value, PrimExpr index, PrimExpr predicate,
                     SyncType sync_type) {
  CHECK(value.defined());
  CHECK(index.defined());
  CHECK(predicate.defined());
  CHECK_EQ(value.dtype().lanes(), index.dtype().lanes());
  CHECK_EQ(value.dtype().lanes(), predicate.dtype().lanes());

  ObjectPtr<StoreNode> node = make_object<StoreNode>();
  node->buffer_var = std::move(buffer_var);
  node->value = std::move(value);
  node->index = std::move(index);
  node->predicate = std::move(predicate);
  node->sync_type = std::move(sync_type);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.Store").set_body([](TVMArgs args, TVMRetValue* ret) {
  PrimExpr value = args[1];
  if (args.size() == 3) {
    *ret = StoreNode::make(args[0], value, args[2], const_true(value.dtype().lanes()), kAll);
  } else {
    *ret = StoreNode::make(args[0], value, args[2], args[3], kAll);
  }
});

Stmt ProvideNode::make(FunctionRef func, int value_index, PrimExpr value, Array<PrimExpr> args,
                       Array<Range> custom_realize_bounds) {
  CHECK(value_index >= 0 && value_index < func->num_outputs())
      << "value index output function return value bound";
  CHECK(value.defined()) << "Provide of undefined value\n";

  CHECK(custom_realize_bounds.size() == 0 || custom_realize_bounds.size() == args.size());

  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i].defined()) << "Provide to undefined location\n " << func;
  }

  ObjectPtr<ProvideNode> node = make_object<ProvideNode>();
  node->func = std::move(func);
  node->value_index = value_index;
  node->value = std::move(value);
  node->args = std::move(args);
  node->custom_realize_bounds = std::move(custom_realize_bounds);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.Provide").set_body_typed(ProvideNode::make);

Stmt AllocateNode::make(Var buffer_var, DataType dtype, Array<PrimExpr> extents, PrimExpr condition,
                        Stmt body, PrimExpr new_expr, std::string free_function) {
  return AllocateNode::make(buffer_var, dtype, extents, NullValue<Modes>(), condition, body,
                            new_expr, free_function);
}

Stmt AllocateNode::make(Var buffer_var, DataType dtype, Array<PrimExpr> extents, ObjectRef layout,
                        PrimExpr condition, Stmt body, PrimExpr new_expr,
                        std::string free_function) {
  // std::cout << "[ALLOC] Allocating " << buffer_var << " " << extents.size() << " " << layout
  // << std::endl;
  for (size_t i = 0; i < extents.size(); ++i) {
    CHECK(extents[i].defined());
    CHECK(extents[i].dtype().is_scalar());
  }
  CHECK(body.defined());
  CHECK(condition.defined());
  CHECK(condition.dtype().is_bool());

  CHECK(!layout.defined() || layout.as<ModesNode>());

  ObjectPtr<AllocateNode> node = make_object<AllocateNode>();
  node->buffer_var = std::move(buffer_var);
  node->dtype = dtype;
  node->extents = std::move(extents);
  node->layout = std::move(layout);
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->new_expr = std::move(new_expr);
  node->free_function = std::move(free_function);
  return Stmt(node);
}

// overloaded, needs special handling
// has default args
TVM_REGISTER_GLOBAL("tir.Allocate")
    .set_body_typed([](Var buffer_var, DataType type, Array<PrimExpr> extents, PrimExpr condition,
                       Stmt body) {
      return AllocateNode::make(buffer_var, type, extents, NullValue<Modes>(), condition, body);
    });

PrimExpr AllocateNode::variable_allocation_size() const {
  if (layout.defined()) {
    auto l = layout.as<ModesNode>();
    return l->GetAllocationSize();
  } else {
    return arith::ComputeReduce<MulNode>(extents, make_const(DataType::Int(32), 1));
  }
}

PrimExpr AllocateNode::GetAllocationSize() const {
  if (layout.defined()) {
    auto l = layout.as<ModesNode>();
    return l->GetAllocationSize();
  } else {
    return arith::ComputeReduce<MulNode>(extents, make_const(DataType::Int(32), 1));
  }
}

int32_t AllocateNode::constant_allocation_size(const Array<PrimExpr>& extents,
                                               const ObjectRef& layout) {
  if (layout.defined() && layout.as<ModesNode>() && layout.as<ModesNode>()->is_ragged()) return 0;

  int64_t result = 1;
  for (size_t i = 0; i < extents.size(); ++i) {
    if (const IntImmNode* int_size = extents[i].as<IntImmNode>()) {
      result *= int_size->value;
      if (result > std::numeric_limits<int32_t>::max()) {
        return 0;
      }
    } else {
      return 0;
    }
  }
  return static_cast<int32_t>(result);
}

Stmt FreeNode::make(Var buffer_var) {
  ObjectPtr<FreeNode> node = make_object<FreeNode>();
  node->buffer_var = buffer_var;
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.Free").set_body_typed(FreeNode::make);

Stmt RealizeNode::make(FunctionRef func, int value_index, DataType dtype, Region bounds,
                       PrimExpr condition, Stmt body, ObjectRef layout) {
  for (size_t i = 0; i < bounds.size(); ++i) {
    CHECK(bounds[i]->min.defined());
    CHECK(bounds[i]->extent.defined());
    CHECK(bounds[i]->min.dtype().is_scalar());
    CHECK(bounds[i]->extent.dtype().is_scalar());
  }
  CHECK(body.defined());
  CHECK(condition.defined());
  CHECK(condition.dtype().is_bool());

  ObjectPtr<RealizeNode> node = make_object<RealizeNode>();
  node->func = std::move(func);
  node->value_index = value_index;
  node->dtype = dtype;
  node->bounds = std::move(bounds);
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->layout = std::move(layout);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.Realize").set_body_typed(RealizeNode::make);

Stmt PrefetchNode::make(FunctionRef func, int value_index, DataType dtype, Region bounds) {
  for (size_t i = 0; i < bounds.size(); ++i) {
    CHECK(bounds[i]->min.defined());
    CHECK(bounds[i]->extent.defined());
    CHECK(bounds[i]->min.dtype().is_scalar());
    CHECK(bounds[i]->extent.dtype().is_scalar());
  }

  CHECK(bounds.size());

  ObjectPtr<PrefetchNode> node = make_object<PrefetchNode>();
  node->func = std::move(func);
  node->value_index = value_index;
  node->dtype = dtype;
  node->bounds = std::move(bounds);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.Prefetch").set_body_typed(PrefetchNode::make);

SeqStmt::SeqStmt(Array<Stmt> seq) {
  auto node = make_object<SeqStmtNode>();
  node->seq = std::move(seq);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.SeqStmt").set_body_typed([](Array<Stmt> seq) {
  return SeqStmt(std::move(seq));
});

Stmt IfThenElseNode::make(PrimExpr condition, Stmt then_case, Stmt else_case) {
  CHECK(condition.defined());
  CHECK(then_case.defined());
  // else_case may be null.

  ObjectPtr<IfThenElseNode> node = make_object<IfThenElseNode>();
  node->condition = std::move(condition);
  node->then_case = std::move(then_case);
  node->else_case = std::move(else_case);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.IfThenElse").set_body_typed(IfThenElseNode::make);

Stmt EvaluateNode::make(PrimExpr value) {
  CHECK(value.defined());

  ObjectPtr<EvaluateNode> node = make_object<EvaluateNode>();
  node->value = std::move(value);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.Evaluate").set_body_typed(EvaluateNode::make);

// Printers

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LetStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const LetStmtNode*>(node.get());
      p->PrintIndent();
      p->stream << "let " << op->var << " = ";
      p->Print(op->value);
      p->stream << '\n';
      p->Print(op->body);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AttrStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AttrStmtNode*>(node.get());
      p->PrintIndent();
      p->stream << "// attr [";
      p->Print(op->node);
      p->stream << "] " << op->attr_key << " = ";
      p->Print(op->value);
      if (op->attr_key == attr::thread_extent) p->stream << " (" << op->hfuse_group_id << ")";
      p->stream << '\n';
      p->Print(op->body);
      if (op->attr_key == attr::prep_code_scope) p->stream << "PrepCodeEnds\n";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AssertStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AssertStmtNode*>(node.get());
      p->PrintIndent();
      p->stream << "assert(";
      p->Print(op->condition);
      p->stream << ", ";
      p->Print(op->message);
      p->stream << ")\n";
      p->Print(op->body);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ProducerConsumerNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ProducerConsumerNode*>(node.get());
      if (op->is_producer) {
        p->PrintIndent();
        p->stream << "produce " << op->func->func_name() << " {\n";
        p->indent += 2;
        p->Print(op->body);
        p->indent -= 2;
        p->PrintIndent();
        p->stream << "}\n";
      } else {
        // p->PrintIndent();
        // p->stream << "consume " << op->func->func_name() << " {\n";
        // p->indent += 2;
        // p->Print(op->body);
        // p->indent -= 2;
        // p->PrintIndent();
        // p->stream << "}\n";
        p->Print(op->body);
      }
    });

std::ostream& operator<<(std::ostream& out, ForType type) {  // NOLINT(*)
  switch (type) {
    case ForType::Serial:
      out << "for";
      break;
    case ForType::Parallel:
      out << "parallel";
      break;
    case ForType::Unrolled:
      out << "unrolled";
      break;
    case ForType::Vectorized:
      out << "vectorized";
      break;
    case ForType::Peeled:
      out << "peeled";
      break;
  }
  return out;
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ForNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ForNode*>(node.get());
      p->PrintIndent();
      p->stream << op->for_type << " (" << op->loop_var << ", ";
      p->Print(op->min);
      p->stream << ", ";
      p->Print(op->extent);
      p->stream << ") {\n";

      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;

      p->PrintIndent();
      p->stream << "}\n";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StoreNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const StoreNode*>(node.get());
      p->PrintIndent();
      p->stream << op->buffer_var << "[";
      p->Print(op->index);
      p->stream << "] = ";
      p->Print(op->value);
      if (!is_one(op->predicate)) {
        p->stream << " if ";
        p->Print(op->predicate);
      }
      p->stream << '\n';
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ProvideNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ProvideNode*>(node.get());
      p->PrintIndent();
      p->stream << op->func->func_name() << "(";
      for (size_t i = 0; i < op->args.size(); ++i) {
        p->Print(op->args[i]);
        if (i < op->args.size() - 1) p->stream << ", ";
      }
      p->stream << ")";
      if (op->func->num_outputs() != 1) {
        p->stream << ".value[" << op->value_index << "]";
      }
      p->stream << " =";
      p->Print(op->value);
      p->stream << '\n';
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AllocateNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AllocateNode*>(node.get());
      p->PrintIndent();
      p->stream << "allocate " << op->buffer_var << "[" << op->dtype;
      for (size_t i = 0; i < op->extents.size(); ++i) {
        p->stream << " * ";
        p->Print(op->extents[i]);
      }
      p->stream << "]";
      if (!is_one(op->condition)) {
        p->stream << " if ";
        p->Print(op->condition);
      }
      p->stream << "\n";
      p->Print(op->body);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FreeNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const FreeNode*>(node.get());
      p->PrintIndent();
      p->stream << "free " << op->buffer_var;
      p->stream << '\n';
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RealizeNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const RealizeNode*>(node.get());
      p->PrintIndent();
      p->stream << "realize " << op->func->func_name() << "(";
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        p->stream << "[";
        p->Print(op->bounds[i]->min);
        p->stream << ", ";
        p->Print(op->bounds[i]->extent);
        p->stream << "]";
        if (i < op->bounds.size() - 1) p->stream << ", ";
      }
      p->stream << ")";
      if (op->func->num_outputs() != 1) {
        p->stream << ".value[" << op->value_index << "]";
      }
      if (!is_one(op->condition)) {
        p->stream << " if ";
        p->Print(op->condition);
      }
      p->stream << " {\n";

      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;

      p->PrintIndent();
      p->stream << "}\n";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrefetchNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrefetchNode*>(node.get());
      p->PrintIndent();
      p->stream << "prefetch " << op->func->func_name() << "(";
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        p->stream << "[";
        p->Print(op->bounds[i]->min);
        p->stream << ", ";
        p->Print(op->bounds[i]->extent);
        p->stream << "]";
        if (i < op->bounds.size() - 1) p->stream << ", ";
      }
      p->stream << ")";
      if (op->func->num_outputs() != 1) {
        p->stream << ".value[" << op->value_index << "]";
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SeqStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SeqStmtNode*>(node.get());
      for (Stmt stmt : op->seq) {
        p->Print(stmt);
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IfThenElseNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IfThenElseNode*>(node.get());
      p->PrintIndent();
      while (true) {
        p->stream << "if (" << op->condition << ") {\n";
        p->indent += 2;
        p->Print(op->then_case);
        p->indent -= 2;

        if (!op->else_case.defined()) {
          break;
        }

        if (const IfThenElseNode* nested_if = op->else_case.as<IfThenElseNode>()) {
          p->PrintIndent();
          p->stream << "} else ";
          op = nested_if;
        } else {
          p->PrintIndent();
          p->stream << "} else {\n";
          p->indent += 2;
          p->Print(op->else_case);
          p->indent -= 2;
          break;
        }
      }
      p->PrintIndent();
      p->stream << "}\n";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<EvaluateNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const EvaluateNode*>(node.get());
      p->PrintIndent();
      p->Print(op->value);
      p->stream << "\n";
    });

template <typename T>
void PrintList(const Array<T>& exprs, ReprPrinter* p) {
  for (size_t i = 0; i < exprs.size(); ++i) {
    p->Print(exprs[i]);
    if (i < exprs.size() - 1) {
      p->stream << ", ";
    }
  }
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShuffleNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ShuffleNode*>(node.get());
      p->stream << "shuffle(";
      PrintList(op->vectors, p);
      p->stream << ", ";
      PrintList(op->indices, p);
      p->stream << ")";
    });

TVM_REGISTER_NODE_TYPE(AttrStmtNode);
TVM_REGISTER_NODE_TYPE(PrefetchNode);
TVM_REGISTER_NODE_TYPE(CallNode);
TVM_REGISTER_NODE_TYPE(LetNode);
TVM_REGISTER_NODE_TYPE(LetStmtNode);
TVM_REGISTER_NODE_TYPE(AssertStmtNode);
TVM_REGISTER_NODE_TYPE(ProducerConsumerNode);
TVM_REGISTER_NODE_TYPE(ForNode);
TVM_REGISTER_NODE_TYPE(StoreNode);
TVM_REGISTER_NODE_TYPE(ProvideNode);
TVM_REGISTER_NODE_TYPE(AllocateNode);
TVM_REGISTER_NODE_TYPE(FreeNode);
TVM_REGISTER_NODE_TYPE(RealizeNode);
TVM_REGISTER_NODE_TYPE(SeqStmtNode);
TVM_REGISTER_NODE_TYPE(IfThenElseNode);
TVM_REGISTER_NODE_TYPE(EvaluateNode);

}  // namespace tir
}  // namespace tvm
