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
 * \file make_api.cc Build API function.
 */
#include <tvm/runtime/device_api.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_set>
#include <utility>
#include <vector>

#include "../ir/var_replacer.h"
#include "arg_binder.h"
#include "ir_util.h"

namespace tvm {
namespace tir {

MakeAPIResult MakeAPIResultNode::make(LoweredFunc function, Array<Buffer> host_intermediate_buffers,
					Array<Buffer> device_intermediate_buffers) {
  auto n = make_object<MakeAPIResultNode>();
  n->function = std::move(function);
  n->host_intermediate_buffers = std::move(host_intermediate_buffers);
  n->device_intermediate_buffers = std::move(device_intermediate_buffers);
  return MakeAPIResult(n);
}

TVM_REGISTER_NODE_TYPE(MakeAPIResultNode);

inline Stmt MakeAssertEQ(PrimExpr lhs, PrimExpr rhs, std::string msg) {
  return AssertStmtNode::make(lhs == rhs, msg, EvaluateNode::make(0));
}

LoweredFunc MakeAPIInternal(Stmt body, std::string name, Array<ObjectRef> api_args,
                            int num_unpacked_args, bool is_restricted,
                            std::unordered_set<const Object*> cpu_args,
                            std::unordered_map<const VarNode*, PrimExpr>* p_vmap,
                            ArgBinder* p_binder, Var* p_device_type, Var* p_device_id) {
  const Stmt nop = EvaluateNode::make(0);
  int num_args = static_cast<int>(api_args.size());
  CHECK_LE(num_unpacked_args, num_args);
  int num_packed_args = num_args - num_unpacked_args;
  // Data field definitions
  // The packed fields
  Var v_packed_args("args", DataType::Handle());
  Var v_packed_arg_type_ids("arg_type_ids", DataType::Handle());
  Var v_num_packed_args("num_args", DataType::Int(32));
  Var v_out_ret_value("out_ret_value", DataType::Handle());
  Var v_out_ret_tcode("out_ret_tcode", DataType::Handle());
  // The arguments of the function.
  Array<Var> args;
  // The device context
  Var& device_type = *p_device_type;
  Var& device_id = *p_device_id;
  // seq_init gives sequence of initialization
  // seq_check gives sequence of later checks after init
  std::vector<Stmt> seq_init, seq_check;
  std::unordered_map<const VarNode*, PrimExpr>& vmap = *p_vmap;
  ArgBinder& binder = *p_binder;
  // ---------------------------
  // local function definitions
  // load i-th argument as type t
  auto f_arg_value = [&](DataType t, int i) {
    Array<PrimExpr> call_args{v_packed_args, IntImm(DataType::Int(32), i),
                              IntImm(DataType::Int(32), intrinsic::kTVMValueContent)};
    // load 64 bit version
    DataType api_type = APIType(t);
    PrimExpr res =
        CallNode::make(api_type, intrinsic::tvm_struct_get, call_args, CallNode::PureIntrinsic);
    // cast to the target version.
    if (api_type != t) {
      res = CastNode::make(t, res);
    }
    return res;
  };
  // get declaration of argument i
  auto f_arg_decl = [&](int i) {
    std::ostringstream os;
    os << "arg" << i;
    const VarNode* v = api_args[i].as<VarNode>();
    Var ret = Var(os.str(), v ? v->dtype : DataType::Handle());
    return ret;
  };
  // ---------------------------
  // start of logics
  // add signature for packed arguments.
  if (num_packed_args != 0) {
    args.push_back(v_packed_args);
    args.push_back(v_packed_arg_type_ids);
    args.push_back(v_num_packed_args);
    std::ostringstream os;

    os << name << ": num_args should be " << num_packed_args;
    seq_init.emplace_back(MakeAssertEQ(v_num_packed_args, num_packed_args, os.str()));
  }

  // Save the input variables and buffers that will be bound later.
  std::vector<std::pair<Var, Var> > var_defs;
  std::vector<std::pair<Buffer, Var> > buf_defs;
  for (int i = 0; i < static_cast<int>(api_args.size()); ++i) {
    Var v_arg = f_arg_decl(i);
    if (i < num_packed_args) {
      // Value loads
      seq_init.emplace_back(LetStmtNode::make(v_arg, f_arg_value(v_arg.dtype(), i), nop));
      // type code checks
      Var tcode(v_arg->name_hint + ".code", DataType::Int(32));
      seq_init.emplace_back(
          LetStmtNode::make(tcode,
                            LoadNode::make(DataType::Int(32), v_packed_arg_type_ids,
                                           IntImm(DataType::Int(32), i), const_true(1), kAll),
                            nop));
      DataType t = v_arg.dtype();
      if (t.is_handle()) {
        std::ostringstream msg;
        msg << name << ": Expect arg[" << i << "] to be pointer";
        seq_check.emplace_back(
            AssertStmtNode::make(tcode == kTVMOpaqueHandle || tcode == kTVMNDArrayHandle ||
                                     tcode == kTVMDLTensorHandle || tcode == kTVMNullptr,
                                 msg.str(), nop));
      } else if (t.is_int() || t.is_uint()) {
        std::ostringstream msg;
        msg << name << ": Expect arg[" << i << "] to be int";
        seq_check.emplace_back(AssertStmtNode::make(tcode == kDLInt, msg.str(), nop));
      } else {
        CHECK(t.is_float());
        std::ostringstream msg;
        msg << name << ": Expect arg[" << i << "] to be float";
        seq_check.emplace_back(AssertStmtNode::make(tcode == kDLFloat, msg.str(), nop));
      }
    } else {
      args.push_back(v_arg);
    }
    // add checks for functions.
    if (api_args[i].as<VarNode>()) {
      var_defs.emplace_back(std::make_pair(Downcast<Var>(api_args[i]), v_arg));
    } else {
      // Buffer checks
      CHECK(api_args[i].as<BufferNode>()) << "api_args can only be Buffer or Var";
      buf_defs.emplace_back(std::make_pair(Downcast<Buffer>(api_args[i]), v_arg));
    }
  }

  // allow return value if the function is packed.
  if (num_packed_args != 0) {
    args.push_back(v_out_ret_value);
    args.push_back(v_out_ret_tcode);
  }

  size_t expected_nargs = num_unpacked_args + (num_packed_args != 0 ? 5 : 0);
  CHECK_EQ(args.size(), expected_nargs);

  // Arg definitions are defined before buffer binding to avoid the use before
  // def errors.
  //
  // For example, for auto broadcasting, checks are required to guarantee that
  // either 0 or the original stride will be correctly used. Checks here have
  // to use the args that may have no let bining yet. Therefore, hoisting let
  // binding for args before buffer declaration is needed.
  for (const auto& arg : var_defs) {
    binder.Bind(arg.first, arg.second, arg.second->name_hint, true);
  }

  for (const auto& buf_arg : buf_defs) {
    if (cpu_args.count(buf_arg.first.get())) {
      binder.BindDLTensor(buf_arg.first, kDLCPU, 0, buf_arg.second, buf_arg.second->name_hint);
    } else {
      binder.BindDLTensor(buf_arg.first, device_type, device_id, buf_arg.second,
                          buf_arg.second->name_hint);
    }
  }

  ObjectPtr<LoweredFuncNode> n = make_object<LoweredFuncNode>();
  n->name = name;
  n->args = args;
  n->handle_data_type = binder.def_handle_dtype();
  n->is_packed_func = num_unpacked_args == 0;
  n->is_restricted = is_restricted;
  body = AttrStmtNode::make(make_zero(DataType::Int(32)), attr::compute_scope,
                            StringImmNode::make(name + "_compute_"), body);
  // Set device context
  if (vmap.count(device_id.get())) {
    PrimExpr node = StringImmNode::make("default");
    CHECK(vmap.count(device_type.get()));
    seq_check.push_back(AttrStmtNode::make(node, attr::device_context_id, device_id, nop));
    seq_check.push_back(AttrStmtNode::make(node, attr::device_context_type, device_type, nop));
    Stmt set_device = IfThenElseNode::make(
        device_type != kDLCPU,
        EvaluateNode::make(CallNode::make(
            DataType::Int(32), intrinsic::tvm_call_packed,
            {StringImmNode::make(runtime::symbol::tvm_set_device), device_type, device_id},
            CallNode::Intrinsic)));
    body = SeqStmt({set_device, body});
  }

  n->body = MergeNest({seq_init, binder.init_nest(), seq_check, binder.asserts()}, body);
  LoweredFunc f(n);
  Array<Var> undefined = UndefinedVars(f->body, f->args);
  if (undefined.size() != 0) {
    std::ostringstream os;
    for (Var v : undefined) {
      os << " \'" << v->name_hint << "\' ";
    }
    os << " does not appear in api_args";
    LOG(FATAL) << "Not all Vars are passed in api_args: " << os.str() << "\n" << f->body;
  }
  return f;
}

class CopyStatementsRewriter : public StmtExprMutator {
 public:
  CopyStatementsRewriter(const Var& device_type, const Var& device_id)
      : device_type_(device_type), device_id_(device_id) {}

  PrimExpr VisitExpr_(const CallNode* op) override {
    if (op->call_type == CallNode::CallType::Intrinsic &&
        op->name == intrinsic::tvm_memcopy_to_device) {
      Array<PrimExpr> args;
      args.push_back(op->args[0]);
      args.push_back(op->args[1]);
      args.push_back(op->args[2]);
      args.push_back(op->args[3]);
      args.push_back(op->args[4]);

      args.push_back(kDLCPU);
      args.push_back(0);

      args.push_back(device_type_);
      args.push_back(device_id_);
      args.push_back(op->args[9]);
      args.push_back(op->args[10]);

      return CallNode::make(op->dtype, op->name, args, op->call_type, {}, op->func, 0, {});
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  const Var& device_type_;
  const Var& device_id_;
};

MakeAPIResult MakeAPI(Stmt body, std::string name, Array<ObjectRef> lengths_api_args,
                      Array<ObjectRef> tensor_api_args, int num_unpacked_args, bool is_restricted,
                      bool handle_prep_code) {
  Var device_type("dev_type"), device_id("dev_id");
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  ArgBinder binder(&vmap);
  if (!handle_prep_code) {
    Array<ObjectRef> full_api_args;
    full_api_args.push_back_all(tensor_api_args);
    full_api_args.push_back_all(lengths_api_args);
    auto func = MakeAPIInternal(body, name, full_api_args, num_unpacked_args, is_restricted, {},
                                &vmap, &binder, &device_id, &device_type);
    return MakeAPIResultNode::make(func, {}, {});
  } else {
    Stmt prep_code;
    Stmt main_body;
    Map<Buffer, Buffer> prep_buffer_map = ExtractPrepCode(body, &prep_code, &main_body);
    Array<Buffer> host_intermediate_api_args;
    Array<Buffer> device_intermediate_api_args;
    for (auto it : prep_buffer_map) {
      host_intermediate_api_args.push_back(it.first);
      device_intermediate_api_args.push_back(it.second);
    }

    // Add copy statements for length api args that are also used in
    // main body
    {
      auto prep_attr = prep_code.as<AttrStmtNode>();
      CHECK(prep_attr);
      auto body_vars = VarCollector(true).collect(main_body);
      Map<Buffer, Buffer> to_copy_l_buffer_map;
      std::unordered_map<const VarNode*, PrimExpr> vsub;
      for (auto arg : lengths_api_args) {
        // std::cout << "[M_API] Length Arg " << arg << std::endl;
        if (auto buf_node = arg.as<BufferNode>()) {
          if (body_vars.count(buf_node->data.get())) {
            // std::cout << "[M_API]  Used in body" << std::endl;
            auto host_buf = Downcast<Buffer>(arg);
            auto dev_buf = BufferNode::make(host_buf->data.copy_with_suffix("_d"), host_buf->dtype,
                                            host_buf->shape, host_buf->strides,
                                            host_buf->elem_offset, host_buf->name + "_d", host_buf->scope,
                                            host_buf->data_alignment, host_buf->offset_factor,
                                            host_buf->buffer_type, host_buf->sync_type);
            to_copy_l_buffer_map.Set(host_buf, dev_buf);
	    prep_buffer_map.Set(host_buf, dev_buf);
            vsub[host_buf->data.operator->()] = dev_buf->data;
          }
        }
      }

      // Add copy statements at the end of the prep_code
      Array<Stmt> l_copy_stmts;
      l_copy_stmts.push_back(prep_attr->body);
      for (auto it : to_copy_l_buffer_map) {
        PrimExpr extent = 1;
        for (auto dim_length : it.first->shape->get_dense_shape()) {
          extent = extent * dim_length;
        }
        auto dtype = it.first->dtype;
        l_copy_stmts.push_back(EvaluateNode::make(
            copy_to_device(it.first->data, 0, it.second->data, 0, extent * dtype.bytes(), kDLCPU, 0,
                           device_type, device_id, dtype.code(), dtype.bits())));
        device_intermediate_api_args.push_back(it.second);
      }

      // Replace the buffers in the main_body
      main_body = VarReplacer(vsub, true)(main_body);
      prep_code = AttrStmtNode::make(prep_buffer_map, prep_attr->attr_key, prep_attr->value, SeqStmt(l_copy_stmts),
                        prep_attr->hfuse_group_id);
    }

    // Construct/rewrite prep_code
    prep_code = CopyStatementsRewriter(device_type, device_id)(prep_code);

    Array<ObjectRef> full_api_args;
    std::unordered_set<const Object*> cpu_args;
    full_api_args.push_back_all(tensor_api_args);
    full_api_args.push_back_all(lengths_api_args);
    for (auto buf : host_intermediate_api_args) {
      full_api_args.push_back(buf);
      cpu_args.insert(buf.get());
    }
    for (auto buf : device_intermediate_api_args) {
      full_api_args.push_back(buf);
    }

    for (auto obj : lengths_api_args) {
      cpu_args.insert(obj.get());
    }

    LoweredFunc full_func =
        MakeAPIInternal(SeqStmt({prep_code, main_body}), name, full_api_args, num_unpacked_args,
                        is_restricted, cpu_args, &vmap, &binder, &device_type, &device_id);

    return MakeAPIResultNode::make(full_func, host_intermediate_api_args, device_intermediate_api_args);
  }
}

class DeviceTypeBinder : public StmtExprMutator {
 public:
  explicit DeviceTypeBinder(int device_type) : device_type_(device_type) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::device_context_type) {
      if (const VarNode* var = op->value.as<VarNode>()) {
        var_ = var;
        PrimExpr value = make_const(op->value.dtype(), device_type_);
        Stmt body = StmtExprMutator::VisitStmt_(op);
        var_ = nullptr;
        std::ostringstream os;
        os << "device_type need to be " << device_type_;
        return AssertStmtNode::make(op->value == value, os.str(), body);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    // eager simplify if guard.
    Stmt res = StmtExprMutator::VisitStmt_(op);
    op = res.as<IfThenElseNode>();
    if (is_zero(op->condition)) {
      if (op->else_case.defined()) return op->else_case;
      return EvaluateNode::make(0);
    }
    if (is_one(op->condition)) {
      return op->then_case;
    }
    return res;
  }

  PrimExpr VisitExpr_(const NENode* op) final {
    // eager check NE for device check
    PrimExpr res = StmtExprMutator::VisitExpr_(op);
    op = res.as<NENode>();
    if (tir::Equal(op->a, op->b)) {
      return make_const(op->dtype, false);
    }
    return res;
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (op == var_) {
      return make_const(op->dtype, device_type_);
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

 public:
  const VarNode* var_{nullptr};
  int device_type_;
};

LoweredFunc BindDeviceType(LoweredFunc f, int device_type) {
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  n->body = DeviceTypeBinder(device_type)(n->body);
  return LoweredFunc(n);
}

}  // namespace tir
}  // namespace tvm
