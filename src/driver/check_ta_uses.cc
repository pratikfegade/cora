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
 *  Compile executable modules.
 * \file driver_api.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/driver/driver_api.h>
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>
#include <tvm/te/dimension.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/ta_declarations.h>
#include <tvm/tir/te_capsule.h>

#include <algorithm>
#include <mutex>
#include <stack>

#include "../te/schedule/graph.h"
#include "../tir/ir/var_replacer.h"

namespace tvm {
using namespace tvm::te;
using namespace tvm::tir;

class TAChecker : public StmtExprVisitor {
 public:
  TAChecker(const TADeclarations declarations_) : declarations(declarations_) {}

  void check(const tir::Stmt& input_program) { this->VisitStmt(input_program); }

  void VisitExpr_(const RegionTALoadNode* load) override {
    StmtExprVisitor::VisitExpr_(load);
    Var ta_var = load->region_ta;
    TensorArray ta = declarations.get_tensor_array(ta_var);
    Array<PrimExpr> indices = load->indices;
    CHECK_EQ(indices.size(), ta->shape.size())
        << "Incorrect indexing for RegionTA load in " << GetRef<PrimExpr>(load);
  }

  void VisitExpr_(const PointerTALoadNode* load) override {
    StmtExprVisitor::VisitExpr_(load);
    Var ta_var = load->pointer_ta;
    TensorArray ta = declarations.get_tensor_array(ta_var);
    Array<PrimExpr> indices = load->indices;
    CHECK_EQ(indices.size(), ta->shape.size())
        << "Incorrect indexing for PointerTA load in " << GetRef<PrimExpr>(load);
  }

  Array<PrimExpr> GetShape(PrimExpr expr) {
    if (auto load = expr.as<RegionTALoadNode>()) {
      return declarations.get_tensor_array(load->region_ta)
          .as<RegionTensorArrayNode>()
          ->tensor_shape;
    } else if (auto load = expr.as<PointerTALoadNode>()) {
      return declarations.get_tensor_array(load->pointer_ta)
          ->GetBaseTensorArray()
          .as<RegionTensorArrayNode>()
          ->tensor_shape;
    } else if (expr.as<VarNode>()) {
      Buffer buffer = declarations.get_buffer(Downcast<Var>(expr));
      return buffer->shape;
    } else {
      CHECK(false) << expr;
      return {};
    }
  }

  void VisitStmt_(const RegionTAStoreNode* store) override {
    StmtExprVisitor::VisitStmt_(store);
    CHECK_EQ(store->region_tas.size(), store->region_ta_indices.size());
    for (size_t i = 0; i < store->region_tas.size(); ++i) {
      Var ta_var = store->region_tas[i];
      TensorArray ta = declarations.get_tensor_array(ta_var);
      Array<PrimExpr> indices = store->region_ta_indices[i];
      CHECK_EQ(indices.size(), ta->shape.size())
          << "Incorrect indexing for RegionTA store in " << ta << " " << GetRef<Stmt>(store);
    }

    CHECK((store->direct_inputs.size() > 0) != (store->inputs.size() > 0))
        << "Only one of inputs and direct_inputs should be provided " << GetRef<Stmt>(store);

    if (store->direct_inputs.size() > 0) {
    } else {
      CHECK(TECapsule::capsules.count(store->te_graph_name))
          << "No such TECaspule found " << store->te_graph_name;

      const TECapsuleNode* capsule = TECapsule::capsules.at(store->te_graph_name);

      CHECK_EQ(capsule->input_vars.size() + capsule->inputs.size(), store->inputs.size())
          << "Incorrect number of inputs to capsule " << GetRef<Stmt>(store);

      for (size_t i = 0; i < capsule->input_vars.size(); ++i) {
        CHECK(capsule->input_vars[i].dtype() == store->inputs[i].dtype())
            << capsule->input_vars[i] << " " << store->inputs[i] << capsule->input_vars[i].dtype()
            << " " << store->inputs[i].dtype();
      }

      for (size_t i = 0; i < capsule->inputs.size(); ++i) {
        te::Tensor tensor = capsule->inputs[i];
        PrimExpr input_expr = store->inputs[i + capsule->input_vars.size()];

        Array<PrimExpr> tensor_shape = tensor->shape;
        Array<PrimExpr> input_shape = GetShape(input_expr);
        CHECK_EQ(tensor_shape.size(), input_shape.size())
            << "Incorrect input shape for input tensor " << tensor << " in " << GetRef<Stmt>(store);
        for (size_t j = 0; j < input_shape.size(); ++j) {
          // This check may not always work as the shape of the
          // capsule tensor may be a function of an parameter variable
          // to the capsule. So before performing the check, we need
          // to substibute the input arguments in the capsule tensor's
          // shape.

          // CHECK(is_zero(input_shape[j] - tensor_shape[j])) <<
          //     "Incorrect input shape extent for input tensor " << i
          //     << " at dim " << j << " in " << GetRef<Stmt>(store);
        }
      }

      for (size_t i = 0; i < capsule->outputs.size(); ++i) {
        te::Tensor tensor = capsule->outputs[i];

        Array<PrimExpr> tensor_shape = tensor->shape;
        TensorArray output_ta = declarations.get_tensor_array(store->region_tas[i]);
        Array<PrimExpr> output_shape = output_ta.as<RegionTensorArrayNode>()->tensor_shape;
        CHECK_EQ(tensor_shape.size(), output_shape.size())
            << "Incorrect input shape for output tensor " << i << " in " << GetRef<Stmt>(store);
        for (size_t j = 0; j < output_shape.size(); ++j) {
          // This check may not always work as the shape of the
          // capsule tensor may be a function of an parameter variable
          // to the capsule. So before performing the check, we need
          // to substibute the input arguments in the capsule tensor's
          // shape.

          // CHECK(is_zero(output_shape[j] - tensor_shape[j]))
          //     << "Incorrect output shape extent for output tensor " << i << " at dim " << j
          //     << " in " << GetRef<Stmt>(store);
        }
      }
    }
  }

  void VisitStmt_(const PointerTAStoreNode* store) override {
    StmtExprVisitor::VisitStmt_(store);
    TensorArray pointer_ta = declarations.get_tensor_array(store->pointer_ta);
    TensorArray base_ta = pointer_ta->GetBaseTensorArray();
    CHECK_EQ(pointer_ta->shape.size(), store->pointer_ta_indices.size());
    CHECK_EQ(base_ta->shape.size(), store->region_ta_indices.size());
  }

 private:
  TADeclarations declarations;
};

void check_ta_uses(const TADeclarations declarations, const tir::Stmt& input_program) {
  // std::cout << "[TE] Checking TA uses" << std::endl;
  TAChecker checker(declarations);
  checker.check(input_program);
}

TVM_REGISTER_GLOBAL("tir.check_ta_uses").set_body_typed(check_ta_uses);

}  // namespace tvm
