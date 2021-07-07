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
 * \file tensor_layout_utils.h
 * \brief Common utilities to do tensor layout changes
 */
#ifndef TVM_TE_TENSOR_LAYOUT_UTILS_H_
#define TVM_TE_TENSOR_LAYOUT_UTILS_H_

#include <tvm/ir/attrs.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/expr_equality.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_set>

#include "../../arith/compute_expr.h"
#include "../../tir/ir/var_replacer.h"
#include "../../tir/pass/ir_util.h"
#include "graph.h"
#include "message_passing.h"
#include "schedule_utils.h"

namespace tvm {
namespace te {
class AccessPattern {
 public:
  // For each dimension of the tensor indexed by an Fdim, what
  // dimension in the reader access is used to index into the
  // tensor?
  Map<Dimension, Dimension> idx_dim_args;
  const CallNode* original_access;
  const BaseVarDimOpNode* reader_op;
  const UninterpFunNode* ufun;
  int idx;
  // What output tensor of the readr op is this access found in?
  int reader_val_idx;

  class Hasher {
   public:
    size_t operator()(const AccessPattern* pattern) const;
  };
  class Equality {
   public:
    bool operator()(const AccessPattern* p1, const AccessPattern* p2) const;
  };
};

using PatternsSet =
    std::unordered_set<AccessPattern*, AccessPattern::Hasher, AccessPattern::Equality>;
using AccessToPatternMap = std::unordered_map<const CallNode*, AccessPattern*>;
using PatternsVec = std::vector<AccessPattern*>;

class AccessPatternCollector {
  class ExprAccessPatternCollector : public ExprVisitor {
    void VisitExpr_(const CallNode* op) override;

   public:
    ExprAccessPatternCollector(const Tensor& tensor_, Array<Dimension> original_index_dimensions_,
                               PatternsSet* access_patterns_,
                               AccessToPatternMap* access_to_pattern_map_,
                               const BaseVarDimOpNode* reader_op_)
        : tensor(tensor_),
          original_index_dimensions(original_index_dimensions_),
          access_patterns(access_patterns_),
          access_to_pattern_map(access_to_pattern_map_),
          reader_op(reader_op_) {
      if (auto op = tensor->op.as<ComputeOpNode>()) {
        this->tensor_index_dims = op->root_index_dimensions;
      } else if (auto op = tensor->op.as<PlaceholderOpNode>()) {
        this->tensor_index_dims = op->self_index_dimensions;
      } else if (auto op = tensor->op.as<ScanOpNode>()) {
        for (size_t i = 0; i < tensor.ndim(); ++i) {
          this->tensor_index_dims.push_back(op->GetBaseIndexDimension(tensor->value_index, i));
        }
      } else if (auto op = tensor->op.as<ConditionalOpNode>()) {
        for (size_t i = 0; i < tensor.ndim(); ++i) {
          this->tensor_index_dims.push_back(op->GetBaseIndexDimension(tensor->value_index, i));
        }
      } else if (auto op = tensor->op.as<SingleKernelEnvelopeOpNode>()) {
        for (size_t i = 0; i < tensor.ndim(); ++i) {
          this->tensor_index_dims.push_back(op->GetBaseIndexDimension(tensor->value_index, i));
        }
      } else {
        CHECK(false) << "Cannot cache operation " << tensor;
      }
    }

    void collect(const UninterpFunNode* ufun, Map<Var, Dimension> var2dim_map_,
                 int reader_val_idx_);

    void collect(PrimExpr expr, Map<Var, Dimension> var2dim_map_, int reader_val_idx_);

    Dimension GetDimForVar(Var var);

    const Tensor& tensor;
    Array<Dimension> original_index_dimensions;
    PatternsSet* access_patterns;
    AccessToPatternMap* access_to_pattern_map;
    const BaseVarDimOpNode* reader_op;
    Array<Map<Var, Dimension>> var2dim_maps;
    Array<Dimension> tensor_index_dims;
    const UninterpFunNode* ufun;
    int reader_val_idx;
  };

 public:
  void collect();

  AccessPatternCollector(const Tensor& tensor_, Array<Dimension> original_index_dimensions_,
                         const Array<Operation>& readers_)
      : tensor(tensor_), original_index_dimensions(original_index_dimensions_), readers(readers_) {}

  const Tensor& tensor;
  Array<Dimension> original_index_dimensions;
  const Array<Operation>& readers;
  PatternsSet access_patterns;
  AccessToPatternMap access_to_pattern_map;
};

IterVar GetIterVarFromDim(Dimension dim, Array<IterVar>& index_variables,
                          Array<IterVar>& loop_variables, Array<Dimension>& index_dimensions,
                          Array<Dimension>& loop_dimensions);

IterVar GetIterVarFromDim(Dimension dim, Array<DimInfo>& dim_infos);

Operation ReplaceInputs(Operation reader, const AccessToPatternMap* patterns_map, Tensor cache,
                        Array<Dimension> cache_idx_dims, Array<Dimension> orig_idx_dims,
                        bool add_variant_dimension);

Operation ReplaceInputsGeneral(Stage s, Operation old_op, Operation new_op, Operation reader,
                               const Map<IterVar, Range>& dom_map, Array<Modes> root_layouts = {});
}  // namespace te
}  // namespace tvm

#endif
