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
 * \file buffer.cc
 */
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>

#include <iterator>
#include <stack>

#include "../../arith/compute_expr.h"

namespace tvm {
namespace tir {
// TODO(tqchen): change to floormod/div
using IndexMod = tir::FloorModNode;
using IndexDiv = tir::FloorDivNode;

Array<PrimExpr> SimplifyArray(Array<PrimExpr> array) {
  for (size_t i = 0; i < array.size(); ++i) {
    array.Set(i, tir::Simplify(array[i]));
  }
  return array;
}

Buffer decl_buffer(Array<PrimExpr> shape, DataType dtype, std::string name, SyncType sync_type) {
  return BufferNode::make(Var(name, DataType::Handle()), dtype, shape, Array<PrimExpr>(),
                          PrimExpr(), name, "", 0, 0, kDefault, sync_type);
}

Buffer decl_buffer(Modes shape, DataType dtype, std::string name, SyncType sync_type) {
  return BufferNode::make(Var(name, DataType::Handle()), dtype, shape, Array<PrimExpr>(),
                          PrimExpr(), name, "", 0, 0, kDefault, sync_type);
}

// Split the given expression w.r.t the add operator
inline std::vector<const PrimExpr*> ExprSplitAddition(const PrimExpr& expr) {
  using namespace tir;
  std::vector<const PrimExpr*> ret;
  std::stack<const PrimExpr*> split_buffer;
  split_buffer.push(&expr);
  while (!split_buffer.empty()) {
    const PrimExpr* top_ele = split_buffer.top();
    split_buffer.pop();
    auto expr_add_match = top_ele->as<AddNode>();
    if (expr_add_match) {
      split_buffer.push(&expr_add_match->b);
      split_buffer.push(&expr_add_match->a);
    } else {
      ret.emplace_back(top_ele);
    }
  }
  return ret;
}

// Searches for the following types of expr:
//   mult_expr = (a1 + a2 + ... + aj + c / (k1 * k2 * ... * ki) * k1 * ... * kt-1 ) * kt * ... * ki
//   mod_l_expr = c
//   mod_r_expr = k1 * k2 * ... * ki
// If it can be optimized, returns (true, (a1 + a2 + ... + aj) * kt * ... * ki + c)
// Currently the we will not search the add/mult combinations exhaustively
//   as it will take too much computation.
inline std::pair<bool, PrimExpr> MergeMulModInner(const PrimExpr& mult_expr,
                                                  const PrimExpr& mod_l_expr,
                                                  const PrimExpr& mod_r_expr) {
  using namespace tir;
  const MulNode* mult_ptr = mult_expr.as<MulNode>();
  if (!mult_ptr) return std::make_pair(false, PrimExpr());
  PrimExpr mult_outer = mult_ptr->b;
  const PrimExpr* inner = &(mult_ptr->a);
  // 1. Calculate the outer multiplier
  while (true) {
    mult_ptr = inner->as<MulNode>();
    if (mult_ptr) {
      inner = &(mult_ptr->a);
      mult_outer = mult_ptr->b * mult_outer;
    } else {
      break;
    }
  }
  // 2. Search for the pattern c / (...) * (...) + c % (...)
  // We match the search element with Add, Mul and Div.
  //   If Add is found, we need to continue our search for the rhs
  //   If Mult is found, we will expand the inner multiplication factor
  //   If Div is found, we will go on testing whether lhs matches the lhs of mod expr
  //      and returns the optimization result.
  const PrimExpr* search_ptr = inner;
  PrimExpr mult_inner;  // The inner multiplication factor
  PrimExpr no_opt_sum;  // Sum of the exprs that cannot be optimized
  while (true) {
    auto inner_div_ptr = search_ptr->as<IndexDiv>();
    auto inner_mult_ptr = search_ptr->as<MulNode>();
    auto inner_add_ptr = search_ptr->as<AddNode>();
    if (!inner_div_ptr && !inner_mult_ptr && !inner_add_ptr) {
      return std::make_pair(false, PrimExpr());
    } else if (inner_div_ptr) {
      PrimExpr overall_mult = mult_inner.get() ? mult_inner * mult_outer : mult_outer;
      if (Equal(overall_mult, inner_div_ptr->b) && Equal(overall_mult, mod_r_expr) &&
          Equal(inner_div_ptr->a, mod_l_expr)) {
        // Found!
        PrimExpr ret = no_opt_sum.get() ? no_opt_sum * mult_outer + mod_l_expr : mod_l_expr;
        return std::make_pair(true, ret);
      } else {
        return std::make_pair(false, PrimExpr());
      }
    } else if (inner_mult_ptr) {
      mult_inner = mult_inner.get() ? inner_mult_ptr->b * mult_inner : inner_mult_ptr->b;
      search_ptr = &(inner_mult_ptr->a);
    } else if (inner_add_ptr) {
      if (mult_inner.get()) {
        return std::make_pair(false, PrimExpr());
      }
      no_opt_sum = no_opt_sum.get() ? no_opt_sum + inner_add_ptr->a : inner_add_ptr->a;
      search_ptr = &(inner_add_ptr->b);
    } else {
      LOG(FATAL) << "Unexpected search result!";
      break;
    }
  }
  return std::make_pair(false, PrimExpr());
}

// Insert the elements into the corresponding mult_exprs and mod_exprs.
// If the element is found to match Mul, it will be pushed to the mult_exprs.
// If the element it found to match Mod, it will be pused to the mod_exprs.
// Otherwise, the elements will be added to the no_opt_sum variable
inline void MergeMulModInsertElements(const std::vector<const PrimExpr*>& eles,
                                      std::list<PrimExpr>* mult_exprs,
                                      std::list<std::pair<PrimExpr, PrimExpr> >* mod_exprs,
                                      PrimExpr* no_opt_sum, bool* has_mult, bool* has_mod) {
  using namespace tir;
  *has_mult = false;
  *has_mod = false;
  for (const PrimExpr* ele : eles) {
    auto mod_ptr = ele->as<IndexMod>();
    auto mult_ptr = ele->as<MulNode>();
    if (mod_ptr) {
      *has_mod = true;
      mod_exprs->emplace_back(std::make_pair(std::move(mod_ptr->a), std::move(mod_ptr->b)));
    } else if (mult_ptr) {
      *has_mult = true;
      mult_exprs->emplace_back(*ele);
    } else {
      *no_opt_sum = no_opt_sum->get() ? *no_opt_sum + *ele : *ele;
    }
  }
}

// Searches for this types of expr:
//   (a1 + a2 + ... + aj + c / (k1 * k2 * ... * ki) * k1 * ... * kt-1 ) * kt * ... * ki
//   + c % (k1 * k2 * ... * ki)
// and simplifies to (a1 + a2 + ... + aj) * kt * ... * ki + c
// The search will be performed repeatively until no pattern is found.
// Return: a pair with (false, Expr()) if cannot be optimized.
//         a pair with (true, optimized_expr) if can be optimized
inline PrimExpr MergeMulMod(const PrimExpr& base) {
  using namespace tir;
  // 1. Prepare the lists.
  // We store two lists, a list that contain all the elements that match Mul and
  //                     a list that contain all the elements that match Mod.
  // The elements in the Mod will be used to match against the elements in Mul.
  // The result will then be split and pushed back to these two lists.
  PrimExpr simplified_base = Simplify(base);
  std::vector<const PrimExpr*> eles = ExprSplitAddition(simplified_base);
  std::list<PrimExpr> mult_exprs;
  std::list<std::pair<PrimExpr, PrimExpr> > mod_exprs;
  PrimExpr no_opt_sum;
  bool has_mult;
  bool has_mod;
  MergeMulModInsertElements(eles, &mult_exprs, &mod_exprs, &no_opt_sum, &has_mult, &has_mod);
  bool find_opt = false;
  std::list<std::pair<PrimExpr, PrimExpr> >::iterator search_mod_it = mod_exprs.begin();
  // 2. Exhaustive Search
  while (search_mod_it != mod_exprs.end()) {
    std::list<PrimExpr>::iterator mult_it = mult_exprs.begin();
    bool inner_find_opt = false;
    while (mult_it != mult_exprs.end()) {
      std::pair<bool, PrimExpr> ret =
          MergeMulModInner(*mult_it, search_mod_it->first, search_mod_it->second);
      if (ret.first) {
        inner_find_opt = true;
        auto temp_mod_it = search_mod_it;
        ++search_mod_it;
        mod_exprs.erase(temp_mod_it);
        mult_exprs.erase(mult_it);
        std::vector<const PrimExpr*> ret_eles = ExprSplitAddition(ret.second);
        MergeMulModInsertElements(ret_eles, &mult_exprs, &mod_exprs, &no_opt_sum, &has_mult,
                                  &has_mod);
        if (has_mult) {
          search_mod_it = mod_exprs.begin();
        } else if (has_mod && search_mod_it == mod_exprs.end()) {
          search_mod_it--;
        }
        break;
      } else {
        ++mult_it;
      }
    }
    find_opt = find_opt || inner_find_opt;
    if (!inner_find_opt) {
      ++search_mod_it;
    }
  }
  if (!find_opt) {
    return simplified_base;
  }
  for (std::list<PrimExpr>::iterator it = mult_exprs.begin(); it != mult_exprs.end(); ++it) {
    no_opt_sum = no_opt_sum.get() ? no_opt_sum + *it : *it;
  }
  for (std::list<std::pair<PrimExpr, PrimExpr> >::iterator it = mod_exprs.begin();
       it != mod_exprs.end(); ++it) {
    no_opt_sum = no_opt_sum.get() ? no_opt_sum + indexmod(it->first, it->second)
                                  : indexmod(it->first, it->second);
  }
  return no_opt_sum;
}

// The buffer offset in convention of number of elements of
// original data ignoring number of lanes.
// We also perform optimization to simplify the indexing expression.
inline PrimExpr ElemOffset(const BufferNode* n, Array<PrimExpr> index) {
  index = n->TransformViaView(index);
  auto dense_shape = n->shape->get_dense_shape();
  PrimExpr base = n->elem_offset;
  bool print = false;  //(n->data->name_hint == "B");
  // if (print) {
  //   std::cout << "[BEO] For " << n->data << " " << n->strides.size() << " " << base << std::endl;
  //   for (size_t i = 0; i < dense_shape.size(); ++i) {
  //     std::cout << "[BEO]    Shape/Index " << dense_shape[i] << " " << index[i] << std::endl;
  //   }
  // }

  if (n->strides.size() == 0) {
    if (n->shape->is_ragged()) {
      if (print) {
        std::cout << "[BEO] Ragged lowering for buffer " << n->data << " " << n->shape << " "
                  << n->name << std::endl;
        for (auto idx : index) {
          std::cout << "[BEO]   Index " << idx << std::endl;
        }
      }
      base = n->shape->ComputePosition(n->name, index);
    } else {
      // Scalar case
      if (dense_shape.size() == 0 && index.size() == 1) {
        auto is_int = index[0].as<IntImmNode>();
        CHECK(is_int && is_int->value == 0);
        base = base + index[0];
      } else {
        CHECK_EQ(dense_shape.size(), index.size()) << n->data << " " << n->view_transforms.size();
        if (index.size() > 0) {
          PrimExpr offset = index[0];
          for (size_t i = 1; i < index.size(); ++i) {
            // offset = MergeMulMod(offset * dense_shape[i] + index[i]);
            offset = offset * dense_shape[i] + index[i];
            if (print)
              std::cout << "[BEO]   It " << i << " " << dense_shape[i] << " " << index[i] << " "
                        << offset << std::endl;
          }
          base = base + offset;
        }
      }
    }
  } else {
    CHECK_EQ(n->strides.size(), index.size()) << n->name << " " << n->data;
    if (is_zero(base)) {
      base = MergeMulMod(index[0] * n->strides[0]);
    } else {
      base = MergeMulMod(base + index[0] * n->strides[0]);
    }
    for (size_t i = 1; i < index.size(); ++i) {
      base = MergeMulMod(base + index[i] * n->strides[i]);
      if (print) std::cout << "[BEO]   " << n->data << " " << base << std::endl;
    }
  }
  return base;
}

inline PrimExpr BufferOffset(const BufferNode* n, Array<PrimExpr> index, DataType dtype) {
  PrimExpr offset = ElemOffset(n, index);
  if (n->dtype.lanes() != 1) {
    offset = offset * make_const(offset.dtype(), dtype.lanes());
  }
  if (dtype.lanes() != 1) {
    return tir::RampNode::make(offset, make_const(offset.dtype(), 1), dtype.lanes());
  } else {
    return offset;
  }
}

PrimExpr Buffer::vload(Array<PrimExpr> begin, DataType dtype, SyncType sync_type) const {
  // specially handle bool, stored asDataType::Int(8)
  const BufferNode* n = operator->();
  CHECK(dtype.element_of() == n->dtype.element_of() && dtype.lanes() % n->dtype.lanes() == 0)
      << "Cannot load " << dtype << " from buffer of " << n->dtype;
  if (dtype == DataType::Bool()) {
    return tir::CastNode::make(
        DataType::Bool(),
        tir::LoadNode::make(DataType::Int(8), n->data, BufferOffset(n, begin, DataType::Int(8)),
                            const_true(), sync_type));
  } else {
    return tir::LoadNode::make(dtype, n->data, BufferOffset(n, begin, dtype),
                               const_true(dtype.lanes()), sync_type);
  }
}

Stmt Buffer::vstore(Array<PrimExpr> begin, PrimExpr value, SyncType sync_type) const {
  // specially handle bool, stored asDataType::Int(8)
  const BufferNode* n = operator->();
  DataType dtype = value.dtype();
  CHECK(dtype.element_of() == n->dtype.element_of() && dtype.lanes() % n->dtype.lanes() == 0)
      << "Cannot load " << dtype << " from buffer of " << n->dtype;
  if (value.dtype() == DataType::Bool()) {
    return tir::StoreNode::make(n->data, tir::CastNode::make(DataType::Int(8), value),
                                BufferOffset(n, begin, DataType::Int(8)), const_true(), sync_type);
  } else {
    return tir::StoreNode::make(n->data, value, BufferOffset(n, begin, dtype),
                                const_true(dtype.lanes()), sync_type);
  }
}

Buffer Buffer::MakeStrideView() const {
  // std::cout << "[BUF] StrideView for " << (*this)->name << std::endl;
  if ((*this)->strides.size() != 0) return *this;
  if ((*this)->shape->ndim() == 0) return *this;
  std::vector<PrimExpr> temp;
  auto n = make_object<BufferNode>(*operator->());
  PrimExpr acc = make_const(n->DefaultIndexType(), 1);
  auto dense_shape = n->shape->get_dense_shape();
  for (size_t i = n->shape->ndim(); i != 0; --i) {
    temp.push_back(acc);
    acc = acc * dense_shape[i - 1];
  }
  for (size_t i = temp.size(); i != 0; --i) {
    n->strides.push_back(temp[i - 1]);
  }
  return Buffer(n);
}

Buffer Buffer::MakeSlice(Array<PrimExpr> begins, Array<PrimExpr> extents) const {
  // std::cout << "[BUF] SliceView for " << (*this)->name << std::endl;
  const BufferNode* n = operator->();
  begins = SimplifyArray(begins);
  PrimExpr elem_offset = tir::Simplify(ElemOffset(n, begins));
  Array<PrimExpr> strides = n->strides;
  if (strides.size() == 0) {
    bool can_relax = true;
    bool need_stride = false;
    // check if stride is needed.
    auto dense_shape = n->shape->get_dense_shape();
    for (size_t i = 0; i < extents.size(); ++i) {
      if (!can_relax) {
        if (!is_zero(begins[i]) || !is_zero(tir::Simplify(extents[i] - dense_shape[i]))) {
          need_stride = true;
        }
      }
      if (!is_one(extents[i])) can_relax = false;
    }
    // make stride.
    if (need_stride) {
      return MakeStrideView().MakeSlice(begins, extents);
    }
  }
  return BufferNode::make(n->data, n->dtype, extents, strides, elem_offset, n->name + "_slice",
                          n->scope, n->data_alignment, 0, n->buffer_type, n->sync_type);
}

PrimExpr Buffer::access_ptr(int access_mask, DataType ptr_type, int content_lanes,
                            PrimExpr offset) const {
  const BufferNode* self = operator->();
  PrimExpr e_dtype;
  PrimExpr extent;
  auto dense_shape = self->shape->get_dense_shape();
  if (self->shape->ndim() == 0) {
    extent = make_const(self->DefaultIndexType(), 1);
  } else if (self->strides.size() == self->shape->ndim()) {
    int highest_dim = 0;
    extent = self->strides[highest_dim] * dense_shape[highest_dim] - offset;
  } else {
    extent = arith::ComputeReduce<tir::MulNode>(dense_shape, PrimExpr()) - offset;
  }
  PrimExpr elem_offset = self->elem_offset + offset;
  if (content_lanes > 1) {
    e_dtype = tir::TypeAnnotation(self->dtype.with_lanes(content_lanes));
    extent = extent / make_const(self->elem_offset.dtype(), content_lanes);
    elem_offset = self->elem_offset / make_const(self->elem_offset.dtype(), content_lanes);
  } else {
    e_dtype = tir::TypeAnnotation(self->dtype);
  }
  Array<PrimExpr> acc_args{e_dtype, self->data, elem_offset, extent,
                           make_const(DataType::Int(32), access_mask)};
  return tir::CallNode::make(ptr_type, tir::intrinsic::tvm_access_ptr, acc_args,
                             tir::CallNode::Intrinsic);
}

Array<PrimExpr> BufferNode::TransformViaView(Array<PrimExpr> pretransformed_extents) const {
  if (this->view_transforms.size() > 0) {
    Array<PrimExpr> transformed_extents;
    Array<Dimension> pretransformed_dimensions = this->pretransformed_dimensions;
    for (auto transform : this->view_transforms) {
      transformed_extents.push_back(UninterpFun::InlineUninterpFunCalls(
          transform.MakeCallTo(pretransformed_extents, pretransformed_dimensions)));
    }
    return transformed_extents;
  } else {
    return pretransformed_extents;
  }
}

Buffer BufferNode::make(Var data, DataType dtype, Array<PrimExpr> shape, Array<PrimExpr> strides,
                        PrimExpr elem_offset, std::string name, std::string scope,
                        int data_alignment, int offset_factor, BufferType buffer_type,
                        SyncType sync_type, Array<UninterpFun> view_transforms,
                        Array<Dimension> pretransformed_dimensions) {
  return BufferNode::make(data, dtype, ModesNode::make(name, shape, false), strides, elem_offset,
                          name, scope, data_alignment, offset_factor, buffer_type, sync_type,
                          view_transforms, pretransformed_dimensions);
}

Buffer BufferNode::make(Var data, DataType dtype, Modes shape, Array<PrimExpr> strides,
                        PrimExpr elem_offset, std::string name, std::string scope,
                        int data_alignment, int offset_factor, BufferType buffer_type,
                        SyncType sync_type, Array<UninterpFun> view_transforms,
                        Array<Dimension> pretransformed_dimensions) {
  auto n = make_object<BufferNode>();
  n->data = std::move(data);
  n->dtype = dtype;
  n->shape = std::move(shape);
  n->strides = std::move(strides);
  n->name = std::move(name);
  if (scope.length() == 0) {
    scope = "global";
  }
  n->scope = std::move(scope);
  if (!elem_offset.defined()) {
    elem_offset = make_const(n->DefaultIndexType(), 0);
  }
  if (data_alignment <= 0) {
    data_alignment = runtime::kAllocAlignment;
  }
  if (offset_factor == 0) {
    offset_factor = 1;
  }
  n->elem_offset = std::move(elem_offset);
  n->data_alignment = data_alignment;
  n->offset_factor = offset_factor;
  n->buffer_type = buffer_type;
  n->sync_type = std::move(sync_type);
  n->view_transforms = std::move(view_transforms);
  n->pretransformed_dimensions = std::move(pretransformed_dimensions);
  if (n->buffer_type == kAutoBroadcast && n->shape->ndim() > 0 && n->strides.empty()) {
    for (size_t i = 0; i < n->shape->ndim(); ++i) {
      n->strides.push_back(Var("stride"));
    }
  }

  return Buffer(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BufferNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const BufferNode*>(node.get());
      p->stream << "buffer(" << op->name << ", " << op << ")";
    });

TVM_REGISTER_NODE_TYPE(BufferNode);

TVM_REGISTER_GLOBAL("tir.Buffer")
    .set_body_typed([](Var data, DataType dtype, Array<PrimExpr> shape, Array<PrimExpr> strides,
                       PrimExpr elem_offset, std::string name, std::string scope,
                       int data_alignment, int offset_factor, std::string buffer_type,
                       int sync_type, Array<UninterpFun> view_transforms,
                       Array<Dimension> pretransformed_dimensions) {
      BufferType type = (buffer_type == "auto_broadcast") ? kAutoBroadcast : kDefault;
      return BufferNode::make(data, dtype, shape, strides, elem_offset, name, scope, data_alignment,
                              offset_factor, type, static_cast<SyncType>(sync_type),
                              view_transforms, pretransformed_dimensions);
    });

TVM_REGISTER_GLOBAL("tir.BufferWithModes")
    .set_body_typed([](Var data, DataType dtype, Modes shape, Array<PrimExpr> strides,
                       PrimExpr elem_offset, std::string name, std::string scope,
                       int data_alignment, int offset_factor, std::string buffer_type,
                       int sync_type, Array<UninterpFun> view_transforms,
                       Array<Dimension> pretransformed_dimensions) {
      BufferType type = (buffer_type == "auto_broadcast") ? kAutoBroadcast : kDefault;
      return BufferNode::make(data, dtype, shape, strides, elem_offset, name, scope, data_alignment,
                              offset_factor, type, static_cast<SyncType>(sync_type),
                              view_transforms, pretransformed_dimensions);
    });

TVM_REGISTER_GLOBAL("tir.BufferAccessPtr").set_body_method(&Buffer::access_ptr);

TVM_REGISTER_GLOBAL("tir.BufferVStore")
    .set_body_typed([](Buffer buf, Array<PrimExpr> begin, PrimExpr value, int sync_type) {
      return buf.vstore(begin, value, static_cast<SyncType>(sync_type));
    });

TVM_REGISTER_GLOBAL("tir.BufferVLoad")
    .set_body_typed([](Buffer buf, Array<PrimExpr> begin, DataType dtype, int sync_type) {
      return buf.vload(begin, dtype, static_cast<SyncType>(sync_type));
    });
}  // namespace tir
}  // namespace tvm
