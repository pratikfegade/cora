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
 * \file tvm/te/schedule.h
 * \brief Define a schedule.
 */
// Acknowledgement: Many schedule primitives originate from Halide and Loopy.
#ifndef TVM_TE_CACHE_INFO_H_
#define TVM_TE_CACHE_INFO_H_

#include <tvm/te/dimension.h>
#include <tvm/te/tensor.h>
#include <tvm/te/tensor_intrin.h>
#include <tvm/tir/expr.h>

#include <string>
#include <unordered_map>

namespace tvm {
namespace te {
class CacheInfo;

class CacheInfoNode : public runtime::Object {
 public:
  Operation orig;
  Operation cached;
  Array<Map<Dimension, Dimension>> variantMappings;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("orig", &orig);
    v->Visit("cached", &cached);
    v->Visit("variantMappings", &variantMappings);
  }

  TVM_DLL static CacheInfo make(Operation orig, Operation cached,
                                Array<Map<Dimension, Dimension>> variantMappings);

  static constexpr const char* _type_key = "te.CacheInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheInfoNode, Object);
};

class CacheInfo : public runtime::ObjectRef {
 public:
  CacheInfo() {}
  // construct from shared ptr.
  explicit CacheInfo(runtime::ObjectPtr<runtime::Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const CacheInfoNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = CacheInfoNode;
};

inline const CacheInfoNode* CacheInfo::operator->() const {
  return static_cast<const CacheInfoNode*>(data_.get());
}
}  // namespace te
}  // namespace tvm
#endif  // TVM_TE_CACHE_INFO_H_
