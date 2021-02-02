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
 * \file tvm/te/ta_declarations.h
 * \brief Define a ta_declarations.
 */
// Akcnowledgement: Many ta_declarations primitives originate from Halide and Loopy.
#ifndef TVM_TIR_TA_DECLARATIONS_H_
#define TVM_TIR_TA_DECLARATIONS_H_

#include <tvm/tir/buffer.h>
#include <tvm/tir/tensor_array.h>

#include <string>
#include <unordered_map>

namespace tvm {
namespace tir {

// Node container for TADeclarations
class TADeclarationsNode;

/*!
 * \brief Global ta_declarations container
 *  For operations and all the operations they depend on.
 *  The ta_declarations per Operation is named as stage.
 */
class TADeclarations : public ObjectRef {
 public:
  TADeclarations() {}
  explicit TADeclarations(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief Get a copy of current schedule.
   * \return The copied schedule.
   */
  TADeclarations copy() const;
  /*!
   * \brief Add or replace a tensor array
   *
   * \param tensor_array The tensor_array to be added.
   * \return The ta_declarations object.
   */
  TVM_DLL TADeclarations add_tensor_array(TensorArray tensor_array);
  /*!
   * \brief Add or replace a buffer
   *
   * \param buffer The buffer to be added.
   * \return The ta_declarations object.
   */
  TVM_DLL TADeclarations add_buffer(Buffer buffer);
  /*!
   * \brief Get the tensor array corresponding to the given variable.
   *
   * \return the tensor array.
   */
  TVM_DLL TensorArray get_tensor_array(Var var);
  /*!
   * \brief Get the buffer corresponding to the given variable.
   *
   * \return the buffer.
   */
  TVM_DLL Buffer get_buffer(Var var);
  /*!
   * \brief Get all base tensor arrays.
   *
   * \return the tensor arrays.
   */
  TVM_DLL Array<TensorArray> get_base_tensor_arrays() const;
  /*!
   * \brief Get all tensor arrays.
   *
   * \return the tensor arrays.
   */
  TVM_DLL Array<TensorArray> get_all_tensor_arrays() const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const TADeclarationsNode* operator->() const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline TADeclarationsNode* operator->();
  // declare container type
  using ContainerType = TADeclarationsNode;
};

/*! \brief node container for ta_declarations */
class TADeclarationsNode : public Object {
 public:
  /*!
   * \brief Map to map vars to tensor_arrays.
   *  This is created on demand and can be invalidated.
   */
  std::unordered_map<const Object*, TensorArray> var2ta_map;
  /*!
   * \brief Map to map vars to buffers.
   *  This is created on demand and can be invalidated.
   */
  std::unordered_map<const Object*, Buffer> var2buf_map;

  void VisitAttrs(AttrVisitor* v) {}

  /*!
   * \brief Create a ta_declarations for given tensor_arrays and buffers.
   * \param ops The buffers and tensor_arrays declared and used in the computation.
   * \return sch The created TADeclarations.
   */
  TVM_DLL static TADeclarations make(const Array<tir::TensorArray> tensor_arrays,
                                     const Array<tir::Buffer> buffers);

  static constexpr const char* _type_key = "TADeclarations";
  TVM_DECLARE_FINAL_OBJECT_INFO(TADeclarationsNode, Object);
};

/*!
 * \brief Create a ta_declarations for array of ops(and their dependencies).
 * \param ops The ops to be ta_declarationsd.
 * \return sch The created TADeclarations.
 */
inline TADeclarations create_ta_declarations(const Array<tir::TensorArray> tensor_arrays,
                                             const Array<tir::Buffer> buffers) {
  return TADeclarationsNode::make(tensor_arrays, buffers);
}

inline const TADeclarationsNode* TADeclarations::operator->() const {
  return static_cast<const TADeclarationsNode*>(get());
}
inline TADeclarationsNode* TADeclarations::operator->() {
  return static_cast<TADeclarationsNode*>(get_mutable());
}

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_TA_DECLARATIONS_H_
