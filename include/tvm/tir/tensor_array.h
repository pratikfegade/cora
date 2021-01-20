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
 * \file tvm/tir/buffer.h
 * \brief Symbolic n-dimensional array, to represent a memory buffer.
 */
#ifndef TVM_TIR_TENSOR_ARRAY_H_
#define TVM_TIR_TENSOR_ARRAY_H_

#include <tvm/node/container.h>
#include <tvm/tir/expr.h>

#include <string>

namespace tvm {
namespace tir {
// TensorArray
class TensorArrayNode;

/*!
 * \brief TensorArray is an in-place TensorArray, lowered to a
 * single large buffer.
 */
class TensorArray : public ObjectRef {
 public:
  TensorArray() {}
  explicit TensorArray(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const TensorArrayNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = TensorArrayNode;
};

/*! \brief Node to represent a TensorArray */
class TensorArrayNode : public Object {
 public:
  // Data fields.
  /*! \brief The variable associated with the TensorArray. The same
      variable will be used when constructing the buffer during
      lowering */
  Var ta_var;
  /*! \brief The shape of the TensorArray */
  // TODO(ppf): Extend this to allow for jagged TensorArrays
  Array<PrimExpr> shape;
  /*! \brief optional name of the TensorArray */
  std::string name;

  /*! \brief constructor */
  TensorArrayNode() {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("var", &ta_var);
    v->Visit("shape", &shape);
    v->Visit("name", &name);
  }

  /*! \return preferred index type for this buffer node */
  DataType DefaultIndexType() const {
    return shape.size() != 0 ? shape[0].dtype() : DataType::Int(32);
  }

  static constexpr const char* _type_key = "TensorArray";
  TVM_DECLARE_BASE_OBJECT_INFO(TensorArrayNode, Object);
};

inline const TensorArrayNode* TensorArray::operator->() const {
  return static_cast<const TensorArrayNode*>(get());
}

/*! \brief Node to represent a RegionTensorArray */
class RegionTensorArrayNode : public TensorArrayNode {
 public:
  /*! \brief data type in the content of the tensor */
  DataType dtype;
  /*! \brief The shape of the Tensors contained in the TensorArray */
  // TODO(ppf): Extend this to allow for jagged TensorArrays
  Array<PrimExpr> tensor_shape;

  /*! \brief constructor */
  RegionTensorArrayNode() {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("tensor_shape", &tensor_shape);
    v->Visit("var", &ta_var);
    v->Visit("shape", &shape);
    v->Visit("name", &name);
  }

  TVM_DLL static TensorArray make(Var ta_var, DataType dtype, Array<PrimExpr> shape,
                                  Array<PrimExpr> tensor_shape, std::string name);

  static constexpr const char* _type_key = "RegionTensorArray";
  TVM_DECLARE_FINAL_OBJECT_INFO(RegionTensorArrayNode, TensorArrayNode);
};

/*!
 * \brief Construct a new RegionTensorArray given shape, and dtype.
 * \param shape The shape of the buffer,
 * \param dtype The content data type.
 * \param name The name of the buffer
 * \return The created buffer.
 * \sa RegionTensorArrayNode::make for complete constructor.
 */
TVM_DLL TensorArray decl_region_tensor_array(Array<PrimExpr> shape,
                                             DataType dtype = DataType::Float(32),
                                             std::string name = "region_ta");

/*! \brief Node to represent a PointerTensorArray */
class PointerTensorArrayNode : public TensorArrayNode {
 public:
  /*! \brief The RegionTensorArray that this array contains references to */
  TensorArray region_ta;

  /*! \brief constructor */
  PointerTensorArrayNode() {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("region_ta", &region_ta);
    v->Visit("var", &ta_var);
    v->Visit("shape", &shape);
    v->Visit("name", &name);
  }

  TVM_DLL static TensorArray make(Var ta_var, TensorArray region_ta, Array<PrimExpr> shape,
                                  std::string name);

  static constexpr const char* _type_key = "PointerTensorArray";
  TVM_DECLARE_FINAL_OBJECT_INFO(PointerTensorArrayNode, TensorArrayNode);
};

/*!
 * \brief Construct a new PointerTensorArray given shape, and dtype.
 * \param shape The shape of the buffer,
 * \param dtype The content data type.
 * \param name The name of the buffer
 * \return The created buffer.
 * \sa PointerTensorArrayNode::make for complete constructor.
 */
TVM_DLL TensorArray decl_pointer_tensor_array(Array<PrimExpr> shape, TensorArray region_ta,
                                              std::string name = "pointer_ta");

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TENSOR_ARRAY_H_
