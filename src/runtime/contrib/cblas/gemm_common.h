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
 * \file tvm/contrib/gemm.h
 * \brief Shared implementation of gemm
 */
#pragma once

#include <tvm/runtime/registry.h>
#include <tvm/runtime/data_type.h>
#include <algorithm>
/* #include <cmath> */

/* #include <fenv.h> */

namespace tvm {
namespace contrib {

using namespace runtime;
inline int ColumnStride(DLTensor *tensor) {
  // If the tensor itself is transposed then it will have strides
  // backward from what we expect.  Regardless, the max of the strides
  // (the other stride is 1) is the column stride.
  if (tensor->strides) {
    return std::max(tensor->strides[0], tensor->strides[1]);
  } else {
    return tensor->shape[1];
  }
}

inline int ElementStride(DLTensor *tensor) {
  if (tensor->strides) {
    return std::min(tensor->strides[0], tensor->strides[1]);
  } else {
    return 1;
  }
}

// Reversed strides indicates an in-place transpose operation.
inline bool IsInPlaceTransposed(DLTensor *tensor) {
  return tensor->strides && (tensor->strides[1] > tensor->strides[0]);
}

inline int RowCount(DLTensor *tensor, bool trans) {
  return tensor->shape[trans ? 1 : 0];
}

inline int ColumnCount(DLTensor *tensor, bool trans) {
  return tensor->shape[trans ? 0 : 1];
}

 /* inline void CheckForNan(DLTensor *C) { */
 /*   float* c_data = (float*)(C->data); */
 /*   int cr = RowCount(C, false); */
 /*   int cc = ColumnCount(C, false); */
 /*   for (int i = 0; i < cr; ++i) { */
 /*     for (int j = 0; j < cc; ++j) { */
 /*       CHECK(!std::isnan(c_data[i * cc + j])); */
 /*       CHECK(!std::isnan(1 / (1 + exp(- c_data[i * cc + j])))); */
 /*     } */
 /*   } */
 /* } */

// Call a column major blas.  Note that data is stored in tvm as row
// major, so this we switch the arguments.
template <typename TGemmOp>
inline void CallGemm(TVMArgs args, TVMRetValue *ret, TGemmOp op) {
  /* feenableexcept(FE_INVALID | FE_OVERFLOW); */
  DLTensor *A = args[0];
  DLTensor *B = args[1];
  DLTensor *C = args[2];
  bool transa = args[3];
  bool transb = args[4];
  int bit_depth = sizeof(typename TGemmOp::TDatatype) * 8;
  CHECK_EQ(A->ndim, 2);
  CHECK_EQ(B->ndim, 2);
  CHECK_EQ(C->ndim, 2);

  CHECK_EQ(ElementStride(A), 1);
  CHECK_EQ(ElementStride(B), 1);
  CHECK_EQ(ElementStride(C), 1);

  // C can never be transposed.
  CHECK(!IsInPlaceTransposed(C));

  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;

  CHECK(TypeMatch(B->dtype, kDLFloat, bit_depth));
  CHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));
  double alpha = args.size() > 5 ? args[5] : 1.0;
  double beta = args.size() > 6 ? args[6] : 0.0;
  op(transb, transa, ColumnCount(B, transb), RowCount(A, transa),
     ColumnCount(A, transa), static_cast<typename TGemmOp::TDatatype>(alpha),
     reinterpret_cast<typename TGemmOp::TDatatype *>(
         static_cast<char *>(B->data) + B->byte_offset),
     ColumnStride(B),
     reinterpret_cast<typename TGemmOp::TDatatype *>(
         static_cast<char *>(A->data) + A->byte_offset),
     ColumnStride(A), static_cast<typename TGemmOp::TDatatype>(beta),
     reinterpret_cast<typename TGemmOp::TDatatype *>(
         static_cast<char *>(C->data) + C->byte_offset),
     ColumnStride(C));
  /* std::cout << "[GEMM] Check A " << RowCount(A, false) << " " << ColumnCount(A, false) << std::endl; */
  /* CheckForNan(A); */
  /* std::cout << "[GEMM] Check B " << RowCount(B, false) << " " << ColumnCount(B, false) << std::endl; */
  /* CheckForNan(B); */
  /* std::cout << "[GEMM] Check C " << RowCount(C, false) << " " << ColumnCount(C, false) << std::endl; */
  /* CheckForNan(C); */
}

// Call a column major blas.  Note that data is stored in tvm as row
// major, so this we switch the arguments.
template <typename TGemmOp>
inline void CallGemv(TVMArgs args, TVMRetValue *ret, TGemmOp op) {
  DLTensor *A = args[0];
  DLTensor *X = args[1];
  DLTensor *Y = args[2];
  bool transa = args[3];
  int bit_depth = sizeof(typename TGemmOp::TDatatype) * 8;
  CHECK_EQ(A->ndim, 2);
  CHECK_EQ(X->ndim, 1);
  CHECK_EQ(Y->ndim, 1);

  CHECK_EQ(ElementStride(A), 1);
  CHECK_EQ(ElementStride(X), 1);
  CHECK_EQ(ElementStride(Y), 1);

  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;

  CHECK(TypeMatch(X->dtype, kDLFloat, bit_depth));
  CHECK(TypeMatch(Y->dtype, kDLFloat, bit_depth));
  double alpha = args.size() > 4 ? args[4] : 1.0;
  double beta = args.size() > 5 ? args[5] : 0.0;

  op(transa, ColumnCount(A, transa), RowCount(A, transa),
     static_cast<typename TGemmOp::TDatatype>(alpha),
     reinterpret_cast<typename TGemmOp::TDatatype *>(
         static_cast<char *>(A->data) + A->byte_offset),
     ColumnStride(A),
     reinterpret_cast<typename TGemmOp::TDatatype *>(
         static_cast<char *>(X->data) + X->byte_offset),
     ElementStride(X),
     static_cast<typename TGemmOp::TDatatype>(beta),
     reinterpret_cast<typename TGemmOp::TDatatype *>(
         static_cast<char *>(Y->data) + Y->byte_offset),
     ElementStride(Y));
}

inline int ColumnStride3D(DLTensor *tensor) {
  // If the tensor itself is transposed then it will have strides
  // backward from what we expect.  Regardless, the max of the strides
  // (the other stride is 1) is the column stride.
  if (tensor->strides) {
    return std::max(tensor->strides[1], tensor->strides[2]);
  } else {
    return tensor->shape[2];
  }
}
inline int ElementStride3D(DLTensor *tensor) {
  if (tensor->strides) {
    return std::min(tensor->strides[1], tensor->strides[2]);
  } else {
    return 1;
  }
}
// Reversed strides indicates an in-place transpose operation.
inline bool IsInPlaceTransposed3D(DLTensor *tensor) {
  return tensor->strides && (tensor->strides[2] > tensor->strides[1]);
}
inline int BatchCount3D(DLTensor *tensor) { return tensor->shape[0]; }
inline int RowCount3D(DLTensor *tensor, bool trans) {
  return tensor->shape[trans ? 2 : 1];
}
inline int ColumnCount3D(DLTensor *tensor, bool trans) {
  return tensor->shape[trans ? 1 : 2];
}
template <typename TBatchGemmOp>
inline void CallBatchGemm(TVMArgs args, TVMRetValue *ret, TBatchGemmOp op) {
  using DType = typename TBatchGemmOp::TDatatype;
  DLTensor *A = args[0];
  DLTensor *B = args[1];
  DLTensor *C = args[2];
  bool transa = args[3];
  bool transb = args[4];
  int bit_depth = sizeof(DType) * 8;
  CHECK_EQ(A->ndim, 3);
  CHECK_EQ(B->ndim, 3);
  CHECK_EQ(C->ndim, 3);
  int batch_size = BatchCount3D(A);
  CHECK_EQ(BatchCount3D(B), batch_size);
  CHECK_EQ(BatchCount3D(C), batch_size);
  CHECK_EQ(ElementStride(A), 1);
  CHECK_EQ(ElementStride(B), 1);
  CHECK_EQ(ElementStride(C), 1);
  // C can never be transposed.
  CHECK(!IsInPlaceTransposed3D(C));
  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed3D(A) ? !transa : transa;
  transb = IsInPlaceTransposed3D(B) ? !transb : transb;
  CHECK(TypeMatch(B->dtype, kDLFloat, bit_depth));
  CHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));
  double alpha = args.size() > 5 ? args[5] : 1.0;
  double beta = args.size() > 6 ? args[6] : 0.0;
  const int A_size = A->shape[1] * A->shape[2];
  const int B_size = B->shape[1] * B->shape[2];
  const int C_size = C->shape[1] * C->shape[2];
  DType *A_data = reinterpret_cast<typename TBatchGemmOp::TDatatype *>(
      static_cast<char *>(A->data) + A->byte_offset);
  DType *B_data = reinterpret_cast<typename TBatchGemmOp::TDatatype *>(
      static_cast<char *>(B->data) + B->byte_offset);
  DType *C_data = reinterpret_cast<typename TBatchGemmOp::TDatatype *>(
      static_cast<char *>(C->data) + C->byte_offset);
  op(batch_size, transb, transa, ColumnCount3D(B, transb),
     RowCount3D(A, transa), ColumnCount3D(A, transa),
     static_cast<typename TBatchGemmOp::TDatatype>(alpha),
     B_data, B_size, ColumnStride3D(B), A_data, A_size, ColumnStride3D(A),
     static_cast<typename TBatchGemmOp::TDatatype>(beta), C_data, C_size, ColumnStride3D(C));
}

}  // namespace contrib
}  // namespace tvm
