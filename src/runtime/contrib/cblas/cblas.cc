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
 * \file Use external cblas library call.
 */
#include <dmlc/logging.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>

#include "gemm_common.h"

extern "C" {
#if USE_MKL_BLAS == 1
#include <mkl.h>
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#if USE_DNNL == 1
#include <dnnl.h>
#endif
}

namespace tvm {
namespace contrib {

using namespace runtime;

// inline CBLAS_TRANSPOSE BooleanToTranspose(bool trans) { return trans ? CblasTrans : CblasNoTrans;
// }
#define BooleanToTranspose(trans) (trans ? CblasTrans : CblasNoTrans)

inline char BooleanToTransposeChar(bool trans) { return trans ? 'T' : 'N'; }

struct CblasSgemmOp {
  typedef float TDatatype;
  void operator()(bool ta, bool tb, int M, int N, int K, float alpha, float* A, int lda, float* B,
                  int ldb, float beta, float* C, int ldc) {
    // std::cout << "[BLAS] Calling multi threaded blas" << std::endl;
#if USE_DNNL == 1
    dnnl_sgemm(BooleanToTransposeChar(tb), BooleanToTransposeChar(ta), N, M, K, alpha, B, ldb, A,
               lda, beta, C, ldc);
#else
    // std::cout << BooleanToTranspose(ta) << " " << BooleanToTranspose(tb) << " " << CblasNoTrans
    // << " " << CblasTrans << std::endl;
    cblas_sgemm(CblasColMajor, BooleanToTranspose(ta), BooleanToTranspose(tb), M, N, K, alpha, A,
                lda, B, ldb, beta, C, ldc);
#endif
    // std::cout << "[BLAS] Exiting multi threaded blas" << std::endl;
  }
};

struct CblasSgemmSingleThreadedOp {
  typedef float TDatatype;
  void operator()(bool ta, bool tb, int M, int N, int K, float alpha, float* A, int lda, float* B,
                  int ldb, float beta, float* C, int ldc) {
    // CHECK(false) << "Do not support single threaded BLAS calls with MKL";
#if USE_MKL_BLAS == 1
    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);

    std::cout << "[CBLASGEMM]" << std::endl;
    std::cout << "[CBLASGEMM]   " << ta << " " << tb << std::endl;
    std::cout << "[CBLASGEMM]   " << M << " " << N << " " << K << std::endl;
    std::cout << "[CBLASGEMM]   " << A << " " << B << " " << C << std::endl;
    std::cout << "[CBLASGEMM]   " << lda << " " << ldb << " " << ldc << std::endl;

    cblas_sgemm(CblasColMajor, BooleanToTranspose(ta), BooleanToTranspose(tb), M, N, K, alpha, A,
                lda, B, ldb, beta, C, ldc);
    // CHECK(false) << "Do not support single threaded BLAS calls with MKL";
#else
    // std::cout << "[BLAS] Calling single threaded blas" << std::endl;
    cblas_sgemm(CblasColMajor, BooleanToTranspose(ta), BooleanToTranspose(tb), M, N, K, alpha, A,
                lda, B, ldb, beta, C, ldc);
#endif
  }
};

struct CblasSgemvSingleThreadedOp {
  typedef float TDatatype;
  void operator()(bool ta, int M, int N, float alpha, float* A, int lda, float* X, int incX,
                  float beta, float* Y, int incY) {
    // CHECK(false) << "Do not support single threaded BLAS calls with MKL";
    CBLAS_TRANSPOSE transa = BooleanToTranspose(ta);
    CBLAS_ORDER order = CblasRowMajor;
    // std::cout << "[BLAS]   Args: " << order << " " << transa << " " << CblasTrans << " " <<
    // CblasNoTrans << std::endl;
#if USE_MKL_BLAS == 1
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, A, lda, X, incX, beta, Y, incY);
    // CHECK(false) << "Do not support single threaded BLAS calls with MKL";
#else
    // std::cout << "[BLAS] Calling single threaded blas" << std::endl;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, A, lda, X, incX, beta, Y, incY);
#endif
  }
};

struct CblasDgemmOp {
  typedef double TDatatype;
  void operator()(bool ta, bool tb, int M, int N, int K, double alpha, double* A, int lda,
                  double* B, int ldb, double beta, double* C, int ldc) {
    cblas_dgemm(CblasColMajor, BooleanToTranspose(ta), BooleanToTranspose(tb), M, N, K, alpha, A,
                lda, B, ldb, beta, C, ldc);
  }
};

struct CblasDgemmSingleThreadedOp {
  typedef double TDatatype;
  void operator()(bool ta, bool tb, int M, int N, int K, double alpha, double* A, int lda,
                  double* B, int ldb, double beta, double* C, int ldc) {
#if USE_MKL_BLAS == 1
    CHECK(false) << "Do not support single threaded BLAS calls with MKL";
#else
    cblas_dgemm(CblasColMajor, BooleanToTranspose(ta), BooleanToTranspose(tb), M, N, K, alpha, A,
                lda, B, ldb, beta, C, ldc);
#endif
  }
};

struct CblasDgemvSingleThreadedOp {
  typedef double TDatatype;
  void operator()(bool ta, int M, int N, double alpha, double* A, int lda, double* X, int incX,
                  double beta, double* Y, int incY) {
    CHECK(false) << "Do not support single threaded BLAS calls with MKL";
#if USE_MKL_BLAS == 1
    CHECK(false) << "Do not support single threaded BLAS calls with MKL";
#else
    std::cout << "[BLAS] Calling single threaded blas" << std::endl;
    cblas_dgemv(CblasColMajor, BooleanToTranspose(ta), M, N, alpha, A, lda, X, incX, beta, Y, incY);
#endif
  }
};

struct CblasSgemmBatchOp {
  typedef float TDatatype;
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, float alpha, float* A,
                  int a_stride, int lda, float* B, int b_stride, int ldb, float beta, float* C,
                  int c_stride, int ldc) {
    CBLAS_TRANSPOSE trans_a = BooleanToTranspose(ta);
    CBLAS_TRANSPOSE trans_b = BooleanToTranspose(tb);
#if USE_MKL_BLAS == 1
    std::vector<const float*> A_array(batch_size);
    std::vector<const float*> B_array(batch_size);
    std::vector<float*> C_array(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      A_array[i] = A + i * a_stride;
      B_array[i] = B + i * b_stride;
      C_array[i] = C + i * c_stride;
    }
    cblas_sgemm_batch(CblasColMajor, &trans_a, &trans_b, &M, &N, &K, &alpha, A_array.data(), &lda,
                      B_array.data(), &ldb, &beta, C_array.data(), &ldc, 1, &batch_size);
#else
    for (int i = 0; i < batch_size; ++i) {
      cblas_sgemm(CblasColMajor, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      A += a_stride;
      B += b_stride;
      C += c_stride;
    }
#endif
  }
};

struct CblasSgemmBatchIterativeOp {
  typedef float TDatatype;
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, float alpha, float* A,
                  int a_stride, int lda, float* B, int b_stride, int ldb, float beta, float* C,
                  int c_stride, int ldc) {
    CBLAS_TRANSPOSE trans_a = BooleanToTranspose(ta);
    CBLAS_TRANSPOSE trans_b = BooleanToTranspose(tb);
    for (int i = 0; i < batch_size; ++i) {
      cblas_sgemm(CblasColMajor, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      A += a_stride;
      B += b_stride;
      C += c_stride;
    }
  }
};

struct CblasDgemmBatchOp {
  typedef double TDatatype;
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, double alpha, double* A,
                  int a_stride, int lda, double* B, int b_stride, int ldb, double beta, double* C,
                  int c_stride, int ldc) {
    CBLAS_TRANSPOSE trans_a = BooleanToTranspose(ta);
    CBLAS_TRANSPOSE trans_b = BooleanToTranspose(tb);
#if USE_MKL_BLAS == 1
    std::vector<const double*> A_array(batch_size);
    std::vector<const double*> B_array(batch_size);
    std::vector<double*> C_array(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      A_array[i] = A + i * a_stride;
      B_array[i] = B + i * b_stride;
      C_array[i] = C + i * c_stride;
    }
    cblas_dgemm_batch(CblasColMajor, &trans_a, &trans_b, &M, &N, &K, &alpha, A_array.data(), &lda,
                      B_array.data(), &ldb, &beta, C_array.data(), &ldc, 1, &batch_size);
#else
    for (int i = 0; i < batch_size; ++i) {
      cblas_dgemm(CblasColMajor, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      A += a_stride;
      B += b_stride;
      C += c_stride;
    }
#endif
  }
};

struct CblasDgemmBatchIterativeOp {
  typedef double TDatatype;
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, double alpha, double* A,
                  int a_stride, int lda, double* B, int b_stride, int ldb, double beta, double* C,
                  int c_stride, int ldc) {
    CBLAS_TRANSPOSE trans_a = BooleanToTranspose(ta);
    CBLAS_TRANSPOSE trans_b = BooleanToTranspose(tb);
    for (int i = 0; i < batch_size; ++i) {
      cblas_dgemm(CblasColMajor, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      A += a_stride;
      B += b_stride;
      C += c_stride;
    }
  }
};

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.cblas.matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  CHECK(TypeMatch(A->dtype, kDLFloat, 32) || TypeMatch(A->dtype, kDLFloat, 64));

  if (TypeMatch(A->dtype, kDLFloat, 32)) {
    CallGemm(args, ret, CblasSgemmOp());
  } else {
    CallGemm(args, ret, CblasDgemmOp());
  }
});

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.cblas.matmul_no_thread")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* A = args[0];
      CHECK(TypeMatch(A->dtype, kDLFloat, 32) || TypeMatch(A->dtype, kDLFloat, 64));

      if (TypeMatch(A->dtype, kDLFloat, 32)) {
        // std::cout << "CBLAS called" << std::endl;
        CallGemm(args, ret, CblasSgemmSingleThreadedOp());
      } else {
        CallGemm(args, ret, CblasDgemmSingleThreadedOp());
      }
    });

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.cblas.matvecmul_no_thread")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* A = args[0];
      CHECK(TypeMatch(A->dtype, kDLFloat, 32) || TypeMatch(A->dtype, kDLFloat, 64));

      if (TypeMatch(A->dtype, kDLFloat, 32)) {
        // std::cout << "CBLAS called" << std::endl;
        CallGemv(args, ret, CblasSgemvSingleThreadedOp());
      } else {
        CallGemv(args, ret, CblasDgemvSingleThreadedOp());
      }
    });

TVM_REGISTER_GLOBAL("tvm.contrib.cblas.batch_matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  CHECK(TypeMatch(A->dtype, kDLFloat, 32) || TypeMatch(A->dtype, kDLFloat, 64));
  if (TypeMatch(A->dtype, kDLFloat, 32)) {
    CallBatchGemm(args, ret, CblasSgemmBatchOp());
  } else {
    CallBatchGemm(args, ret, CblasDgemmBatchOp());
  }
});

TVM_REGISTER_GLOBAL("tvm.contrib.cblas.batch_matmul_iterative")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* A = args[0];
      CHECK(TypeMatch(A->dtype, kDLFloat, 32) || TypeMatch(A->dtype, kDLFloat, 64));
      if (TypeMatch(A->dtype, kDLFloat, 32)) {
        CallBatchGemm(args, ret, CblasSgemmBatchIterativeOp());
      } else {
        CallBatchGemm(args, ret, CblasDgemmBatchIterativeOp());
      }
    });
}  // namespace contrib
}  // namespace tvm
