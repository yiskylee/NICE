// The MIT License (MIT)
//
// Copyright (c) 2016 Northeastern University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef CPP_INCLUDE_GPU_OPERATIONS_H_
#define CPP_INCLUDE_GPU_OPERATIONS_H_

#include<cuda_runtime_api.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include <cublas_v2.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"

namespace Nice {

// Abstract class of common matrix operation interface
template <typename T>
class GpuOperations {
 public:
  static Matrix<T> Multiply(const Matrix<T> &a, const T &scalar);
  static Matrix<T> Multiply(const Matrix<T> &a, const Matrix<T> &b);
  static Matrix<T> Add(const Matrix<T> &a, const T &scalar);
  static Matrix<T> Add(const Matrix<T> &a, const Matrix<T> &b);
  static Matrix<T> Subtract(const Matrix<T> &a, const T &scalar);
  static Matrix<T> Subtract(const Matrix<T> &a, const Matrix<T> &b);
  static Matrix<T> Inverse(const Matrix<T> &a);
  static Matrix<T> Norm(const int &p = 2, const int &axis = 0);
  static T Determinant(const Matrix<T> &a);
  static T Rank(const Matrix<T> &a);
  static T FrobeniusNorm(const Matrix<T> &a);
  static T Trace(const Matrix<T> &a);
  static T DotProduct(const Vector<T> &a, const Vector<T> &b);
  static Matrix<T> OuterProduct(const Vector<T> &a, const Vector<T> &b);
};

template <typename T>
Matrix<T> GpuOperations<T>::Multiply(const Matrix<T> &a, const Matrix<T> &b) {
  if (a.cols() == b.rows()) {
  int m = a.rows();
  int n = b.cols();
  int k = a.cols();
  int lda = m;
  int ldb = k;
  int ldc = m;

  float alpha = 1.0;
  float beta =  0.0;

  const T * h_a = &a(0);
  const T * h_b = &b(0);
  Matrix<T> h_c(m, n);

  T * d_a;  gpuErrchk(cudaMalloc(&d_a, m * k * sizeof(T)));
  T * d_b;  gpuErrchk(cudaMalloc(&d_b, k * m * sizeof(T)));
  T * d_c;  gpuErrchk(cudaMalloc(&d_c, m * n * sizeof(T)));

  gpuErrchk(cudaMemcpy(d_a, h_a, m * k * sizeof(T), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_b, h_b, k * n * sizeof(T), cudaMemcpyHostToDevice));


  cublasHandle_t  handle;
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                  m, n, k,
                  &alpha,
                  d_a, lda,
                  d_b, ldb,
                  &beta,
                  d_c, ldc);
  cudaDeviceSynchronize();
  gpuErrchk(cudaMemcpy(&h_c(0, 0), d_c, m * n * sizeof(T),
                       cudaMemcpyHostToDevice));

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  cublasDestroy(handle);
  return h_c;
  } else {
    std::cerr << "Matricies in gpu matrix multiply's sizes aren't compatible"
             << std::endl;
    exit(1);
  }
}
template class GpuOperations<float>;
}  // namespace Nice

#endif  // CPP_INCLUDE_GPU_OPERATIONS_H_

