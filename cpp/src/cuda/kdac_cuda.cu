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

#include "include/kdac_cuda.h"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"
#include "include/util.h"
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <unistd.h>
#include <stdexcept>
#include <ctime>

namespace Nice {

template <typename T>
T* CUDAMallocAndCpy(const Matrix<T> &mat) {
  GpuUtil<T> *util = GpuUtil<T>::GetInstance();
  int n = mat.cols() * mat.rows();
  const T *h_mat = &mat(0);
  T *d_mat;
  util -> SetupMem(&d_mat, h_mat, n);
  std::cout << "allocating " << n * sizeof(T) << " bytes." << std::endl;
  return d_mat;
}
// Template explicit instantiation
template
float* CUDAMallocAndCpy<float>(const Matrix<float> &mat);
template
double* CUDAMallocAndCpy<double>(const Matrix<double> &mat);


template <typename T>
T* CUDAMallocAndCpy(const Vector <T> &vec) {
  GpuUtil<T> *util = GpuUtil<T>::GetInstance();
  int n = vec.size();
  const T *h_vec = &vec(0);
  T *d_vec;
  util -> SetupMem(&d_vec, h_vec, n);
  std::cout << "allocating " << n * sizeof(T) << " bytes." << std::endl;
  return d_vec;
}

template
float* CUDAMallocAndCpy<float>(const Vector<float> &vec);
template
double* CUDAMallocAndCpy<double>(const Vector<double> &vec);

// Position for Column-Major index
#define IDXC(i,j,ld) (((j)*(ld))+(i))
// Position for Row-Major index
#define IDXR(i,j,ld) (((i)*(ld))+(j))

template <typename T>
__global__ void GPUGenAMatricesKernel
    (T *x_matrix_d, T *a_matrices_d, T *all_delta_ijs_d, int n, int d) {

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  // This is to index an n x n matrix where each cell is a
  // d x d matrix. No matter what orientation (row or column) the
  // d x d matrix is, to find the starting location of the (i, j)
  // matrix, we just need to use the following to do so
  if (i < n && j < n) {
    T *a_ij_matrix = a_matrices_d + IDXR(i, j, n) * (d * d);
    T *delta_ij = all_delta_ijs_d + IDXR(i, j, n) * d;

    // x_matrix_d is column major
    for (int k = 0; k < d; k++) {
      delta_ij[k] = x_matrix_d[IDXC(i, k, n)] - x_matrix_d[IDXC(j, k, n)];
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0;
    const float beta = 0.0;

//  Each thread (i, j) generates a matrix Aij
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                d, d, 1,
                &alpha, delta_ij, d,
                delta_ij, 1,
                &beta, a_ij_matrix, d);
  }

////  cublasStatus_t cublasSgemm(cublasHandle_t handle,
////                             cublasOperation_t transa,
////                             cublasOperation_t transb,
////                             int m, int n, int k,
////                             const float *alpha, const float *A, int lda,
////                             const float *B, int ldb,
////                             const float *beta, float *C, int ldc)
}

// Explicit Instantiation
template
__global__ void GPUGenAMatricesKernel<float>(float *x_matrix_d,
                            float *a_matrices_d,
                            float *all_delta_ijs_d,
                            int n,
                            int d);



template<typename T>
void GPUGenAMatrices(T *x_matrix_d, T *a_matrices_d, int n, int d) {

  GpuUtil<T> *gpu_util = GpuUtil<T>::GetInstance();
  int size_delta_ijs = n * n * d;
//  T *all_delta_ijs = new T[size_delta_ijs];
  T *all_delta_ijs_d;
  gpu_util -> SetupMem(&all_delta_ijs_d, nullptr, size_delta_ijs, false);
  int block_size = 16;
  dim3 dim_block(block_size, block_size);
  dim3 dim_grid( (n-1) / block_size + 1, (n-1) / block_size + 1);
  GPUGenAMatricesKernel<<<dim_grid, dim_block>>>(x_matrix_d, a_matrices_d,
      all_delta_ijs_d, n, d);
//  gpu_util->SyncMem(all_delta_ijs_d, all_delta_ijs, size_delta_ijs);
//  for (int i = 0; i < n; i++) {
//    for (int j = 0; j < n; j++) {
//      T *delta_ij = all_delta_ijs + IDXR(i, j, n) * d;
//      std::cout << "delta (" << (i + 1) << ", " << (j + 1) << "): "
//                << std::endl;
//      util::PrintMatrix<T>(delta_ij, d, 1, false);
//    }
//  }
}

// Explicit Instantiation
template
void GPUGenAMatrices<float>(float *x_matrix_d,
                            float *a_matrices_d,
                            int n,
                            int d);

// Cannot instantiate it to double if I am using cublasSgemm
// Only cublasDgemm is for double
//template
//void GPUGenAMatrices<double>(double *x_matrix,
//                             double *a_matrices,
//                             double *delta_ijs,
//                             int n,
//                             int d);
//template <typename T>
//__global__ void GPUGenPhiCoeffKernel(T *x_matrix,
//                                     T *a_matrices,
//                                     T *delta_x_ijs,
//                                     T *waw_matrix,
//                                     T *waf_matrix,
//                                     T *faf_matrix,
//                                     T *w_l,
//                                     T *gradient,
//                                     int n,
//                                     int d) {
//  int i = blockIdx.y * blockDim.y + threadIdx.y;
//  int j = blockIdx.x * blockDim.x + threadIdx.x;
//  // This is to index an n x n matrix where each cell is a
//  // d x d matrix. No matter what orientation (row or column) the
//  // d x d matrix is, to find the starting location of the (i, j)
//  // matrix, we just need to use the following to do so
//  T *a_ij_matrix = a_matrices + (i * n + j) * d * d;
//  T *delta_x_ij = delta_x_ijs + (i * n + j) * d;
//  GenAMatrix(x_matrix, a_matrices, a_ij_matrix, delta_x_ij, i, j, n, d);
//}
//
//template<typename T>
//void GPUGenPhiCoeff(T *x_matrix, T *a_matrices, T *waw_matrix, T *waf_matrix,
//                    T *faf_matrix, T *w_l, T *gradient, int n, int d) {
//  std::cout << "in GPUGenPhiCoeff" << std::endl;
//
//  int block_size = 16;
//  dim3 dim_block(block_size, block_size);
//  dim3 dim_grid( (n-1) / block_size + 1, (n-1) / block_size + 1);
//  GPUGenPhiCoeffKernel<<<dim_grid, dim_block>>>(x_matrix, waw_matrix,
//      waf_matrix, faf_matrix, w_l, gradient, n d);
//}
//
//template
//void GPUGenPhiCoeff<float>(float *x_matrix,
//                           float *a_matrices,
//                           float *delta_x_ijs,
//                           float *waw_matrix,
//                           float *waf_matrix,
//                           float *faf_matrix,
//                           float *w_l,
//                           float *gradient,
//                           int n,
//                           int d);
//template
//void GPUGenPhiCoeff<double>(double *x_matrix,
//                            double *a_matrices,
//                            double *delta_x_ijs,
//                            double *waw_matrix,
//                            double *waf_matrix,
//                            double *faf_matrix,
//                            double *w_l,
//                            double *gradient,
//                            int n,
//                            int d);
}