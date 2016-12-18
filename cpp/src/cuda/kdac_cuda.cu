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
#include "include/gpu_util.h"
#include "../../include/gpu_util.h"

namespace Nice {

template <typename T>
__global__ void GPUGenAMatricesKernel(const T *x_matrix_d,
                                      const int n,
                                      const int d,
                                      const float *alpha_d,
                                      const float *beta_d,
                                      T *a_matrices_d,
                                      T *all_delta_ijs_d,
                                      cublasStatus_t *return_status) {

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
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      *(return_status + IDXR(i, j, n)) = status;
      return;
    }

//  Each thread (i, j) generates a matrix Aij
    status =
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    d, d, 1,
                    alpha_d,
                    delta_ij, d,
                    delta_ij, 1,
                    beta_d,
                    a_ij_matrix, d);
    cublasDestroy(handle);
    *(return_status + IDXR(i, j, n)) = status;
  }
}

template <typename T>
__global__ void GPUGenPhiCoeffKernel(const T *w_l_d,
                                     const T *gradient_d,
                                     const T *a_matrices_d,
                                     const int n,
                                     const int d,
                                     const float *alpha_d,
                                     const float *beta_d,
                                     const int *incx_d,
                                     const int *incy_d,
                                     T *waw_matrix_d,
                                     T *waf_matrix_d,
                                     T *faf_matrix_d,
                                     T *temp_d,
                                     cublasStatus_t *return_status) {

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i < n) && (j < n)) {
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      *(return_status + IDXR(i, j, n)) = status;
      return;
    }
    const T *a_ij_matrix = a_matrices_d + IDXR(i, j, n) * (d * d);
    T *temp_ij = temp_d + IDXR(i, j, n) * d;

    // Calculate waw
//    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//                1, d, d,
//                &alpha,
//                w_l_d, 1,
//                a_ij_matrix, d,
//                &beta,
//                temp_ij, 1);

    status = cublasSgemv(handle, CUBLAS_OP_N,
                         d, d,
                         alpha_d,
                         a_ij_matrix, d,
                         w_l_d, *incx_d,
                         beta_d,
                         temp_ij, *incy_d);

    cublasSdot(handle, d,
               temp_ij, *incx_d,
               w_l_d, *incy_d,
               &waw_matrix_d[IDXC(i, j, n)]);

    // Calculate waf
    // temp_ij is the intermediate result of w_l.transpose() * a_matrix_ij
    // So here we are only going to use cublasSdot and reuse temp_ij
    cublasSdot(handle, d,
               temp_ij, *incx_d,
               gradient_d, *incy_d,
               &waf_matrix_d[IDXC(i, j, n)]);

    // Calculate faf
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                1, d, d,
                alpha_d,
                gradient_d, 1,
                a_ij_matrix, d,
                beta_d,
                temp_ij, 1);

    cublasSdot(handle, d,
               temp_ij, *incx_d,
               gradient_d, *incy_d,
               &faf_matrix_d[IDXC(i, j, n)]);

    cublasDestroy(handle);
    *(return_status + IDXR(i, j, n)) = status;
  }
}

template<typename T>
void GPUGenAMatrices(const T *x_matrix_d,
                     const int n,
                     const int d,
                     T *a_matrices_d) {

  // Setup cublas params alpha, beta, incx and incy
  CUBLASParams params = {1.0, // alpha
                           0.0, // beta
                           1,   // incx
                           1    // incy
                          };
  CUBLASParams *params_d;
  CUDA_CALL(cudaMalloc((void**)&params_d, sizeof(CUBLASParams)));
  CUDA_CALL(cudaMemcpy(params_d, &params, sizeof(CUBLASParams),
                       cudaMemcpyHostToDevice));

  // Intermediate data: n delta_ijs
  T *all_delta_ijs_d;
  CUDA_CALL(cudaMalloc(&all_delta_ijs_d, n * n * d * sizeof(T)));

  // n * n cublas return status from calling inside a kernel
  cublasStatus_t *statuses_d;
  cublasStatus_t *statuses = new cublasStatus_t[n*n];
  CUDA_CALL(cudaMalloc((void**)&statuses_d, sizeof(cublasStatus_t)*n*n));

  int block_size = 16;
  dim3 dim_block(block_size, block_size);
  dim3 dim_grid( (n-1) / block_size + 1, (n-1) / block_size + 1);
  GPUGenAMatricesKernel<<<dim_grid, dim_block>>>(x_matrix_d,
                                                 n,
                                                 d,
                                                 &(params_d->alpha),
                                                 &(params_d->beta),
                                                 a_matrices_d,
                                                 all_delta_ijs_d,
                                                 statuses_d);

  // Check if error happens in kernel launch
  CUDA_CALL(cudaGetLastError());

  CUDA_CALL(cudaMemcpy(statuses, statuses_d, sizeof(cublasStatus_t)*n*n,
                       cudaMemcpyDeviceToHost));
  for (int i = 0; i < n*n; i++)
    CUBLAS_CALL(statuses[i]);

  // Free parameters, intermediate delta and parameters
  CUDA_CALL(cudaFree(all_delta_ijs_d));
  CUDA_CALL(cudaFree(statuses_d));
  CUDA_CALL(cudaFree(params_d));
}

// Explicit Instantiation
template
void GPUGenAMatrices<float>(const float*,
                            const int,
                            const int,
                            float*);

template <typename T>
void GPUGenPhiCoeff(const T *w_l_d,
                    const T *gradient_d,
                    const T *a_matrices_d,
                    const int n,
                    const int d,
                    T *temp_d,
                    T *waw_matrix_d,
                    T *waf_matrix_d,
                    T *faf_matrix_d) {
  // Setup cublas params alpha, beta, incx and incy
  CUBLASParams params = {1.0, // alpha
                         0.0, // beta
                         1,   // incx
                         1   // incy
  };
  CUBLASParams *params_d;
  CUDA_CALL(cudaMalloc((void**)&params_d, sizeof(CUBLASParams)));
  CUDA_CALL(cudaMemcpy(params_d, &params, sizeof(CUBLASParams),
                       cudaMemcpyHostToDevice));

  // n * n cublas return status from calling inside a kernel
  cublasStatus_t *statuses_d;
  cublasStatus_t *statuses = new cublasStatus_t[n*n];
  CUDA_CALL(cudaMalloc((void**)&statuses_d, sizeof(cublasStatus_t)*n*n));


  int block_size = 16;
  dim3 dim_block(block_size, block_size);
  dim3 dim_grid( (n-1) / block_size + 1, (n-1) / block_size + 1);
  GPUGenPhiCoeffKernel<<<dim_grid, dim_block>>>(w_l_d,
                                                gradient_d,
                                                a_matrices_d,
                                                n,
                                                d,
                                                &(params_d->alpha),
                                                &(params_d->beta),
                                                &(params_d->incx),
                                                &(params_d->incy),
                                                waw_matrix_d,
                                                waf_matrix_d,
                                                faf_matrix_d,
                                                temp_d,
                                                statuses_d);

  // Check if error happens in kernel launch
  CUDA_CALL(cudaGetLastError());

  CUDA_CALL(cudaMemcpy(statuses, statuses_d, sizeof(cublasStatus_t)*n*n,
                       cudaMemcpyDeviceToHost));
  for (int i = 0; i < n*n; i++)
    CUBLAS_CALL(statuses[i]);

  // Free parameters, intermediate delta and parameters
  CUDA_CALL(cudaFree(statuses_d));
  CUDA_CALL(cudaFree(params_d));
}

template
void GPUGenPhiCoeff<float>(const float*,
                           const float*,
                           const float*,
                           const int n,
                           const int d,
                           float*,
                           float*,
                           float*,
                           float*);
}