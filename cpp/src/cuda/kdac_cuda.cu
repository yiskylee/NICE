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
#include "include/kdac.h"
#include "include/gpu_util.h"

// Hack to cope with Clion
#include "../../include/gpu_util.h"
#include "../../include/kdac_cuda.h"
#include "../../include/kdac.h"
#include "../../../../../../../../usr/local/cuda/include/driver_types.h"

namespace Nice {

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

bool isPow2(unsigned int x) {
  return ((x & (x - 1)) == 0);
}

template <typename T>
__global__ void GPUGenDeltaKernel(const T *x_matrix_d,
                                      const int n,
                                      const int d,
                                      T *all_delta_ijs_d) {

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  // This is to index an n x n matrix where each cell is a
  // d x d matrix. No matter what orientation (row or column) the
  // d x d matrix is, to find the starting location of the (i, j)
  // matrix, we just need to use the following to do so
  if (i < n && j < n) {
    T *delta_ij = all_delta_ijs_d + IDXR(i, j, n) * d;
    // x_matrix_d is column major
    for (int k = 0; k < d; k++)
      delta_ij[k] = x_matrix_d[IDXC(i, k, n)] - x_matrix_d[IDXC(j, k, n)];
  }
}

template <typename T>
__global__ void GPUGenAMatricesKernel(const T *x_matrix_d,
                                      const int n,
                                      const int d,
                                      T *a_matrices_d) {
  T *delta_ij = SharedMemory<T>();
  int tx = threadIdx.x;
  int i = blockIdx.y;
  int j = blockIdx.x;

  if (tx < d) {
    T *a_ij = a_matrices_d + IDXR(i, j, n) * (d * d);
    delta_ij[tx] = x_matrix_d[IDXC(i, tx, n)] - x_matrix_d[IDXC(j, tx, n)];
    __syncthreads();
    // thread tx calculates a whole row tx of the output matrix a_ij
    for (int col = 0; col < d; col++)
      a_ij[IDXC(tx, col, d)] = delta_ij[col] * delta_ij[tx];
  }
}

template <typename T>
__global__ void GPUGenPhiCoeffKernel(const T *w_l_d,
                                     const T *gradient_d,
                                     const T *a_matrices_d,
                                     const int n,
                                     const int d,
                                     T *waw_matrix_d,
                                     T *waf_matrix_d,
                                     T *faf_matrix_d) {
  T *vec_s = SharedMemory<T>();
  T *waw_s = (T*)vec_s;
  T *waf_s = (T*)&vec_s[blockDim.x];
  T *faf_s = (T*)&vec_s[2*blockDim.x];
  T *w_s = (T*)&vec_s[3*blockDim.x];
  T *grad_s = (T*)&vec_s[4*blockDim.x];

  int i = blockIdx.y;
  int j = blockIdx.x;
  int tx = threadIdx.x;
  const T *a_ij = a_matrices_d + IDXR(i, j, n) * (d * d);

  waw_s[tx] = 0.0;
  waf_s[tx] = 0.0;
  faf_s[tx] = 0.0;

  if (tx < d) {
    w_s[tx] = w_l_d[tx];
    grad_s[tx] = gradient_d[tx];
  }
  __syncthreads();


  if (tx < d) {
    // Each tx takes care of one row of matrix in order to have a
    // coalesced access pattern
    // Each time it aggreates a column
    for (int col = 0; col < d; col++) {
      waw_s[tx] += a_ij[IDXC(tx, col, d)] * w_s[col];
      waf_s[tx] += a_ij[IDXC(tx, col, d)] * grad_s[col];
//      faf_s[tx] += a_ij[IDXC(tx, col, d)] * gradient_d[col];
    }
    faf_s[tx] = waf_s[tx];

    // This is the dot product
    waw_s[tx] = waw_s[tx] * w_s[tx];
    waf_s[tx] = waf_s[tx] * w_s[tx];
    faf_s[tx] = faf_s[tx] * grad_s[tx];
  }
  __syncthreads();

  // Reduction for dot product
  for (unsigned int s = blockDim.x / 2; s > 0; s>>=1) {
    if (tx < s) {
      waw_s[tx] += waw_s[tx + s];
      waf_s[tx] += waf_s[tx + s];
      faf_s[tx] += faf_s[tx + s];
    }
    __syncthreads();
  }
//    if (tx < 8) {
//      vec_s[tx] += vec_s[tx + 8];
//      vec_s[tx] += vec_s[tx + 4];
//      vec_s[tx] += vec_s[tx + 2];
//      vec_s[tx] += vec_s[tx + 1];
//    }
//    __syncthreads();

    // Transposed access for better access pattern as waw_s matrix is column-major
  if (tx == 0) {
    waw_matrix_d[IDXC(j, i, n)] = waw_s[tx];
    waf_matrix_d[IDXC(j, i, n)] = waf_s[tx];
    faf_matrix_d[IDXC(j, i, n)] = faf_s[tx];
  }
}

template <typename T>
__global__ void GPUGenPhiKernel(const T alpha,
                                const T sqrt_one_minus_alpha,
                                const T denom,
                                const T *waw_matrix_d,
                                const T *waf_matrix_d,
                                const T *faf_matrix_d,
                                const T *gamma_matrix_d,
                                const int n,
                                const int d,
                                bool w_l_changed,
                                T *phi_of_alphas_d,
                                T *phi_of_zeros_d,
                                T *phi_of_zero_primes_d) {


  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  T *mat_s = SharedMemory<T>();
  T *phi_of_alphas_s = (T*)mat_s;
  T *phi_of_zeros_s = (T*)&mat_s[blockDim.x * blockDim.y];
  T *phi_of_zero_primes_s = (T*)&mat_s[2 * blockDim.x * blockDim.y];
  int tid_local = IDXR(threadIdx.y, threadIdx.x, blockDim.x);
  int bid = IDXR(blockIdx.y, blockIdx.x, gridDim.x);

  phi_of_alphas_s[tid_local] = 0.0;
  if (w_l_changed) {
    phi_of_zeros_s[tid_local] = 0.0;
    phi_of_zero_primes_s[tid_local] = 0.0;
  }
  __syncthreads();

  if ((i < n) && (j < n)) {
    T waw = waw_matrix_d[IDXC(j, i, n)];
    T waf = waf_matrix_d[IDXC(j, i, n)];
    T faf = faf_matrix_d[IDXC(j, i, n)];
    T gammaij = gamma_matrix_d[IDXC(j, i, n)];
    T kij = expf(denom * ((faf - waw) * (alpha * alpha) +
        2 * waf * sqrt_one_minus_alpha * alpha + waw));
    phi_of_alphas_s[tid_local] = gammaij * kij;
    if (w_l_changed) {
      T kij = expf(denom * waw);
      phi_of_zeros_s[tid_local] = gammaij * kij;
      phi_of_zero_primes_s[tid_local] = gammaij * denom * 2 * waf * kij;
//    phi_of_alphas_d[IDXC(j, i, n)] = gammaij * kij;
    }
    __syncthreads();
    for (unsigned int s = (blockDim.x * blockDim.y / 2); s > 0; s >>= 1) {
      if (tid_local < s) {
        phi_of_alphas_s[tid_local] += phi_of_alphas_s[tid_local + s];
        if (w_l_changed) {
          phi_of_zeros_s[tid_local] += phi_of_zeros_s[tid_local + s];
          phi_of_zero_primes_s[tid_local] +=
              phi_of_zero_primes_s[tid_local + s];
        }
      }
      __syncthreads();
    }

    if(tid_local == 0) {
      phi_of_alphas_d[bid] = phi_of_alphas_s[tid_local];
      if (w_l_changed) {
        phi_of_zeros_d[bid] = phi_of_zeros_s[tid_local];
        phi_of_zero_primes_d[bid] = phi_of_zero_primes_s[tid_local];
      }
    }
  }
}


template<typename T>
void KDAC<T>::GPUGenAMatrices() {

  unsigned int block_size = nextPow2(d_);
  int shared_mem_size = d_ * sizeof(T) * 2;

  dim3 dim_block(block_size, 1);
  dim3 dim_grid(n_, n_);
  GPUGenAMatricesKernel
      <<<dim_grid, dim_block, shared_mem_size>>>
      (x_matrix_d_, n_, d_, a_matrices_d_);
}

template
void KDAC<float>::GPUGenAMatrices();

template <typename T>
void KDAC<T>::GPUGenPhiCoeff() {
  int block_size = (isPow2(d_)) ? d_ : nextPow2(d_);
  int shared_mem_size = 5 * block_size * sizeof(T);
  dim3 dim_block(block_size, 1);
  dim3 dim_grid(n_, n_);
  GPUGenPhiCoeffKernel
      <<<dim_grid, dim_block, shared_mem_size>>>
      (w_l_d_, gradient_d_, a_matrices_d_, n_, d_,
          waw_matrix_d_, waf_matrix_d_, faf_matrix_d_);
  CUDA_CALL(cudaGetLastError());
}

template
void KDAC<float>::GPUGenPhiCoeff();



template<typename T>
void KDAC<T>::GPUGenPhi(const T sqrt_one_minus_alpha,
               const T denom,
               const bool w_l_changed) {
  int block_dim_x = 16;
  int block_dim_y = 16;
  dim3 dim_block(block_dim_x, block_dim_y);
  // If matrix is n x m, then I need an m x n grid for contiguous
  // memory access
  dim3 dim_grid( (n_-1) / block_dim_x + 1, (n_-1) / block_dim_y + 1);
  int num_blocks = ((n_-1) / block_dim_x + 1) * ((n_-1) / block_dim_y + 1);
  int shared_mem_size = 3 * block_dim_x * block_dim_y * sizeof(T);
  phi_of_alpha_gpu_ = 0;
  if (w_l_changed) {
    phi_of_zero_gpu_ = 0;
    phi_of_zero_prime_gpu_ = 0;
  }

  GPUGenPhiKernel<<<dim_grid, dim_block, shared_mem_size>>>(alpha_,
                                           sqrt_one_minus_alpha,
                                           denom,
                                           waw_matrix_d_,
                                           waf_matrix_d_,
                                           faf_matrix_d_,
                                           gamma_matrix_d_,
                                           n_,
                                           d_,
                                           w_l_changed,
                                           phi_of_alphas_d_,
                                           phi_of_zeros_d_,
                                           phi_of_zero_primes_d_);

  // Check if error happens in kernel launch
  CUDA_CALL(cudaGetLastError());

  CUDA_CALL(cudaMemcpy(phi_of_alphas_h_, phi_of_alphas_d_,
                       num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_blocks; i++)
    phi_of_alpha_gpu_ += phi_of_alphas_h_[i];
  if (w_l_changed) {
    CUDA_CALL(cudaMemcpy(phi_of_zeros_h_, phi_of_zeros_d_,
                         num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(phi_of_zero_primes_h_, phi_of_zero_primes_d_,
                         num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
  }
  for (int i = 0; i < num_blocks; i++) {
    phi_of_zero_gpu_ += phi_of_zeros_h_[i];
    phi_of_zero_prime_gpu_ += phi_of_zero_primes_h_[i];
  }
}

template
void KDAC<float>::GPUGenPhi(const float sqrt_one_minus_alpha,
                      const float denom,
                      const bool w_l_changed);
}  // Namespace NICE