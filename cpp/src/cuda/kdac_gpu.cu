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

#include <iostream>
#include "include/kdac_gpu.h"
#include "include/gpu_util.h"

// Hack to cope with Clion
#include "../../include/gpu_util.h"
#include "../../../../../../../../usr/local/cuda/include/driver_types.h"
#include "../../include/kdac_gpu.h"

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
__device__ void mv(T *mat_s,
                   T* vec_in_s,
                   const int num_rows,
                   const int num_cols,
                   T* vec_out_s) {
  int tx = threadIdx.x;
  int block_size = blockDim.x;
  for (int k = tx; k < num_rows; k += block_size) {
    for (int col = 0; col < num_cols; col++)
      vec_out_s[k] += mat_s[IDXC(k, col, num_rows)] * vec_in_s[col];
  }
  __syncthreads();
}

template <typename T>
__device__ void reduce_sum(T *data_s, int n) {
  T sum = 0;
  int block_size = blockDim.x;
  int tx = threadIdx.x;

  for (int k = tx; k < n; k += block_size)
    sum += data_s[k];

  data_s[tx] = sum;
  __syncthreads();

  if ((block_size >= 512) && (tx < 256))
    data_s[tx] = sum = sum + data_s[tx + 256];
  __syncthreads();

  if ((block_size >= 256) && (tx < 128))
    data_s[tx] = sum = sum + data_s[tx + 128];
  __syncthreads();

  if ((block_size >= 128) && (tx < 64))
    data_s[tx] = sum = sum + data_s[tx + 64];
  __syncthreads();

//    if ((block_size >= 64) && (tx < 32))
//      exp_term_s[tx] = sum = sum + exp_term_s[tx + 32];
//    __syncthreads();
//
//    if ((block_size >= 32) && (tx < 16))
//      exp_term_s[tx] = sum = sum + exp_term_s[tx + 16];
//    __syncthreads();
//
//    if ((block_size >= 16) && (tx < 8))
//      exp_term_s[tx] = sum = sum + exp_term_s[tx + 8];
//    __syncthreads();
//
//    if ((block_size >= 8) && (tx < 4))
//      exp_term_s[tx] = sum = sum + exp_term_s[tx + 4];
//    __syncthreads();
//
//    if ((block_size >= 4) && (tx < 2))
//      exp_term_s[tx] = sum = sum + exp_term_s[tx + 2];
//    __syncthreads();
//
//    if ((block_size >= 2) && (tx < 1))
//      exp_term_s[tx] = sum = sum + exp_term_s[tx + 1];
//    __syncthreads();

  if (tx < 32) {
    if (block_size >= 64)
      sum += data_s[tx + 32];
    for (int offset = warpSize / 2; offset >0; offset /=2)
      sum += __shfl_down(sum, offset);
  }
  if (tx == 0)
    data_s[tx] = sum;
  __syncthreads();
}


template <typename T>
__device__ void GenAij(const T *x_matrix_d,
                       const int n,
                       const int d,
                       T *a_ij_d,
                       T *delta_ij_d) {
  int tx = threadIdx.x;
  int i = blockIdx.y;
  int j = blockIdx.x;

  while (tx < d) {
    delta_ij_d[tx] = x_matrix_d[IDXC(i, tx, n)] -
        x_matrix_d[IDXC(j, tx, n)];
    tx += blockDim.x;
    __syncthreads();
  }

  tx = threadIdx.x;

  while (tx < d) {
    for (int col = 0; col < d; col++)
      // thread tx calculates a whole row tx of the output matrix a_ij
      a_ij_d[IDXC(tx, col, d)] = delta_ij_d[col] * delta_ij_d[tx];
    tx += blockDim.x;
  }
}

template<typename T>
__global__ void UpdateGOfWKernel(const T *x_matrix_d,
                                 const T *w_l_d,
                                 const float constant,
                                 const int n,
                                 const int d,
                                 T *g_of_w_d_) {
  T *vec_s = SharedMemory<T>();
  // Shared memory to store a_ij * w_l
  T *aw_s = (T *) vec_s;
  // Shared memory for w_l
  T *w_s = (T *) &vec_s[d];
  // Shared memory for a_ij
  T *a_ij_s = (T *) &vec_s[2 * d];
  // Shared memory for delta_ij
  T *delta_ij_s = (T *) &vec_s[2 * d + d * d];

  GenAij(x_matrix_d, n, d, a_ij_s, delta_ij_s);

  int i = blockIdx.y;
  int j = blockIdx.x;
  int tx = threadIdx.x;
  int block_size = blockDim.x;

  for (int k = tx; k < d; k += block_size) {
    w_s[k] = w_l_d[k];
    aw_s[k] = 0.0;
  }
  __syncthreads();

  mv(a_ij_s, w_s, d, d, aw_s);

  // Dot Product
  for (int k = tx; k < d; k += block_size)
    aw_s[k] = aw_s[k] * (-w_s[k]);
  __syncthreads();
  reduce_sum(aw_s, d);
  if (tx == 0) {
    g_of_w_d_[IDXC(i,j,n)] = g_of_w_d_[IDXC(i,j,n)] *
        aw_s[0] / (2 * constant * constant);
	}
}

template<typename T>
__global__ void GenAMatricesKernel(const T *x_matrix_d,
                                   const int n,
                                   const int d,
                                   T *a_matrices_d) {
  int i = blockIdx.y;
  int j = blockIdx.x;
  T *delta_ij_d = SharedMemory<T>();
  T *a_ij_d = a_matrices_d + IDXR(i, j, n) * (d * d);
  GenAij(x_matrix_d, n, d, a_ij_d, delta_ij_d);
//  int tx = threadIdx.x;
//  int i = blockIdx.y;
//  int j = blockIdx.x;
//
//  while (tx < d) {
//    delta_ij_d[tx] = x_matrix_d[IDXC(i, tx, n)] -
//        x_matrix_d[IDXC(j, tx, n)];
//    tx += blockDim.x;
//    __syncthreads();
//  }
//
//  tx = threadIdx.x;
//  T *a_ij_d = a_matrices_d + IDXR(i, j, n) * (d * d);
//
//  while (tx < d) {
//    for (int col = 0; col < d; col++)
//      // thread tx calculates a whole row tx of the output matrix a_ij_d
//      a_ij_d[IDXC(tx, col, d)] = delta_ij_d[col] * delta_ij_d[tx];
//    tx += blockDim.x;
//  }
}

template<typename T>
__global__ void GenPhiCoeffKernel(const T *x_matrix_d,
                                  const T *w_l_d,
                                  const T *gradient_d,
                                  const int n,
                                  const int d,
                                  T *waw_matrix_d,
                                  T *waf_matrix_d,
                                  T *faf_matrix_d) {
  T *vec_s = SharedMemory<T>();
  T *waw_s = (T *) vec_s;
  T *waf_s = (T *) &vec_s[d];
  T *faf_s = (T *) &vec_s[2 * d];
  T *w_s = (T *) &vec_s[3 * d];
  T *grad_s = (T *) &vec_s[4 * d];
  T *a_ij_s = (T *) &vec_s[5 * d];
  T *delta_ij_s = (T *) &vec_s[5 * d + d * d];

  GenAij(x_matrix_d, n, d, a_ij_s, delta_ij_s);

  int i = blockIdx.y;
  int j = blockIdx.x;
  int tx = threadIdx.x;
  int block_size = blockDim.x;

  for (int k = tx; k < d; k += block_size) {
    waw_s[k] = 0.0;
    waf_s[k] = 0.0;
    faf_s[k] = 0.0;
    w_s[k] = w_l_d[k];
    grad_s[k] = gradient_d[k];
  }
  __syncthreads();

  mv(a_ij_s, w_s, d, d, waw_s);
  mv(a_ij_s, grad_s, d, d, waf_s);
  mv(a_ij_s, grad_s, d, d, faf_s);

  // Dot Product
  for (int k = tx; k < d; k += block_size) {
    waw_s[k] = waw_s[k] * w_s[k];
    waf_s[k] = waf_s[k] * w_s[k];
    faf_s[k] = faf_s[k] * grad_s[k];
  }
  __syncthreads();
  reduce_sum(waw_s, d);
  reduce_sum(waf_s, d);
  reduce_sum(faf_s, d);

  // Transposed access for better access pattern as waw_s matrix is column-major
  if (tx == 0) {
    waw_matrix_d[IDXC(j, i, n)] = waw_s[tx];
    waf_matrix_d[IDXC(j, i, n)] = waf_s[tx];
    faf_matrix_d[IDXC(j, i, n)] = faf_s[tx];
  }


//  if (tx < d) {
//    // Each tx takes care of one row of matrix in order to have a
//    // coalesced access pattern
//    // Each time it aggreates a column
//    for (int col = 0; col < d; col++) {
//      waw_s[tx] += a_ij[IDXC(tx, col, d)] * w_s[col];
//      waf_s[tx] += a_ij[IDXC(tx, col, d)] * grad_s[col];
////      faf_s[tx] += a_ij[IDXC(tx, col, d)] * gradient_d[col];
//    }
//    faf_s[tx] = waf_s[tx];
//
//    // This is the dot product
//    waw_s[tx] = waw_s[tx] * w_s[tx];
//    waf_s[tx] = waf_s[tx] * w_s[tx];
//    faf_s[tx] = faf_s[tx] * grad_s[tx];
//  }
//  __syncthreads();

  // Reduction for dot product
//  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//    if (tx < s) {
//      waw_s[tx] += waw_s[tx + s];
//      waf_s[tx] += waf_s[tx + s];
//      faf_s[tx] += faf_s[tx + s];
//    }
//    __syncthreads();
//  }
//    if (tx < 8) {
//      vec_s[tx] += vec_s[tx + 8];
//      vec_s[tx] += vec_s[tx + 4];
//      vec_s[tx] += vec_s[tx + 2];
//      vec_s[tx] += vec_s[tx + 1];
//    }
//    __syncthreads();


}

template<typename T>
__global__ void GenPhiKernel(const T alpha,
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
  T *phi_of_alphas_s = (T *) mat_s;
  T *phi_of_zeros_s = (T *) &mat_s[blockDim.x * blockDim.y];
  T *phi_of_zero_primes_s = (T *) &mat_s[2 * blockDim.x * blockDim.y];
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

    if (tid_local == 0) {
      phi_of_alphas_d[bid] = phi_of_alphas_s[tid_local];
      if (w_l_changed) {
        phi_of_zeros_d[bid] = phi_of_zeros_s[tid_local];
        phi_of_zero_primes_d[bid] = phi_of_zero_primes_s[tid_local];
      }
    }
  }
}

template<typename T>
__global__ void GenWGradientKernel(const T *x_matrix_d,
                                   const T *g_of_w_d,
                                   const T *w_l_d,
                                   const T *gamma_matrix_d,
                                   const float constant,
                                   const int n,
                                   const int d,
                                   T *gradient_fs_d) {
  T *vec_s = SharedMemory<T>();
  // Memory to store a_ij * w_l
  T *aw_s = (T *) vec_s;
  // exp_term_s first store neg_waw_s, and then parallelly sum it
  // to store the dot product result
  T *exp_term_s = (T *) &vec_s[d];
  T *w_s = (T *) &vec_s[2 * d];
  T *a_ij_s = (T *) &vec_s[3 * d];
  T *delta_ij_s = (T *) &vec_s[3 * d + d * d];

  GenAij(x_matrix_d, n, d, a_ij_s, delta_ij_s);

  int i = blockIdx.y;
  int j = blockIdx.x;
  int tx = threadIdx.x;
  int block_size = blockDim.x;

  T *gradient_f_ij = gradient_fs_d + IDXR(i, j, n) * d;

  // Copy data from global memory to shared memory
  for (int k = tx; k < d; k += block_size) {
    aw_s[k] = 0.0;
    exp_term_s[k] = 0.0;
    w_s[k] = w_l_d[k];
  }
  __syncthreads();
  // Matrix vector multiplication
  mv(a_ij_s, w_s, d, d, aw_s);

  // Dot Product
  for (int k = tx; k < d; k += block_size)
    exp_term_s[k] = aw_s[k] * (-w_s[k]);
  __syncthreads();

  // Reudction for dot product, result stored in exp_term_s[0]
  reduce_sum(exp_term_s, d);
  T exp_term = expf(exp_term_s[0] / (2 * constant * constant));

  int index_ij = IDXC(i, j, n);
  for (int k = tx; k < d; k += block_size)
    gradient_f_ij[k] = -gamma_matrix_d[index_ij] * g_of_w_d[index_ij] *
        exp_term * aw_s[k] / (constant * constant);
}

template<typename T>
void KDACGPU<T>::GenPhiCoeff(const Vector <T> &w_l,
                             const Vector <T> &gradient) {
  int n = this->n_;
  int d = this->d_;
  // Three terms used to calculate phi of alpha
  // They only change if w_l or gradient change
  CUDA_CALL(cudaMemcpy(w_l_d_, &w_l(0), d * sizeof(T),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(gradient_d_, &gradient(0), d * sizeof(T),
                       cudaMemcpyHostToDevice));

  unsigned int block_size = (d < block_limit_ * 2) ?
                            nextPow2((d+1)/2) : block_limit_;

  int shared_mem_size = 5 * d * sizeof(T);
  shared_mem_size += d * d * sizeof(T);
  shared_mem_size += d * sizeof(T);
  dim3 dim_block(block_size, 1);
  dim3 dim_grid(n, n);
  GenPhiCoeffKernel <<<dim_grid, dim_block, shared_mem_size>>> (
      x_matrix_d_,
      w_l_d_,
      gradient_d_,
      n,
      d,
      waw_matrix_d_,
      waf_matrix_d_,
      faf_matrix_d_);
  CUDA_CALL(cudaGetLastError());
}

template
void KDACGPU<float>::GenPhiCoeff(const Vector<float> &w_l,
                                 const Vector<float> &gradient);

// Generate phi(alpha), phi(0) and phi'(0) for LineSearch
// If this is the first time to generate phi(), then w_l_changed is true
// Or if the w_l is negated because phi'(0) is negative,
// then w_l_changed is true
// If w_l_changed is true, generate phi(0) and phi'(0), otherwise
// when we are only computing phi(alpha) with a different alpha in the loop
// of the LineSearch, the w_l_changed is false and we do not generate
// new waw, waf and faf
template<typename T>
void KDACGPU<T>::GenPhi(const Vector <T> &w_l,
                        const Vector <T> &gradient,
                        bool w_l_changed) {
  int n = this->n_;
  int d = this->d_;

  if (this->kernel_type_ == kGaussianKernel) {
    float alpha_square = pow(this->alpha_, 2);
    float sqrt_one_minus_alpha = pow((1 - alpha_square), 0.5);
    float denom = -1 / (2 * pow(this->constant_, 2));

    this->profiler_.gen_phi.Start();
    this->phi_of_alpha_ = 0;

    if (w_l_changed) {
      GenPhiCoeff(w_l, gradient);
      this->phi_of_zero_ = 0;
      this->phi_of_zero_prime_ = 0;
    }

    int block_dim_x = 16;
    int block_dim_y = 16;
    dim3 dim_block(block_dim_x, block_dim_y);
    // If matrix is n x m, then I need an m x n grid for contiguous
    // memory access
    dim3 dim_grid((n - 1) / block_dim_x + 1,
                  (n - 1) / block_dim_y + 1);
    int num_blocks =
        ((n - 1) / block_dim_x + 1) * ((n - 1) / block_dim_y + 1);
    int shared_mem_size = 3 * block_dim_x * block_dim_y * sizeof(T);

    GenPhiKernel << < dim_grid, dim_block, shared_mem_size >> >
        (this->alpha_,
        sqrt_one_minus_alpha,
        denom,
        waw_matrix_d_,
        waf_matrix_d_,
        faf_matrix_d_,
        gamma_matrix_d_,
        n,
        d,
        w_l_changed,
        phi_of_alphas_d_,
        phi_of_zeros_d_,
        phi_of_zero_primes_d_);

    // Check if error happens in kernel launch
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaMemcpy(phi_of_alphas_h_, phi_of_alphas_d_,
                         num_blocks * sizeof(T), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_blocks; i++) {
      this->phi_of_alpha_ += phi_of_alphas_h_[i];
    }
    if (w_l_changed) {
      CUDA_CALL(cudaMemcpy(phi_of_zeros_h_, phi_of_zeros_d_,
                           num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaMemcpy(phi_of_zero_primes_h_, phi_of_zero_primes_d_,
                           num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
      for (int i = 0; i < num_blocks; i++) {
        this->phi_of_zero_ += phi_of_zeros_h_[i];
        this->phi_of_zero_prime_ += phi_of_zero_primes_h_[i];
      }
    }
    this->profiler_.gen_phi.Record();
  }
}

template
void KDACGPU<float>::GenPhi(const Vector<float> &w_l,
                            const Vector<float> &gradient,
                            bool w_l_changed);

template<typename T>
Vector <T> KDACGPU<T>::GenWGradient(const Vector <T> &w_l) {
  int n = this->n_;
  int d = this->d_;
  Vector <T> w_gradient = Vector<T>::Zero(d);
  if (this->kernel_type_ == kGaussianKernel) {
    CUDA_CALL(cudaMemcpy(w_l_d_, &w_l(0), d * sizeof(T),
                         cudaMemcpyHostToDevice));
    // When block_limit is 512
    // If d is 128, block_size is 64
    // If d is 6, block_size is 4
    // If d is 1025, block_size is 512
    unsigned int block_size = (d < block_limit_ * 2) ?
                              nextPow2((d+1)/2) : block_limit_;

    int shared_mem_size = 3 * d * sizeof(T);
    shared_mem_size += d * d * sizeof(T);
    shared_mem_size += d * sizeof(T);

    dim3 dim_block(block_size, 1);
    dim3 dim_grid(n, n);
    GenWGradientKernel
        << < dim_grid, dim_block, shared_mem_size >> >
        (x_matrix_d_,
            g_of_w_d_,
            w_l_d_,
            gamma_matrix_d_,
            this->constant_,
            n,
            d,
            gradient_fs_d_);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaMemcpy(gradient_fs_h_, gradient_fs_d_,
                         n * n * d * sizeof(T),
                         cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        T *gradient_f_ij = gradient_fs_h_ + IDXR(i, j, n) * d;
        w_gradient += Eigen::Map < Vector < T >> (gradient_f_ij, d);
      }
    }
  }
  return w_gradient;
}

template
Vector<float> KDACGPU<float>::GenWGradient(const Vector<float> &w_l);


template<typename T>
void KDACGPU<T>::UpdateGOfW(const Vector<T> &w_l) {
  int n = this->n_;
  int d = this->d_;
  CUDA_CALL(cudaMemcpy(w_l_d_, &w_l(0), d * sizeof(T),
                       cudaMemcpyHostToDevice));
  if (this->kernel_type_ == kGaussianKernel) {
    unsigned int block_size = (d < block_limit_ * 2) ?
        nextPow2((d+1)/2) : block_limit_;
    int shared_mem_size = (d + d + d*d + d) * sizeof(T);
    dim3 dim_block(block_size, 1);
    dim3 dim_grid(n, n);
    UpdateGOfWKernel <<<dim_grid, dim_block, shared_mem_size>>>
        (x_matrix_d_,
         w_l_d_,
         this->constant_,
         n,
         d,
         g_of_w_d_);
  }
}

template
void KDACGPU<float>::UpdateGOfW(const Vector<float> &w_l);
}  // Namespace NICE