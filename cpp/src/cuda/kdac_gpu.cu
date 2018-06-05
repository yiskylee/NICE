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
#include "../../include/kernel_types.h"

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
__device__ T reduce_sum(T *data_s, int n) {
  T sum = 0;
  int block_size = blockDim.x * blockDim.y;
  int tx = threadIdx.y * blockDim.x + threadIdx.x;

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
  return data_s[0];
}



template<typename T>
__global__ void GenKijKernel(const T *x_matrix_d,
                       const T *w_l_d,
                       const float sigma,
                       const int n,
                       const int d,
                       T *kij_matrix_d,
                       T *projection_matrix_d) {
  T *delta_ij_s = SharedMemory<T>();
  T *delta_w_s = SharedMemory<T>() + d;
  int i = blockIdx.y;
  int j = blockIdx.x;
  int tx = threadIdx.x;
  int block_size = blockDim.x;

  for (int k = tx; k < d; k += block_size) {
    delta_ij_s[k] = x_matrix_d[IDXC(i, k, n)] - x_matrix_d[IDXC(j, k, n)];
    // Dot product for delta' * w
    delta_w_s[k] = delta_ij_s[k] * w_l_d[k];
  }
  __syncthreads();

  T projection = reduce_sum(delta_w_s, d);
  T denom = -1.f / (2 * sigma * sigma);
  int index_ij = IDXC(i, j, n);

  if (tx == 0) {
    kij_matrix_d[index_ij] = expf(denom * projection * projection);
    projection_matrix_d[index_ij] = projection;
  }
}

template<typename T>
void KDACGPU<T>::GenKij(const Vector<T> &w_l) {
  if (kernel_type_ == kGaussianKernel) {
    gpu_util_->EigenToDevBuffer(w_l_d_, w_l);
    unsigned int block_size = (d_ < block_limit_ * 2) ?
                              nextPow2((d_+1)/2) : block_limit_;
    int shared_mem_size = 2 * d_ * sizeof(T);
    dim3 dim_block(block_size, 1);
    dim3 dim_grid(n_, n_);
    GenKijKernel<<<dim_grid, dim_block, shared_mem_size>>>(
        x_matrix_d_,
            w_l_d_,
            constant_,
            n_,
            d_,
            kij_matrix_d_,
            wl_deltaxij_proj_matrix_d_);

    gpu_util_->DevBufferToEigen(kij_matrix_, kij_matrix_d_);
    gpu_util_->DevBufferToEigen(wl_deltaxij_proj_matrix_, wl_deltaxij_proj_matrix_d_);
  }
}
template void KDACGPU<float>::GenKij(const Vector<float> &w_l);
template void KDACGPU<double>::GenKij(const Vector<double> &w_l);

//template<typename T>
//__global__ void GenWGradientKernel(const T *x_matrix_d,
//                                   const T *g_of_w_d,
//                                   const T *w_l_d,
//                                   const T *gamma_matrix_d,
//                                   const float constant,
//                                   const int n,
//                                   const int d,
//                                   T *gradient_fs_d) {
//
//  T *delta_ij_s = SharedMemory<T>();
//  T *delta_w_s = SharedMemory<T>() + d;
//  int i = blockIdx.y;
//  int j = blockIdx.x;
//  int tx = threadIdx.x;
//  int block_size = blockDim.x;
//
//
//  for (int k = tx; k < d; k += block_size) {
//    delta_ij_s[k] = x_matrix_d[IDXC(i, k, n)] - x_matrix_d[IDXC(j, k, n)];
//    // Dot product for delta' * w
//    delta_w_s[k] = delta_ij_s[k] * w_l_d[k];
//  }
//  __syncthreads();
//
//  T delta_w = reduce_sum(delta_w_s, d);
//  T waw = delta_w * delta_w;
//
//  T sigma_sq = constant * constant;
//
//  int index_ij = IDXC(i, j, n);
//  T gamma_ij = gamma_matrix_d[index_ij];
//  T g_of_w_ij = g_of_w_d[index_ij];
//  T exp_term = expf(-waw / (2 * sigma_sq));
//  T coeff = -gamma_ij * g_of_w_ij * exp_term / sigma_sq;
//  T *gradient_f_ij = gradient_fs_d + IDXR(i, j, n) * d;
//  // delta * delta_w == Aij * w
//  for (int k = tx; k < d; k += block_size)
//    gradient_f_ij[k] = coeff * delta_ij_s[k] * delta_w;
//}


//template<typename T>
//Vector<T> KDACGPU<T>::GenWGradient(const Vector <T> &w_l) {
//  Vector<T> w_gradient = Vector<T>::Zero(d_);
//  if (kernel_type_ == kGaussianKernel) {
//    gpu_util_->EigenToDevBuffer(w_l_d_, w_l);
//    gpu_util_->EigenToDevBuffer(g_of_w_d_, g_of_w_);
//
////    CUDA_CALL(cudaMemcpy(w_l_d_, &w_l(0), d_ * sizeof(T),
////                         cudaMemcpyHostToDevice));
//    // When block_limit is 512
//    // If d is 128, block_size is 64
//    // If d is 6, block_size is 4
//    // If d is 1025, block_size is 512
//    unsigned int block_size = (d_ < block_limit_ * 2) ?
//                              nextPow2((d_+1)/2) : block_limit_;
//
//    int shared_mem_size = 2 * d_ * sizeof(T);
//
//    dim3 dim_block(block_size, 1);
//    dim3 dim_grid(n_, n_);
//    GenWGradientKernel
//        << < dim_grid, dim_block, shared_mem_size >> >
//        (x_matrix_d_,
//            g_of_w_d_,
//            w_l_d_,
//            gamma_matrix_d_,
//            constant_,
//            n_,
//            d_,
//            grad_f_arr_d_);
//    CUDA_CALL(cudaGetLastError());
//    CUDA_CALL(cudaMemcpy(grad_f_arr_h_, grad_f_arr_d_,
//                         n_ * n_ * d_ * sizeof(T),
//                         cudaMemcpyDeviceToHost));
//
//
//    for (int i = 0; i < n_; i++) {
//      for (int j = 0; j < n_; j++) {
//        T *grad_f_ij = grad_f_arr_h_ + IDXR(i, j, n_) * d_;
//        Vector<T> grad_temp = Eigen::Map < Vector < T >> (grad_f_ij, d_);
//        util::CheckFinite(grad_temp, "grad_temp_"+std::to_string(i));
//        w_gradient = w_gradient + grad_temp;
//      }
//    }
//  }
//  util::CheckFinite(w_gradient, "w_gradient");
//  return w_gradient;
//}
//
//template
//Vector<float> KDACGPU<float>::GenWGradient(const Vector<float> &w_l);
//template
//Vector<double> KDACGPU<double>::GenWGradient(const Vector<double> &w_l);

}  // Namespace NICE