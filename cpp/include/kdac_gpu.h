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

// Kernel Dimension Alternative Clustering (KDACCPU)
// Please refer to the paper published in PAMI by Liu, Dy and Jordan at:
// http://people.eecs.berkeley.edu/~jordan/papers/niu-dy-jordan-pami.pdf
// We try to follow naming conventions in the paper as much as possible.
// The lower cased variable names is the same as in the paper, and the
// upper cased matrix variable names in the paper are converted to lower
// case suffixed with "_matrix". For example:
// matrix U in the paper is named u_matrix in this implementation.

#ifndef CPP_INCLUDE_KDAC_GPU_H
#define CPP_INCLUDE_KDAC_GPU_H

//#ifndef CUDA_AND_GPU
//#define CUDA_AND_GPU
//#endif

#ifdef CUDA_AND_GPU

#include "include/kdac.h"
#include "include/gpu_util.h"
#include "../../../../../../../usr/local/cuda/include/driver_types.h"

namespace Nice {
template<typename T>
class KDACGPU: public KDAC<T> {
 public:
  using KDAC<T>::n_;
  using KDAC<T>::d_;
  using KDAC<T>::x_matrix_;
  using KDAC<T>::profiler_;
  using KDAC<T>::g_of_w_;
  using KDAC<T>::gamma_matrix_;
  using KDAC<T>::kernel_type_;
  using KDAC<T>::phi_of_alpha_;
  using KDAC<T>::phi_of_zero_;
  using KDAC<T>::phi_of_zero_prime_;
  using KDAC<T>::constant_;
  using KDAC<T>::kij_matrix_;
  using KDAC<T>::wl_deltaxij_proj_matrix_;

  /// This is the default constructor for KDACGPU
  /// Number of clusters c and reduced dimension q will be both set to 2
  KDACGPU() :
      block_limit_(256) {}

  ~KDACGPU() {
    // Free parameters, intermediate delta and parameters
    CUDA_CALL(cudaFree(x_matrix_d_));
    CUDA_CALL(cudaFree(g_of_w_d_));
    CUDA_CALL(cudaFree(gamma_matrix_d_));
    CUDA_CALL(cudaFree(w_l_d_));
    CUDA_CALL(cudaFree(kij_matrix_d_));
    CUDA_CALL(cudaFree(wl_deltaxij_proj_matrix_d_));
  }
  KDACGPU(const KDACGPU &rhs) {}

//  Vector<T> GenWGradient(const Vector<T> &w_l);
  void GenKij(const Vector<T> &w_l);

 private:
  // Input matrix X (n by d) on device
  T* x_matrix_d_;
  // Stores the result of current g_of_w cwiseProduct current Kij
  // when the current Kij is finalized, new_g_of_w becomes the kernel matrix
  T *g_of_w_d_;
  // Kernel matrix kij for X's projection on one column w_l
  // Temporarily store kernel matrix for converged w_ls
  T* kij_matrix_d_;
  // Store the projection of w_l on delta_x_ij
  T* wl_deltaxij_proj_matrix_d_;
  T* gamma_matrix_d_;
  // Device memory for each column (1 x d) in W,
  T* w_l_d_;
  // GPUUtil object to setup memory etc.
  GpuUtil<T> *gpu_util_;
  unsigned int block_limit_;

  void InitX(const Matrix<T> &input_matrix) {
    KDAC<T>::InitX(input_matrix);
    gpu_util_->SetupMem(&x_matrix_d_, &(x_matrix_(0)), n_ * d_);
    gpu_util_->SetupMem(&g_of_w_d_, &(g_of_w_(0)), n_ * n_);
  }

  void InitW() {
    KDAC<T>::InitW();
    gpu_util_->SetupMem(&gamma_matrix_d_, nullptr, n_ * n_, false);
    gpu_util_->SetupMem(&w_l_d_, nullptr, d_, false);
    gpu_util_->SetupMem(&kij_matrix_d_, nullptr, n_ * n_, false);
    gpu_util_->SetupMem(&wl_deltaxij_proj_matrix_d_, nullptr, n_ * n_, false);
  }

  void OptimizeW() {
    KDAC<T>::GenGammaMatrix();
    gpu_util_->EigenToDevBuffer(gamma_matrix_d_, gamma_matrix_);
    KDAC<T>::OptimizeW();
  }
};
}  // namespace Nice

#endif  // CUDA_AND_GPU
#endif  // CPP_INCLUDE_KDAC_GPU_H_