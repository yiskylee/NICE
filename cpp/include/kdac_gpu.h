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

#ifdef CUDA_AND_GPU

#include "include/kdac.h"
#include "include/gpu_util.h"

namespace Nice {
template<typename T>
class KDACGPU: public KDAC<T> {
 public:
  /// This is the default constructor for KDACGPU
  /// Number of clusters c and reduced dimension q will be both set to 2
  KDACGPU() :
      block_limit_(256){}

  ~KDACGPU() {
    // Free parameters, intermediate delta and parameters
    CUDA_CALL(cudaFree(x_matrix_d_));
    CUDA_CALL(cudaFree(gamma_matrix_d_));
    CUDA_CALL(cudaFree(waw_matrix_d_));
    CUDA_CALL(cudaFree(waf_matrix_d_));
    CUDA_CALL(cudaFree(faf_matrix_d_));
    CUDA_CALL(cudaFree(w_l_d_));
    CUDA_CALL(cudaFree(gradient_d_));
    CUDA_CALL(cudaFree(phi_of_alphas_d_));
    CUDA_CALL(cudaFree(phi_of_zeros_d_));
    CUDA_CALL(cudaFree(phi_of_zero_primes_d_));
    CUDA_CALL(cudaFree(g_of_w_d_));
    CUDA_CALL(cudaFree(gradient_fs_d_));

    delete [] phi_of_alphas_h_;
    delete [] phi_of_zeros_h_;
    delete [] phi_of_zero_primes_h_;
  }
  KDACGPU(const KDACGPU &rhs) {}

  void GenPhiCoeff(const Vector<T> &w_l, const Vector<T> &gradient);
  void GenPhi(const Vector<T> &w_l,
              const Vector<T> &gradient,
              bool w_l_changed);
  Vector<T>
  GenWGradient(const Vector<T> &w_l);
  void UpdateGOfW(const Vector<T> &w_l);

 private:
  T* x_matrix_d_; // Input matrix X (n by d) on device
  T* gamma_matrix_d_;
  T* waw_matrix_d_;
  T* waf_matrix_d_;
  T* faf_matrix_d_;
  // Device memory for each column (1 x d) in W,
  T* w_l_d_;
  // Device memory for gradient (1 x d) for each column in W
  T* gradient_d_;
  T *phi_of_alphas_d_, *phi_of_zeros_d_, *phi_of_zero_primes_d_;
  T *phi_of_alphas_h_, *phi_of_zeros_h_, *phi_of_zero_primes_h_;
  T *g_of_w_d_;
  T *gradient_fs_d_, *gradient_fs_h_;
  // GPUUtil object to setup memory etc.
  GpuUtil<T> *gpu_util_;
  unsigned int block_limit_;

  // Initialization for generating alternative views with a given Y
  void Init(const Matrix<T> &input_matrix, const Matrix<T> &y_matrix) {
    KDAC<T>::Init(input_matrix, y_matrix);
    int n = this->n_;
    int d = this->d_;
    this->profiler_["gen_phi"].Start();
    gpu_util_->SetupMem(&x_matrix_d_,
                        &(this->x_matrix_(0)), n * d);
    gpu_util_->SetupMem(&waw_matrix_d_, nullptr, n * n, false);
    gpu_util_->SetupMem(&waf_matrix_d_, nullptr, n * n, false);
    gpu_util_->SetupMem(&faf_matrix_d_, nullptr, n * n, false);
    gpu_util_->SetupMem(&w_l_d_, nullptr, d, false);
    gpu_util_->SetupMem(&gradient_d_, nullptr, d, false);
    gpu_util_->SetupMem(&gamma_matrix_d_, nullptr, n * n, false);
    gpu_util_->SetupMem(&g_of_w_d_, nullptr, n * n, false);
    gpu_util_->SetupMem(&gradient_fs_d_, nullptr,
                        n * n * d, false);
    int num_blocks = ((n - 1) / 16 + 1) * ((n - 1) / 16 + 1);
    gpu_util_->SetupMem(&phi_of_alphas_d_, nullptr, num_blocks, false);
    gpu_util_->SetupMem(&phi_of_zeros_d_, nullptr, num_blocks, false);
    gpu_util_->SetupMem(&phi_of_zero_primes_d_, nullptr, num_blocks, false);
    phi_of_alphas_h_ = new T[num_blocks];
    phi_of_zeros_h_ = new T[num_blocks];
    phi_of_zero_primes_h_ = new T[num_blocks];
    gradient_fs_h_ = new T[n * n * d];
    this->profiler_["gen_phi"].Record();
  }

  void Init(const Matrix<T> &input_matrix) {
    KDAC<T>::Init(input_matrix);
    int n = this->n_;
    int d = this->d_;
    this->profiler_["gen_phi"].Start();
    gpu_util_->SetupMem(&x_matrix_d_,
                        &(this->x_matrix_(0)), n * d);
    gpu_util_->SetupMem(&waw_matrix_d_, nullptr, n * n, false);
    gpu_util_->SetupMem(&waf_matrix_d_, nullptr, n * n, false);
    gpu_util_->SetupMem(&faf_matrix_d_, nullptr, n * n, false);
    gpu_util_->SetupMem(&w_l_d_, nullptr, d, false);
    gpu_util_->SetupMem(&gradient_d_, nullptr, d, false);
    gpu_util_->SetupMem(&gamma_matrix_d_, nullptr, n * n, false);
    gpu_util_->SetupMem(&g_of_w_d_, nullptr, n * n, false);
    gpu_util_->SetupMem(&gradient_fs_d_, nullptr,
                        n * n * d, false);
    int num_blocks = ((n - 1) / 16 + 1) * ((n - 1) / 16 + 1);
    gpu_util_->SetupMem(&phi_of_alphas_d_, nullptr, num_blocks, false);
    gpu_util_->SetupMem(&phi_of_zeros_d_, nullptr, num_blocks, false);
    gpu_util_->SetupMem(&phi_of_zero_primes_d_, nullptr, num_blocks, false);
    phi_of_alphas_h_ = new T[num_blocks];
    phi_of_zeros_h_ = new T[num_blocks];
    phi_of_zero_primes_h_ = new T[num_blocks];
    gradient_fs_h_ = new T[n * n * d];
    this->profiler_["gen_phi"].Record();
  }

  void OptimizeW(void) {
    KDAC<T>::GenGammaMatrix();
    CUDA_CALL(cudaMemcpy(gamma_matrix_d_, &(this->gamma_matrix_)(0),
                         this->n_ * this->n_ * sizeof(T),
                         cudaMemcpyHostToDevice));
    KDAC<T>::GenGofW();
    CUDA_CALL(cudaMemcpy(g_of_w_d_, &(this->g_of_w_)(0),
                         this->n_ * this->n_ * sizeof(T),
                         cudaMemcpyHostToDevice));
    KDAC<T>::OptimizeW();
  }
};
}  // namespace Nice

#endif  // CUDA_AND_GPU
#endif  // CPP_INCLUDE_KDAC_GPU_H_
