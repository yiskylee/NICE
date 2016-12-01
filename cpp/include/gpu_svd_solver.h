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

#ifndef CPP_INCLUDE_GPU_SVD_SOLVER_H_
#define CPP_INCLUDE_GPU_SVD_SOLVER_H_

#ifdef NEED_CUDA
#include<cuda_runtime_api.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cusolverDn.h>

#include<iostream>

#include "Eigen/Dense"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"

namespace Nice {

template<typename T>
class GpuSvdSolver {
 private:
  Matrix<T> u_;
  Matrix<T> v_;
  Vector<T> s_;
  static GpuUtil<T> *util_;

 public:
  GpuSvdSolver() {}

  void Compute(const Matrix<T> &A) {
    int M = A.rows();
    int N = A.cols();
    const T *h_A = &A(0);
    // --- Setting the device matrix and moving the host matrix to the device
    T *d_A;
    util_->SetupMem(&d_A, h_A, M * N);

    //--- host side SVD results space
    s_.resize(M, 1);
    u_.resize(M, M);
    v_.resize(N, N);

    // --- device side SVD workspace and matrices
    int work_size = 0;
    int devInfo_h = 0;
    int *devInfo;
    util_->SetupIntMem(&devInfo, nullptr, 1, false);
    T *d_U;
    T *d_V;
    T *d_S;

    util_->SetupMem(&d_U, nullptr, M * M, false);
    util_->SetupMem(&d_V, nullptr, N * N, false);
    util_->SetupMem(&d_S, nullptr, N, false);

    cusolverDnSgesvd_bufferSize(util_->GetSolverHandle(), M, N, &work_size);

    T *work;
    util_->SetupMem(&work, nullptr, work_size, false);

    // --- CUDA SVD execution
    GpuSvd(util_->GetSolverHandle(), M, N,
           d_A, d_S, d_U, d_V,
           work, work_size, devInfo);

    // Error Check
    util_->SyncIntMem(devInfo, &devInfo_h, 1);

    // Device sync
    util_->SyncDev();

    // Transfer memories back, clear memrory, and return result
    util_->SyncMem(d_S, &s_(0, 0), N);
    util_->SyncMem(d_U, &u_(0, 0), M * M);
    util_->SyncMem(d_V, &v_(0, 0), N * N);
  }

  Matrix<T> MatrixU() const              { return u_; }

  Matrix<T> MatrixV() const              { return v_; }

  Vector<T> SingularValues() const       { return s_; }
};

template <typename T>
GpuUtil<T> *GpuSvdSolver<T>::util_ = GpuUtil<T>::GetInstance();
}  // namespace Nice

#endif  // NEED_CUDA

#endif  // CPP_INCLUDE_GPU_SVD_SOLVER_H_

