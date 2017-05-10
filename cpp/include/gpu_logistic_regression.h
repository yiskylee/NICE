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

#ifndef CPP_INCLUDE_GPU_LOGISTIC_REGRESSION_H_
#define CPP_INCLUDE_GPU_LOGISTIC_REGRESSION_H_

#ifdef NEED_CUDA

#include <stdlib.h>
#include <time.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <unistd.h>
#include <stdexcept>
#include <ctime>

#include <iostream>

#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"
#include "include/gpu_svd_solver.h"

namespace Nice {

// Abstract class of common matrix operation interface
template <typename T>
class GpuLogisticRegression {
 private:
  static GpuUtil<T> *util_;

 public:
  static Vector<T> Predict(const Matrix<T> &inputs, const Vector<T> thetas){
      Vector<T> predictions, yhat;
      //Matrix<T> product(inputs.rows(),inputs.cols());
      // Allocate and transfer memories
      int m = inputs.rows();
      int n = thetas.cols() - 1;
      int k = inputs.cols();

      const T * h_inputs = &inputs(0);
      const T * h_thetas = &thetas(1);
      Vector<T> product(m * n);

      T * d_inputs;
      T * d_thetas;
      T * d_yhat;
      T * d_product;
      T * d_predictions;

      // Setup GPU memory
      util_->SetupMem(&d_inputs, h_inputs, m * k);
      util_->SetupMem(&d_thetas, h_thetas, k * n);
      util_->SetupMem(&d_product, nullptr, m * n, false);

      // Set up and do cublas matrix multiply
      GpuMatrixMatrixMul(util_->GetBlasHandle(), m, n, k, d_inputs, d_thetas, d_product);

      // Device sync
      util_->SyncDev();

      // Transfer memories back, clear memrory, and return result
      util_->SyncMem(d_inputs, nullptr, 0, false);
      util_->SyncMem(d_thetas, nullptr, 0, false);
      util_->SyncMem(d_product, &product(0, 0), m * n);

      return product;
   }
 
};

template <typename T>
GpuUtil<T> *GpuOperations<T>::util_ = GpuUtil<T>::GetInstance();

}  // namespace Nice
#endif  // NEED_CUDA
#endif  // CPP_INCLUDE_GPU_LOGISTIC_REGRESSION_H_

