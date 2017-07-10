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

#ifndef CPP_INCLUDE_CUDA_MATRIX_VECTOR_MULTIPLY_H_
#define CPP_INCLUDE_CUDA_MATRIX_VECTOR_MULTIPLY_H_

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
class CudaMatrixVectorMultiply: public CudaMatrixVectorMultiply<T> {
  private:
    static GpuUtil<T> *util_;
    const T * h_a;
    const T * h_x;
    Vector<T> h_y;
    T * d_a;
    T * d_x;
    T * d_y;



  public:
     CudaMatrixVectorMultiply(){}
     Nice::Matrix<T> GetMatrix() {return h_a};
     Nice::Vector<T> GetVector() {return h_x};
     void Multiply(const Matrix<T> &a, const Vector<T> &b) {
       if (a.cols() == b.rows() && !a.isZero()) {
         // Allocate and transfer memories
         int m = a.rows();
         int n = b.cols();
         int k = a.cols();

         const T * h_a = &a(0);
         const T * h_x = &b(0);
         Vector<T> h_y(m);

         T * d_a;
         T * d_x;
         T * d_y;

         // Setup GPU memory
         util_->SetupMem(&d_a, h_a, m * k);
         util_->SetupMem(&d_x, h_x, k * n);
         util_->SetupMem(&d_y, nullptr, m, false);

         // Launch kernel here
         CudaMatrixVectorMultiply<<(a.cols() + 255) / 256, 256>>(d_a, d_x, d_y, a.cols());

         // Device sync
         util_->SyncDev();

         // Transfer memories back, clear memrory, and return result
         util_->SyncMem(d_a, nullptr, 0, false);
         util_->SyncMem(d_x, nullptr, 0, false);
         util_->SyncMem(d_y, &h_y(0), m);

         return h_y;
       } else if (a.cols() != b.rows()) {
         std::cerr << "Matricies in gpu matrix multiply's sizes aren't compatible"
                   << std::endl;
         exit(1);
       } else if (a.isZero() && b.isZero()) {
         std::cerr << "The maxtrix and the vector are empty"
                   << std::endl;
         exit(1);
       } else if (a.isZero()) {
         std::cerr << "The maxtrix is empty"
                   << std::endl;
         exit(1);
       } else if (b.isZero()) {
         std::cerr << "The vector is empty"
                   << std::endl;
         exit(1);
       } else {
         std::cerr << "Unknown error"
                   << std::endl;
         exit(1);
       }
     }
};

template <typename T>
GpuUtil<T> *CudaMatrixVectorMultiply<T>::util_ = GpuUtil<T>::GetInstance();
}  // namespace Nice
#endif  // NEED_CUDA
#endif  // CPP_INCLUDE_CUDA_MATRIX_VECTOR_MULTIPLY_H_
