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


#include<unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime_api.h>
#include<iostream>
#include "build/include/cuda_runtime.h"
#include "build/include/device_launch_parameters.h"
#include<cusolverDn.h>
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
 public:
  GpuSvdSolver() {}
  void      Compute(const Matrix<T> &A);
  Matrix<T> MatrixU() const              { return u_; }
  Matrix<T> MatrixV() const              { return v_; }
  Vector<T> SingularValues() const       { return s_; }
};
}  // namespace Nice

#endif  // CPP_INCLUDE_GPU_SVD_SOLVER_H_

