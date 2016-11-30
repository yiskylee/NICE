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

#ifndef CPP_INCLUDE_KDAC_IN_CUDA_H_
#define CPP_INCLUDE_KDAC_IN_CUDA_H_
#define NEED_CUDA
#ifdef NEED_CUDA

#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <unistd.h>
#include <stdexcept>
#include <ctime>

namespace Nice {
template<typename T>
T* CUDAMallocAndCpy(const Matrix<T> &mat);

template <typename T>
T* CUDAMallocAndCpy(const Vector<T> &vec);
//template<typename T>
//void GPUGenPhiCoeff(T *waw_matrix_d_, T *waf_matrix_d_, T *faf_matrix_d_,
//    T *w_l_d, T *gradient_d);
//}

template<typename T>
void GPUGenPhiCoeff(T *a);
}
#endif  // NEED_CUDA
#endif  // CPP_INCLUDE_KDAC_IN_CUDA_H_