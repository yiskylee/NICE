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
template <typename T>
void GPUGenPhiCoeff(const T *w_l_d,
                    const T *gradient_d,
                    const T *a_matrices_d,
                    const CUBLASParams &params,
                    const int n,
                    const int d,
                    T *a_mul_w_d,
                    T *a_mul_grad_d,
                    T *waw_matrix,
                    T *waf_matrix,
                    T *faf_matrix);

template<typename T>
void GPUGenAMatrices(const T *x_matrix_d,
                     const CUBLASParams &params,
                     const int n,
                     const int d,
                     T *delta_ijs_d,
                     T *a_matrices_d);

template<typename T>
void GPUGenPhi(const T alpha,
               const T sqrt_one_minus_alpha,
               const T denom,
               const T *waw_matrix_d,
               const T *waf_matrix_d,
               const T *faf_matrix_d,
               const T *gamma_matrix_d,
               const int n,
               const int d,
               const bool w_l_changed,
               T *phi_of_alphas_in_d,
               T *phi_of_zeros_in_d,
               T *phi_of_zero_primes_in_d);
}
#endif  // CPP_INCLUDE_KDAC_IN_CUDA_H_
