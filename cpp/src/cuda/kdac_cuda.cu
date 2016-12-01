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

#define NEED_CUDA
#ifdef NEED_CUDA

#include "include/kdac_cuda.h"
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

//template<typename T>
//void GPUGenPhiCoeff(T *waw_matrix_d_, T *waf_matrix_d_, T *faf_matrix_d_,
//                    T *w_l_d, T *gradient_d) {
//
//}
//
//template
//void GPUGenPhiCoeff<float>(float *waw_matrix_d_, float *waf_matrix_d_, float *faf_matrix_d_,
//                           float *w_l_d, float *gradient_d);


template <typename T>
T* CUDAMallocAndCpy(const Matrix<T> &mat) {
  int n = mat.cols() * mat.rows();
  const T *h_mat = &mat(0);
  T *d_mat;
  gpuErrchk(cudaMalloc(&d_mat, n * sizeof(T)));
  std::cout << "allocating " << n * sizeof(T) << " bytes." << std::endl;
  gpuErrchk(cudaMemcpy(d_mat, h_mat, n * sizeof(T), cudaMemcpyHostToDevice));
  return d_mat;
}
// Template explicit instantiation
template
float* CUDAMallocAndCpy<float>(const Matrix<float> &mat);
template
double* CUDAMallocAndCpy<double>(const Matrix<double> &mat);


template <typename T>
T* CUDAMallocAndCpy(const Vector <T> &vec) {
  int n = vec.size();
  const T *h_vec = &vec(0);
  T *d_vec;
  gpuErrchk(cudaMalloc(&d_vec, n * sizeof(T)));
  std::cout << "allocating " << n * sizeof(T) << " bytes." << std::endl;
  gpuErrchk(cudaMemcpy(d_vec, h_vec, n * sizeof(T), cudaMemcpyHostToDevice));
  return d_vec;
}

template <typename T>
__global__ void GPUGenPhiCoeffKernel(T )

template<typename T>
void GPUGenPhiCoeff(T *a) {
  std::cout << "in GPUGenPhiCoeff" << std::endl;

}

template
float* CUDAMallocAndCpy<float>(const Vector<float> &vec);
template
double* CUDAMallocAndCpy<double>(const Vector<double> &vec);
template
void GPUGenPhiCoeff<float>(float *a);
template
void GPUGenPhiCoeff<double>(double *a);

}

#endif  // NEED_CUDA