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
#ifndef CPP_INCLUDE_GPU_UTIL_H_
#define CPP_INCLUDE_GPU_UTIL_H_

#ifdef NEED_CUDA

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusolverDn.h>

#include <iostream>

namespace Nice {

//
// Helper functions
//
void gpuAssert(cudaError_t, const char *, int, bool);
void gpuErrchk(cudaError_t);

//
// Cusolver wraper functions
//
cusolverStatus_t GpuSvd(cusolverDnHandle_t solver_handle,
           int M,
           int N,
           float * d_A,
           float * d_S,
           float * d_U,
           float * d_V,
           float * work,
           int work_size,
           int * devInfo);

cusolverStatus_t GpuSvd(cusolverDnHandle_t solver_handle,
           int M,
           int N,
           double * d_A,
           double * d_S,
           double * d_U,
           double * d_V,
           double * work,
           int work_size,
           int * devInfo);

cusolverStatus_t GpuGetLUDecompWorkspace(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    float *A,
                                    int lda,
                                    int *Lwork);

cusolverStatus_t GpuGetLUDecompWorkspace(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    double *A,
                                    int lda,
                                    int *Lwork);

cusolverStatus_t GpuLUDecomposition(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    float *A,
                                    int lda,
                                    float *Workspace,
                                    int *devIpiv, int *devInfo);

cusolverStatus_t GpuLUDecomposition(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    double *A,
                                    int lda,
                                    double *Workspace,
                                    int *devIpiv, int *devInfo);

cusolverStatus_t GpuLinearSolver(cusolverDnHandle_t handle,
                                 cublasOperation_t trans,
                                 int n,
                                 int nrhs,
                                 const float *A,
                                 int lda,
                                 const int *devIpiv,
                                 float *B,
                                 int ldb,
                                 int *devInfo);

cusolverStatus_t GpuLinearSolver(cusolverDnHandle_t handle,
                                 cublasOperation_t trans,
                                 int n,
                                 int nrhs,
                                 const double *A,
                                 int lda,
                                 const int *devIpiv,
                                 double *B,
                                 int ldb,
                                 int *devInfo);


//
// Cublas wraper functions
//
cublasStatus_t GpuMatrixVectorMul(cublasHandle_t handle,
                                  cublasOperation_t trans,
                                  int m, int n,
                                  const float *alpha,
                                  const float *A, int lda,
                                  const float *x, int incx,
                                  const float *beta,
                                  float *y, int incy);

cublasStatus_t GpuMatrixVectorMul(cublasHandle_t handle,
                                  cublasOperation_t trans,
                                  int m, int n,
                                  const double *alpha,
                                  const double *A, int lda,
                                  const double *x, int incx,
                                  const double *beta,
                                  double *y, int incy);

}  // namespace Nice

#endif  // NEED_CUDA
#endif  // CPP_INCLUDE_GPU_UTIL_H_
