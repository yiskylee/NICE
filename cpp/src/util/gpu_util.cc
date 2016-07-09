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

#ifdef NEED_CUDA

#include "include/gpu_util.h"

namespace Nice {

//
// Helper functions
//
void gpuAssert(cudaError_t code, const char *file,
               int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
    if (abort) { exit(code); }
    }
}

void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

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
           int * devInfo) {
  return cusolverDnSgesvd(solver_handle,
                          'A', 'A', M, N, d_A, M, d_S,
                          d_U, M, d_V, N, work, work_size,
                          NULL, devInfo);
}

cusolverStatus_t GpuSvd(cusolverDnHandle_t solver_handle,
           int M,
           int N,
           double * d_A,
           double * d_S,
           double * d_U,
           double * d_V,
           double * work,
           int work_size,
           int * devInfo) {
  return cusolverDnDgesvd(solver_handle,
                         'A', 'A', M, N, d_A, M, d_S,
                          d_U, M, d_V, N, work, work_size,
                          NULL, devInfo);
}

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
                                  float *y, int incy) {
  return cublasSgemv(handle, trans, m, n, alpha,
                     A, lda, x, incx, beta, y, incy);
}

cublasStatus_t GpuMatrixVectorMul(cublasHandle_t handle,
                                  cublasOperation_t trans,
                                  int m, int n,
                                  const double *alpha,
                                  const double *A, int lda,
                                  const double *x, int incx,
                                  const double *beta,
                                  double *y, int incy) {
  return cublasDgemv(handle, trans, m, n, alpha,
                     A, lda, x, incx, beta, y, incy);
}

}  // namespace Nice
#endif  // Need Cuda
