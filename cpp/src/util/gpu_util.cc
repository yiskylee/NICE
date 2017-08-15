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

#ifdef CUDA_AND_GPU
#include "include/gpu_util.h"

namespace Nice {

//
// Cusolver wraper functions
//
void GpuSvd(cusolverDnHandle_t solver_handle,
           int M,
           int N,
           float * d_A,
           float * d_S,
           float * d_U,
           float * d_V,
           float * work,
           int work_size,
           int * devInfo) {
  CUSOLVER_CALL(cusolverDnSgesvd(solver_handle,
                          'A', 'A', M, N, d_A, M, d_S,
                          d_U, M, d_V, N, work, work_size,
                          NULL, devInfo));
}

void GpuSvd(cusolverDnHandle_t solver_handle,
           int M,
           int N,
           double * d_A,
           double * d_S,
           double * d_U,
           double * d_V,
           double * work,
           int work_size,
           int * devInfo) {
  CUSOLVER_CALL(cusolverDnDgesvd(solver_handle,
                         'A', 'A', M, N, d_A, M, d_S,
                          d_U, M, d_V, N, work, work_size,
                          NULL, devInfo));
}

void GpuGetLUDecompWorkspace(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    float *A,
                                    int lda,
                                    int *Lwork) {
  CUSOLVER_CALL(cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork));
}

void GpuGetLUDecompWorkspace(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    double *A,
                                    int lda,
                                    int *Lwork) {
  CUSOLVER_CALL(cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork));
}

void GpuLUDecomposition(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    float *A,
                                    int lda,
                                    float *Workspace,
                                    int *devIpiv, int *devInfo) {
  CUSOLVER_CALL(cusolverDnSgetrf(handle, m, n, A, lda,
    Workspace, devIpiv, devInfo));
}

void GpuLUDecomposition(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    double *A,
                                    int lda,
                                    double *Workspace,
                                    int *devIpiv, int *devInfo) {
  CUSOLVER_CALL(cusolverDnDgetrf(handle, m, n, A, lda,
    Workspace, devIpiv, devInfo));
}

void GpuLinearSolver(cusolverDnHandle_t handle,
                                 cublasOperation_t trans,
                                 int n,
                                 int nrhs,
                                 const float *A,
                                 int lda,
                                 const int *devIpiv,
                                 float *B,
                                 int ldb,
                                 int *devInfo) {
  CUSOLVER_CALL(cusolverDnSgetrs(handle, trans, n, nrhs, A, lda,
                          devIpiv, B, ldb, devInfo));
}

void GpuLinearSolver(cusolverDnHandle_t handle,
                                 cublasOperation_t trans,
                                 int n,
                                 int nrhs,
                                 const double *A,
                                 int lda,
                                 const int *devIpiv,
                                 double *B,
                                 int ldb,
                                 int *devInfo) {
  CUSOLVER_CALL(cusolverDnDgetrs(handle, trans, n, nrhs, A, lda,
                          devIpiv, B, ldb, devInfo));
}
void GpuLuWorkspace(cusolverDnHandle_t handle,
                                int m,
                                int n,
                                float *a,
                                int *worksize) {
  CUSOLVER_CALL(cusolverDnSgetrf_bufferSize(handle, m, n, a, m, &(*worksize)));
}
void GpuLuWorkspace(cusolverDnHandle_t handle,
                                int m,
                                int n,
                                double *a,
                                int *worksize) {
  CUSOLVER_CALL(cusolverDnDgetrf_bufferSize(handle, m, n, a, m, &(*worksize)));
}
void GpuDeterminant(cusolverDnHandle_t handle,
                                int m,
                                int n,
                                float *a,
                                float *workspace,
                                int *devIpiv,
                                int *devInfo) {
  CUSOLVER_CALL(cusolverDnSgetrf(handle, m, n, a, m,
    workspace, devIpiv, devInfo));
}
void GpuDeterminant(cusolverDnHandle_t handle,
                                int m,
                                int n,
                                double *a,
                                double *workspace,
                                int *devIpiv,
                                int *devInfo) {
  CUSOLVER_CALL(cusolverDnDgetrf(handle, m, n, a, m,
    workspace, devIpiv, devInfo));
}

//
// Cublas wraper functions
//
void GpuMatrixScalarMul(cublasHandle_t handle,
                                  int n,
                                  const float &scalar,
                                  float *a) {
  CUBLAS_CALL(cublasSscal(handle, n, &scalar, a, 1));
}

void GpuMatrixScalarMul(cublasHandle_t handle,
                                  int n,
                                  const double &scalar,
                                  double *a) {
  CUBLAS_CALL(cublasDscal(handle, n, &scalar, a, 1));
}

void GpuMatrixVectorMul(cublasHandle_t handle, cublasOperation_t norm,
                                  int m,
                                  int n,
                                  const float *alpha,
                                  const float *a,
                                  int lda,
                                  const float *x,
                                  int incx,
                                  const float *beta,
                                  float *y,
                                  int incy) {
  CUBLAS_CALL(cublasSgemv(handle, norm,
                     m, n, alpha, a, lda, x, incx, beta, y, incy));
}

void GpuMatrixVectorMul(cublasHandle_t handle, cublasOperation_t norm,
                                  int m,
                                  int n,
                                  const double *alpha,
                                  const double *a,
                                  int lda,
                                  const double *x,
                                  int incx,
                                  const double *beta,
                                  double *y,
                                  int incy) {
  CUBLAS_CALL(cublasDgemv(handle, norm,
                     m, n, alpha, a, lda, x, incx, beta, y, incy));
}

void GpuMatrixMatrixMul(cublasHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  float *a,
                                  float *b,
                                  float *c) {
  const float alpha = 1.0; const float beta = 0.0;
  CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, k, &alpha, a, m, b, k, &beta, c, m));
}

void GpuMatrixMatrixMul(cublasHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  double *a,
                                  double *b,
                                  double *c) {
  const double alpha = 1.0; const double beta = 0.0;
  CUBLAS_CALL(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, k, &alpha, a, m, b, k, &beta, c, m));
}

void GpuMatrixAdd(cublasHandle_t handle,
                            int m,
                            int n,
                            const float *alpha,
                            const float *A, int lda,
                            const float *beta,
                            const float *B, int ldb,
                            float *C, int ldc) {
  CUBLAS_CALL(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, alpha, A, lda, beta, B, ldb, C, ldc));
}

void GpuMatrixAdd(cublasHandle_t handle,
                            int m,
                            int n,
                            const double *alpha,
                            const double *A, int lda,
                            const double *beta,
                            const double *B, int ldb,
                            double *C, int ldc) {
  CUBLAS_CALL(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, alpha, A, lda, beta, B, ldb, C, ldc));
}

void GpuMatrixMatrixSub(cublasHandle_t handle, int m, int n,
                                  const float *alpha, float *a, int lda,
                                  const float *beta, float *b, int ldb,
                                  float *c, int ldc ) {
  CUBLAS_CALL(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, alpha, a, lda,
                     beta, b, ldb, c, ldc));
}

void GpuMatrixMatrixSub(cublasHandle_t handle, int m, int n,
                                  const double *alpha, double *a, int lda,
                                  const double *beta, double *b, int ldb,
                                  double *c, int ldc ) {
  CUBLAS_CALL(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, alpha, a, lda,
                     beta, b, ldb, c, ldc));
}

void GpuVectorVectorDot(cublasHandle_t handle,
                                  int n,
                                  float *a,
                                  float *b,
                                  float *c) {
  CUBLAS_CALL(cublasSdot(handle, n, a, 1.0, b, 1.0, c));
}

void GpuVectorVectorDot(cublasHandle_t handle,
                                  int n,
                                  double *a,
                                  double *b,
                                  double *c) {
  CUBLAS_CALL(cublasDdot(handle, n, a, 1.0, b, 1.0, c));
}

void GpuFrobeniusNorm(cublasHandle_t handle,
                                int n,
                                int incx,
                                float *a,
                                float *c) {
  CUBLAS_CALL(cublasSnrm2(handle, n, a, incx, c));
}

void GpuFrobeniusNorm(cublasHandle_t handle,
                                int n,
                                int incx,
                                double *a,
                                double *c) {
  CUBLAS_CALL(cublasDnrm2(handle, n, a, incx, c));
}

}  // namespace Nice
#endif  // CUDA_AND_GPU
