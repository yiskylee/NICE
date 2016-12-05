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

#define NEED_CUDA
#ifdef NEED_CUDA

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <iostream>
#include <memory>

namespace Nice {

//
// Helper macros
//
#define CUDA_CALL(x) \
do {\
  cudaError_t ret = x;\
  if (ret != cudaSuccess) {\
    std::cout << "CUDA Error at " << __FILE__ << __LINE__ << std::endl;\
    std::cout << cudaGetErrorString(ret) << std::endl;\
    exit(EXIT_FAILURE);\
  }\
} while (0)\

#define CURAND_CALL(x) \
do {\
  if ((x) != CURAND_STATUS_SUCCESS) {\
    std::cout << "CURAND Error at " << __FILE__ << __LINE__;\
    exit(EXIT_FAILURE);\
  }\
} while (0)\

#define CUBLAS_CALL(x) \
do {\
  if ((x) != CUBLAS_STATUS_SUCCESS) {\
    std::cout << "CUBLAS Error at " << __FILE__ << __LINE__;\
    exit(EXIT_FAILURE);\
  }\
} while (0)\

#define CUSOLVER_CALL(x) \
do {\
  if ((x) != CUSOLVER_STATUS_SUCCESS) {\
    std::cout << "CUSOLVER Error at " << __FILE__ << __LINE__;\
    exit(EXIT_FAILURE);\
  }\
} while (0)\


//
// GPU utilities
//
template <typename T>
class GpuUtil {
 private:
  cusolverDnHandle_t solver_handle_;
  cublasHandle_t blas_handle_;

  GpuUtil() {
    CUSOLVER_CALL(cusolverDnCreate(&solver_handle_));
    CUBLAS_CALL(cublasCreate(&blas_handle_));
  }

  static std::unique_ptr<GpuUtil> instance_;

 public:
  static GpuUtil *GetInstance() {
    if (instance_.get())
      return instance_.get();
    instance_.reset(new GpuUtil());
    return instance_.get();
  }
  ~GpuUtil() {
    CUSOLVER_CALL(cusolverDnDestroy(solver_handle_));
    CUBLAS_CALL(cublasDestroy(blas_handle_));
  }
  cusolverDnHandle_t GetSolverHandle() {
    return solver_handle_;
  }

  cublasHandle_t GetBlasHandle() {
    return blas_handle_;
  }

  void SetupMem(T **dev, const T *host, int size, bool copy = true) {
    // Create memory
    CUDA_CALL(cudaMalloc(dev, size * sizeof(T)));

    // Copy memory over to device
    if (copy)
      CUDA_CALL(cudaMemcpy(*dev, host, size * sizeof(T),
        cudaMemcpyHostToDevice));
    else
      CUDA_CALL(cudaMemset(*dev, 0, size * sizeof(T)));
  }
  void SyncMem(T *dev, T *host, int size, bool copy = true) {
    // Copy memory over to device
    if (copy)
      CUDA_CALL(cudaMemcpy(host, dev, size * sizeof(T),
        cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CALL(cudaFree(dev));
  }

  void SetupIntMem(int **dev, const int *host, int size, bool copy = true) {
    // Create memory
    CUDA_CALL(cudaMalloc(dev, size * sizeof(int)));

    // Copy memory over to device
    if (copy)
      CUDA_CALL(cudaMemcpy(*dev, host, size * sizeof(int),
        cudaMemcpyHostToDevice));
    else
      CUDA_CALL(cudaMemset(*dev, 0, size * sizeof(int)));
  }
  void SyncIntMem(int *dev, int *host, int size, bool copy = true) {
    // Copy memory over to device
    if (copy)
      CUDA_CALL(cudaMemcpy(host, dev, size * sizeof(int),
        cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CALL(cudaFree(dev));
  }

  void SyncDev() {
    CUDA_CALL(cudaDeviceSynchronize());
  }
};

template <typename T>
std::unique_ptr<GpuUtil<T>> GpuUtil<T>::instance_ = nullptr;

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
           int * devInfo);

void GpuSvd(cusolverDnHandle_t solver_handle,
           int M,
           int N,
           double * d_A,
           double * d_S,
           double * d_U,
           double * d_V,
           double * work,
           int work_size,
           int * devInfo);

void GpuGetLUDecompWorkspace(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    float *A,
                                    int lda,
                                    int *Lwork);

void GpuGetLUDecompWorkspace(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    double *A,
                                    int lda,
                                    int *Lwork);

void GpuLUDecomposition(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    float *A,
                                    int lda,
                                    float *Workspace,
                                    int *devIpiv, int *devInfo);

void GpuLUDecomposition(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    double *A,
                                    int lda,
                                    double *Workspace,
                                    int *devIpiv, int *devInfo);

void GpuLinearSolver(cusolverDnHandle_t handle,
                                 cublasOperation_t trans,
                                 int n,
                                 int nrhs,
                                 const float *A,
                                 int lda,
                                 const int *devIpiv,
                                 float *B,
                                 int ldb,
                                 int *devInfo);

void GpuLinearSolver(cusolverDnHandle_t handle,
                                 cublasOperation_t trans,
                                 int n,
                                 int nrhs,
                                 const double *A,
                                 int lda,
                                 const int *devIpiv,
                                 double *B,
                                 int ldb,
                                 int *devInfo);

void GpuLuWorkspace(cusolverDnHandle_t handle,
                                int m,
                                int n,
                                float *a,
                                int *worksize);

void GpuLuWorkspace(cusolverDnHandle_t handle,
                                int m,
                                int n,
                                double *a,
                                int *worksize);

void GpuDeterminant(cusolverDnHandle_t handle,
                                int m,
                                int n,
                                float *a,
                                float *workspace,
                                int *devIpiv,
                                int *devInfo);

void GpuDeterminant(cusolverDnHandle_t handle,
                                int m,
                                int n,
                                double *a,
                                double *workspace,
                                int *devIpiv,
                                int *devInfo);

//
// Cublas wraper functions
//
void GpuMatrixVectorMul(cublasHandle_t handle,
                                  cublasOperation_t trans,
                                  int m, int n,
                                  const float *alpha,
                                  const float *A, int lda,
                                  const float *x, int incx,
                                  const float *beta,
                                  float *y, int incy);

void GpuMatrixVectorMul(cublasHandle_t handle,
                                  cublasOperation_t trans,
                                  int m, int n,
                                  const double *alpha,
                                  const double *A, int lda,
                                  const double *x, int incx,
                                  const double *beta,
                                  double *y, int incy);
void GpuMatrixScalarMul(cublasHandle_t handle,
                                  int n,
                                  const float &scalar,
                                  float *a);

void GpuMatrixScalarMul(cublasHandle_t handle,
                                  int n,
                                  const double &scalar,
                                  double *a);

void GpuMatrixMatrixMul(cublasHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  float *a,
                                  float *b,
                                  float *c);

void GpuMatrixMatrixMul(cublasHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  double *a,
                                  double *b,
                                  double *c);

void GpuMatrixAdd(cublasHandle_t handle,
                            int m,
                            int n,
                            const float *alpha,
                            const float *A, int lda,
                            const float *beta,
                            const float *B, int ldb,
                            float *C, int ldc);

void GpuMatrixAdd(cublasHandle_t handle,
                            int m,
                            int n,
                            const double *alpha,
                            const double *A, int lda,
                            const double *beta,
                            const double *B, int ldb,
                            double *C, int ldc);

void GpuMatrixMatrixSub(cublasHandle_t handle,
                                  int m,
                                  int n,
                                  const float *alpha,
                                  float *a, int lda,
                                  const float *beta,
                                  float *b, int ldb,
                                  float *c, int ldc);

void GpuMatrixMatrixSub(cublasHandle_t handle,
                                  int m,
                                  int n,
                                  const double *alpha,
                                  double *a, int lda,
                                  const double *beta,
                                  double *b, int ldb,
                                  double *c, int ldc);

void GpuVectorVectorDot(cublasHandle_t handle,
                                  int n,
                                  float *a,
                                  float *b,
                                  float *c);
void GpuVectorVectorDot(cublasHandle_t handle,
                                  int n,
                                  double *a,
                                  double *b,
                                  double *c);

void GpuFrobeniusNorm(cublasHandle_t handle,
                                int n,
                                int incx,
                                float * a,
                                float * c);

void GpuFrobeniusNorm(cublasHandle_t handle,
                                int n,
                                int incx,
                                double * a,
                                double * c);
}  // namespace Nice

#endif  // NEED_CUDA
#endif  // CPP_INCLUDE_GPU_UTIL_H_
