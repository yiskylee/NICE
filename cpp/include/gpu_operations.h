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

#ifndef CPP_INCLUDE_GPU_OPERATIONS_H_
#define CPP_INCLUDE_GPU_OPERATIONS_H_
#ifdef NEED_CUDA

#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <unistd.h>
#include <stdexcept>

#include <iostream>

#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"

namespace Nice {

// Abstract class of common matrix operation interface
template <typename T>
class GpuOperations {
 public:
  static cublasStatus_t do_multiply(cublasHandle_t handle, int n,
                                  const float &scalar, float *a) {
    return cublasSscal(handle, n, &scalar, a, 1);
  }
  static cublasStatus_t do_multiply(cublasHandle_t handle, int n,
                                  const double &scalar, double *a) {
    return cublasDscal(handle, n, &scalar, a, 1);
  }
  static cublasStatus_t do_multiply(cublasHandle_t handle, int m, int n, int k,
                                  float *a, float *b, float *c) {
    const float alpha = 1.0; const float beta = 0.0;
    return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         m, n, k, &alpha, a, m, b, k, &beta, c, m);
  }
  static cublasStatus_t do_multiply(cublasHandle_t handle, int m, int n, int k,
                                  double *a, double *b, double *c) {
    const double alpha = 1.0; const double beta = 0.0;
    return cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         m, n, k, &alpha, a, m, b, k, &beta, c, m);
  }
  static cublasStatus_t do_dot(cublasHandle_t handle, int n,
                             float *a, float *b, float *c) {
    return cublasSdot(handle, n, a, 1.0, b, 1.0, c);
  }
  static cublasStatus_t do_dot(cublasHandle_t handle, int n,
                             double *a,  double *b, double *c) {
    return cublasDdot(handle, n, a, 1.0, b, 1.0, c);
  }
  static cusolverStatus_t do_get_det_buffer(cusolverDnHandle_t handle, int m,
                                          int n, float *a, int *worksize) {
    return cusolverDnSgetrf_bufferSize(handle, m, n, a, m, &(*worksize));
  }
  static cusolverStatus_t do_get_det_buffer(cusolverDnHandle_t handle, int m,
                                          int n, double *a, int *worksize) {
    return cusolverDnDgetrf_bufferSize(handle, m, n, a, m, &(*worksize));
  }
  static cusolverStatus_t do_det(cusolverDnHandle_t handle, int m, int n,
                               float *a, float *workspace, int *devIpiv,
                               int *devInfo) {
    return cusolverDnSgetrf(handle, m, n, a, m, workspace, devIpiv, devInfo);
  }
  static cusolverStatus_t do_det(cusolverDnHandle_t handle, int m, int n,
                               double *a, double *workspace, int *devIpiv,
                               int *devInfo) {
    return cusolverDnDgetrf(handle, m, n, a, m, workspace, devIpiv, devInfo);
  }
  static cublasStatus_t do_Frobenius_Norm(cublasHandle_t handle, int n,
                                          const float * a, float * c) {
    return cublasSnrm2(handle, n, a, 1.0, c);
  }
  static cublasStatus_t do_Frobenius_Norm(cublasHandle_t handle, int n,
                                          const double * a, double * c) {
    return cublasDnrm2(handle, n, a, 1.0, c);
  }



  static Matrix<T> Multiply(const Matrix<T> &a, const T &scalar) {
      // Alocate and transfer memories
      int n = a.cols() * a.rows();
      const T * h_a = &a(0);
      Matrix<T> h_c(a.rows(), a.cols());
      T * d_a;  gpuErrchk(cudaMalloc(&d_a, n * sizeof(T)));
      gpuErrchk(cudaMemcpy(d_a, h_a, n * sizeof(T),
                           cudaMemcpyHostToDevice));

      // Set up and do cublas matrix scalar multiply
      cublasStatus_t stat;
      cublasHandle_t  handle;
      cublasCreate(&handle);
      stat = do_multiply(handle, n, scalar, d_a);

      // Error check
      if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "GPU Matrix Scalar Multiply Internal Failure" << std::endl;
        cudaFree(d_a);
        cublasDestroy(handle);
        exit(1);
      }
      cudaDeviceSynchronize();

      // Transfer memories back, clear memories, and return result
      gpuErrchk(cudaMemcpy(&h_c(0, 0), d_a, n * sizeof(T),
                           cudaMemcpyDeviceToHost));
      cudaFree(d_a);
      cublasDestroy(handle);
      return h_c;
  }
  static Matrix<T> Multiply(const Matrix<T> &a, const Matrix<T> &b) {
    if (a.cols() == b.rows()) {  // Check if matricies k vals are equal
      // Allocate and transfer memories
      int m = a.rows();
      int n = b.cols();
      int k = a.cols();

      const T * h_a = &a(0);
      const T * h_b = &b(0);
      Matrix<T> h_c(m, n);

      T * d_a;  gpuErrchk(cudaMalloc(&d_a, m * k * sizeof(T)));
      T * d_b;  gpuErrchk(cudaMalloc(&d_b, k * n * sizeof(T)));
      T * d_c;  gpuErrchk(cudaMalloc(&d_c, m * n * sizeof(T)));

      gpuErrchk(cudaMemcpy(d_a, h_a, m * k * sizeof(T),
                           cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(d_b, h_b, k * n * sizeof(T),
                           cudaMemcpyHostToDevice));

      // Set up and do cublas matrix multiply
      cublasStatus_t stat;
      cublasHandle_t  handle;
      cublasCreate(&handle);
      stat = do_multiply(handle, m, n, k, d_a, d_b, d_c);

      // Error check
      if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "GPU Matrix Matrix Multiply Internal Failure" << std::endl;
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cublasDestroy(handle);
        exit(1);
      }
      cudaDeviceSynchronize();
      // Transfer memories back, clear memrory, and return result
      gpuErrchk(cudaMemcpy(&h_c(0, 0), d_c, m * n * sizeof(T),
                           cudaMemcpyDeviceToHost));
      cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
      cublasDestroy(handle);
      return h_c;
    } else {
      std::cerr << "Matricies in gpu matrix multiply's sizes aren't compatible"
                << std::endl;
      exit(1);
    }
  }
  static Matrix<T> Add(const Matrix<T> &a, const T &scalar);
  static Matrix<T> Add(const Matrix<T> &a, const Matrix<T> &b);
  static Matrix<T> Subtract(const Matrix<T> &a, const T &scalar);
  static Matrix<T> Subtract(const Matrix<T> &a, const Matrix<T> &b);
  static Matrix<T> Inverse(const Matrix<T> &a);
  static Matrix<T> Norm(const int &p = 2, const int &axis = 0);
  static T Determinant(const Matrix<T> &a) {
    int m = a.rows();
    int n = a.cols();
    const T *h_a = &a(0);
    T det;

    // Allocating and transfering memories
    T *d_a;   gpuErrchk(cudaMalloc(&d_a, m * n * sizeof(T)));
    int *devIpiv_h = reinterpret_cast<int *>(malloc(m * n * sizeof(int)));
    int *devIpiv_d; gpuErrchk(cudaMalloc(&devIpiv_d, m * n * sizeof(int)));
    cudaMemset(devIpiv_d, 0, m * n * sizeof(int));
    int devInfo_h = 0;
    int *devInfo_d;   gpuErrchk(cudaMalloc(&devInfo_d, sizeof(int)));
    T *h_c = reinterpret_cast<T *>(malloc(m * n *sizeof(T)));
    gpuErrchk(cudaMemcpy(d_a, h_a, m * n * sizeof(T), cudaMemcpyHostToDevice));

    // Setup and do get LU decomposition buffer
    int work_size = 0;
    cusolverStatus_t stat;
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
    stat = do_get_det_buffer(handle, m, n, d_a, &work_size);

    // Error check
    if (stat != CUSOLVER_STATUS_SUCCESS) {
      std::cout << "Initialization of determinant buffer failed." << std::endl;
      cudaFree(d_a); cudaFree(devIpiv_d); free(devIpiv_h); free(h_c);
      cudaFree(devInfo_d);
      cusolverDnDestroy(handle);
      exit(1);
    }

    // Allocate LU decomposition workspace memory and do LU decomposistion
    T *workspace;    gpuErrchk(cudaMalloc(&workspace, work_size * sizeof(T)));
    stat = do_det(handle, m, n, d_a, workspace, devIpiv_d, devInfo_d);

    // Error check
    gpuErrchk(cudaMemcpy(&devInfo_h, devInfo_d, sizeof(int),
              cudaMemcpyDeviceToHost));
    if (stat != CUSOLVER_STATUS_SUCCESS || devInfo_h != 0) {
      std::cerr << "GPU Determinant Internal Failure" << std::endl;
      cudaFree(d_a); cudaFree(devIpiv_d); free(devIpiv_h); free(h_c);
      cudaFree(devInfo_d); cudaFree(workspace);
      cusolverDnDestroy(handle);
      exit(1);
    }
    cudaDeviceSynchronize();

    // Transfer memories back to host
    gpuErrchk(cudaMemcpy(devIpiv_h, devIpiv_d, m * n * sizeof(int),
              cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_c, d_a, m * n * sizeof(T),
              cudaMemcpyDeviceToHost));
    // Count number of pivot points, if odd multiply determinant by -1
    int cnt = 0;
    for (int i = 0; i < m; ++i) {
       for (int j = 0; j < n; ++j) {
          if (*(devIpiv_h+j+i*n) != 0) cnt = cnt + 1;
       }
    }

    // Determinante is product of U matrix diagonal * (-1)^cnt
    det = *(h_c);
    for (int i = 1; i < m; ++i) {
      det = det * *(h_c + (i * m) + i);
    }
    if (cnt % 2 != 0) det = det * -1;  // if odd multiply by -1

    // Free memories and return answer
    cudaFree(d_a); cudaFree(devIpiv_d); free(devIpiv_h); free(h_c);
    cudaFree(devInfo_d); cudaFree(workspace);
    cusolverDnDestroy(handle);
    return det;
  }
  static T Rank(const Matrix<T> &a);
  static T FrobeniusNorm(const Matrix<T> &a) {
    int m = a.rows();
    int n = a.cols();
    const T * h_a = &a(0);

    // Allocate and transfer memories
    T * h_c = reinterpret_cast<T *>(malloc(sizeof(T)));
    T * d_a;  gpuErrchk(cudaMalloc(&d_a, m * n * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_a, h_a, m * n * sizeof(T), cudaMemcpyHostToDevice));

    // Setup and do Frobenious Norm
    cublasHandle_t  handle;
    cublasCreate(&handle);
    cublasStatus_t stat;
    stat = do_Frobenius_Norm(handle, n * m, d_a, h_c);

    // Error Check
    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "GPU Matrix Frobenius Norm Internal Failure"
                << std::endl;
      cudaFree(d_a); free(h_c);
      cublasDestroy(handle);
      exit(1);
    }
    cudaDeviceSynchronize();

    // Free memories and return answer
    cudaFree(d_a);
    cublasDestroy(handle);
    return *h_c;
  }
  static T Trace(const Matrix<T> &a);
  static T DotProduct(const Vector<T> &a, const Vector<T> &b) {
    int n = a.rows();

    // Allocate and transfer memories
    const T * h_a = &a(0);
    const T * h_b = &b(0);
    T * h_c = reinterpret_cast<T *>(malloc(sizeof(T)));

    T * d_a;  gpuErrchk(cudaMalloc(&d_a, n * sizeof(T)));
    T * d_b;  gpuErrchk(cudaMalloc(&d_b, n * sizeof(T)));

    gpuErrchk(cudaMemcpy(d_a, h_a, n * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, h_b, n * sizeof(T), cudaMemcpyHostToDevice));

    // Setup and do dot product
    cublasHandle_t  handle;
    cublasCreate(&handle);
    cublasStatus_t stat;
    stat = do_dot(handle, n, d_a, d_b, h_c);

    // Error Check
    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "GPU Vector Vector Dot Product Internal Failure"
                << std::endl;
      cudaFree(d_a); cudaFree(d_b);
      cublasDestroy(handle);
      exit(1);
    }
    cudaDeviceSynchronize();

    // Free memories and return result
    cudaFree(d_a); cudaFree(d_b);
    cublasDestroy(handle);
    return *h_c;
  }
  static Matrix<T> OuterProduct(const Vector<T> &a, const Vector<T> &b) {
    if (a.cols() == b.cols()) {
      int m = a.rows();
      int n = b.rows();
      int k = 1;

      // Allocate and transfer memories
      const T * h_a = &a(0);
      const T * h_b = &b(0);
      Matrix<T> h_c(m, n);

      T * d_a;  gpuErrchk(cudaMalloc(&d_a, m * k * sizeof(T)));
      T * d_b;  gpuErrchk(cudaMalloc(&d_b, k * n * sizeof(T)));
      T * d_c;  gpuErrchk(cudaMalloc(&d_c, m * n * sizeof(T)));

      gpuErrchk(cudaMemcpy(d_a, h_a, m * k * sizeof(T),
                           cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(d_b, h_b, k * n * sizeof(T),
                           cudaMemcpyHostToDevice));

      // Setup and do outer product multiply
      cublasStatus_t stat;
      cublasHandle_t  handle;
      cublasCreate(&handle);
      stat = do_multiply(handle, m, n, k, d_a, d_b, d_c);

      // Error check
      if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "GPU Outer Product Internal Failure" << std::endl;
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cublasDestroy(handle);
        exit(1);
      }
      cudaDeviceSynchronize();

      // Transfer results back, clear memories, return answer
      gpuErrchk(cudaMemcpy(&h_c(0, 0), d_c, m * n * sizeof(T),
                           cudaMemcpyDeviceToHost));
      cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
      cublasDestroy(handle);
      return h_c;
    } else {
      std::cerr << "Vectors in gpu outer product's sizes aren't compatible"
                << std::endl;
      exit(1);
    }
  }
};
}  // namespace Nice
#endif  // NEED_CUDA
#endif  // CPP_INCLUDE_GPU_OPERATIONS_H_

