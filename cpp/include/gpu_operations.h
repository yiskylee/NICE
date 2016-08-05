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

#include <stdlib.h>
#include <time.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <unistd.h>
#include <stdexcept>
#include <ctime>

#include <iostream>

#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"
#include "include/gpu_svd_solver.h"

namespace Nice {

// Abstract class of common matrix operation interface
template <typename T>
class GpuOperations {
 public:
  ///  static Matrix<T> Multiply(const Matrix<T> &a, const T &scalar)
  ///                   calculates the element by element product of
  ///                   a matrix and a scalar.
  ///
  ///  \param a
  ///  const Matrix<T> &a : The m x n matrix to be multiplied by the scalar
  ///  \param b
  ///  const T &scalar : The scalar to multiply the matirx
  ///
  ///  \return
  ///   returns the resulting matrix of a[i] = a[i] * scalar
  ///
  ///  \sa
  ///  \ref static Matrix<T> Multiply(const Matrix<T> &a, const Matrix<T> &b)
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
      stat = GpuMatrixScalarMul(handle, n, scalar, d_a);

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

  ///  static Matrix<T> Multiply(const Matrix<T> &a, const Matrix<T> &b)
  ///                   calculates the matrix multiplication of two matrices.
  ///
  ///  \param a
  ///  const Matrix<T> &a : The m x k input matrix
  ///  \param b
  ///  const Matrix<T> &b : The k x n input matrix
  ///
  ///  \return
  ///  returns the resulting matrix of
  ///  c[i,j] = sum(l=0->l=k){a[i,l] * b[l,j]}
  ///
  ///  \sa
  ///  \ref static Matrix<T> Multiply(const Matrix<T> &a, const T &scalar)
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
      stat = GpuMatrixMatrixMul(handle, m, n, k, d_a, d_b, d_c);

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

  /// This function calculates the sum of the input Matrix and scalar
  ///
  /// \param a
  /// Input Matrix
  /// \param scalar
  /// Input scalar of type T
  ///
  /// \return
  /// This function returns a Matrix of type T
  static Matrix<T> Add(const Matrix<T> &a, const T &scalar) {
    int m = a.rows();
    int n = a.cols();
    int lda = m;
    int ldb = m;
    int ldc = m;

    T alpha = 1.0;
    T beta = 1.0;

    const T * h_a = &a(0);
    Matrix<T> b(m, n);
    b = Matrix<T>::Constant(m, n, scalar);
    const T * h_b = &b(0);
    Matrix<T> h_c(m, n);

    T * d_a;  gpuErrchk(cudaMalloc(&d_a, m * n * sizeof(T)));
    T * d_b;  gpuErrchk(cudaMalloc(&d_b, m * n * sizeof(T)));
    T * d_c;  gpuErrchk(cudaMalloc(&d_c, m * n * sizeof(T)));

    gpuErrchk(cudaMemcpy(d_a, h_a, m * n * sizeof(T),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, h_b, m * n * sizeof(T),
                         cudaMemcpyHostToDevice));

    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);
    stat = GpuMatrixAdd(handle,
                       m, n,
                       &alpha,
                       d_a, lda,
                       &beta,
                       d_b, ldb,
                       d_c, ldc);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "GPU Matrix Add Internal Failure" << std::endl;
      cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
      cublasDestroy(handle);
      exit(1);
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaMemcpy(&h_c(0, 0), d_c, m * n * sizeof(T),
                         cudaMemcpyDeviceToHost));
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cublasDestroy(handle);
    return h_c;
  }

  /// This function calculates the sum of the input Matricies
  ///
  /// \param a
  /// Input Matrix 1
  /// \param b
  /// Input Matrix 2
  ///
  /// \return
  /// This function returns a Matrix of type T
  static Matrix<T> Add(const Matrix<T> &a, const Matrix<T> &b) {
    if (a.rows() == b.rows() && a.cols() == b.cols()) {
      int m = a.rows();
      int n = b.cols();
      int lda = m;
      int ldb = n;
      int ldc = m;

      T alpha = 1.0;
      T beta = 1.0;

      const T * h_a = &a(0);
      const T * h_b = &b(0);
      Matrix<T> h_c(m, n);

      T * d_a;  gpuErrchk(cudaMalloc(&d_a, m * n * sizeof(T)));
      T * d_b;  gpuErrchk(cudaMalloc(&d_b, m * n * sizeof(T)));
      T * d_c;  gpuErrchk(cudaMalloc(&d_c, m * n * sizeof(T)));

      gpuErrchk(cudaMemcpy(d_a, h_a, m * n * sizeof(T),
                           cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(d_b, h_b, m * n * sizeof(T),
                           cudaMemcpyHostToDevice));

      cublasStatus_t stat;
      cublasHandle_t handle;
      cublasCreate(&handle);
      stat = GpuMatrixAdd(handle,
                         m, n,
                         &alpha,
                         d_a, lda,
                         &beta,
                         d_b, ldb,
                         d_c, ldc);
      if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "GPU Matrix Add Internal Failure" << std::endl;
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cublasDestroy(handle);
        exit(1);
      }
      cudaDeviceSynchronize();
      gpuErrchk(cudaMemcpy(&h_c(0, 0), d_c, m * n * sizeof(T),
                           cudaMemcpyDeviceToHost));
      cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
      cublasDestroy(handle);
      return h_c;
    } else {
      std::cerr << "Matricies in gpu matrix add's sizes aren't compatible"
                << std::endl;
      exit(1);
    }
  }
  static Matrix<T> Subtract(const Matrix<T> &a, const T &scalar);

  /// This function subtracts one matrix from another and returns the resulting
  /// matrix.
  ///
  /// \param a
  /// Input Matrix 1
  /// \param b
  /// Input Matrix 2
  ///
  /// \return
  /// A matrix that reflects the difference of matricies a and b.
  ///
  /// \sa
  /// \ref Subtract(const Matrix<T> &a, const T &scalar)
  static Matrix<T> Subtract(const Matrix<T> &a, const Matrix<T> &b) {
    // If the matricies aren't identical sizes then we cannot subtract them.
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
      std::cerr << "Matricies are not the same size" << std::endl;
      exit(1);

    // If the matricies are empty this function should not run.
    } else if (a.rows() == 0) {
      std::cerr << "Matricies are empty" << std::endl;
      exit(1);

    // Otherwise, everything should run fine.
    } else {
      // Allocate and Transfer Memory
      int m = a.rows();
      int n = a.cols();
      int lda = m;
      int ldb = n;
      int ldc = m;
      T alpha = 1.0;
      T beta = -1.0;

      const T * h_a = &a(0);
      const T * h_b = &b(0);
      Matrix<T> h_c(m, n);

      T * d_a; gpuErrchk(cudaMalloc(&d_a, m * n * sizeof(T)));
      T * d_b; gpuErrchk(cudaMalloc(&d_b, m * n * sizeof(T)));
      T * d_c; gpuErrchk(cudaMalloc(&d_c, m * n * sizeof(T)));

      gpuErrchk(cudaMemcpy(d_a, h_a, m * n * sizeof(T),
                           cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(d_b, h_b, m * n * sizeof(T),
                           cudaMemcpyHostToDevice));

      // Set up and do cublas matrix subtract
      cublasStatus_t stat;
      cublasHandle_t handle;
      cublasCreate(&handle);
      stat = GpuMatrixMatrixSub(handle, m, n, &alpha, d_a, lda,
                                &beta, d_b, ldb, d_c, ldc);

      // Error Check
      if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "GPU Matrix Subtract Internal Failure" << std::endl;
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cublasDestroy(handle);
        exit(1);
      }
      cudaDeviceSynchronize();
      // Transfer memory back and clear it
      gpuErrchk(cudaMemcpy(&h_c(0, 0), d_c, m * n * sizeof(T),
                           cudaMemcpyDeviceToHost));
      cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
      cublasDestroy(handle);
      // Return result
      return h_c;
    }
  }
  /// Return the inversion of a matrix
  /// Computation all done in GPU
  ///
  /// \param a
  /// An arbitrary matrix
  ///
  /// \return
  /// Inversed matrix
  static Matrix<T> Inverse(const Matrix<T> &a) {
    // Sanity Check
    if (a.rows() != a.cols()) {
      std::cerr << "Matrix is singular" << std::endl;
      exit(1);
    }

    // Get the row/column number
    int n = a.rows();

    // Create host memory
    const T *h_a = &a(0);

    // Create device memory needed
    T *d_a;
    int *d_ipiv;
    int *d_info;
    gpuErrchk(cudaMalloc(&d_a, n * n * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_ipiv, n * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_info, sizeof(T)));

    // Copy host memory over to device
    gpuErrchk(cudaMemcpy(d_a, h_a, n * n * sizeof(T),
                         cudaMemcpyHostToDevice));

    // Setup cusolver parameters
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
    cusolverStatus_t stat;
    int lda = n;
    int nrhs = n;
    int ldb = lda;

    // Setup workspace for LU decomposition
    int workspace_size;
    stat = GpuGetLUDecompWorkspace(handle, n, n, d_a, lda, &workspace_size);
    if (stat != CUSOLVER_STATUS_SUCCESS) {
      std::cerr << "LU decomposition: Workspace allocation failed"
                << std::endl;
      cudaFree(d_a);
      cudaFree(d_ipiv);
      cudaFree(d_info);
      exit(1);
    }

    T *workspace;
    gpuErrchk(cudaMalloc(&workspace, workspace_size * sizeof(T)));

    // Do LU docomposition
    stat = GpuLUDecomposition(handle, n, n, d_a, lda,
                              workspace, d_ipiv, d_info);
    if (stat != CUSOLVER_STATUS_SUCCESS) {
      std::cerr << "LU decomposition: decomposition failed"
                << std::endl;
      cudaFree(d_a);
      cudaFree(d_ipiv);
      cudaFree(d_info);
      cudaFree(workspace);
      exit(1);
    }
    cudaFree(workspace);

    // Create an identity matrix
    Matrix<T> b = Matrix<T>::Identity(n, n);

    // Create host memory
    T *h_b = &b(0);

    // Create device memory needed
    T *d_b;
    gpuErrchk(cudaMalloc(&d_b, n * n * sizeof(T)));

    // Copy host memory over to device
    gpuErrchk(cudaMemcpy(d_b, h_b, n * n * sizeof(T),
                         cudaMemcpyHostToDevice));

    // Do lineaer solver
    stat = GpuLinearSolver(handle, CUBLAS_OP_N, n, nrhs, d_a, lda, d_ipiv, d_b,
                           ldb, d_info);
    if (stat != CUSOLVER_STATUS_SUCCESS) {
      std::cerr << "Linear solver failed"
                << std::endl;
      cudaFree(d_a);
      cudaFree(d_ipiv);
      cudaFree(d_info);
      cudaFree(d_b);
      exit(1);
    }

    // Copy device result over to host
    gpuErrchk(cudaMemcpy(h_b, d_b, n * n * sizeof(T),
                         cudaMemcpyDeviceToHost));

    // Synchonize and clean up
    cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_ipiv);
    cudaFree(d_info);
    cudaFree(d_b);

    // Destroy the handle
    cusolverDnDestroy(handle);

    // Return the result
    return b;
  }

  ///  static Vector<T> Norm(const Matrix<T> &a, const int &p =2,
  ///                        const int &axis = 0)
  ///                   calculates the norm of each coloumn of a matrix and
  ///                   stores the result in that column's corresponding
  ///                   index in a vector
  ///
  ///  \param a
  ///  const Matrix<T> &a : The input matrix to have the norm found
  ///  \param b
  ///  const int &p : Sets which Lx norm to calcualte. Here set to 2 for L2 norm
  ///  \param c
  ///  const int &axis : Sets row or column major order. Here set to 0 for
  ///                    column major order.
  ///
  ///  \return
  ///  returns the resulting vector of v[i] = norm(j=0->j=m){a[j,i]}
  ///
  ///  \sa
  ///  \ref static T FrobeniousNorm(const Matrix<T> &a)
  static Vector<T> Norm(const Matrix<T> a, const int &p = 2,
                        const int &axis = 0) {
    int m = a.rows();
    int n = a.cols();
    int incx = 1;
    Vector<T> c(n);
    const T * h_a = &a(0);

    // Allocate and transfer memories
    T * h_c = reinterpret_cast<T *>(malloc(sizeof(T)));
    T * d_a;  gpuErrchk(cudaMalloc(&d_a, m * n * sizeof(T)));
    T * d_t;  gpuErrchk(cudaMalloc(&d_t, m *     sizeof(T)));
    gpuErrchk(cudaMemcpy(d_a, h_a, m * n * sizeof(T), cudaMemcpyHostToDevice));

    // Setup and do Frobenious Norm
    cublasHandle_t  handle;
    cublasCreate(&handle);
    cublasStatus_t stat;
    int iter = 0;
    for (int i = 0; i < n; ++i) {
      gpuErrchk(cudaMemcpy(d_t, d_a + i * m, m * sizeof(T),
                           cudaMemcpyDeviceToDevice));
      stat = GpuFrobeniusNorm(handle, m, incx, d_t, h_c);
      // Error Check
      if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "GPU Matrix Norm Internal Failure"
                  << std::endl;
        cudaFree(d_a); free(h_c);
        cublasDestroy(handle);
        exit(1);
      }
      cudaDeviceSynchronize();
      c(iter) = *h_c;
      iter++;
    }
    // Free memories and return answer
    cudaFree(d_a); free(h_c);
    cublasDestroy(handle);
    return c;
  }

  ///  static T Determinant(const Matrix<T> &a)
  ///           Calculates the determinant of input matrix a using
  ///           LU decomposition
  ///
  ///  \param a
  ///  const Matrix<T> &a : The m x n matrix to have determinant found
  ///
  ///  \return
  ///  returns the result of det = product(i == j){U(i,j)} * (-1)^n
  ///  where U is the upper triangular matrix produced by doing LU
  ///  decomposition on matrix a, product(i == j){U(i,j)} means
  ///  means multiplying the diagonal elements of U, and n is the
  ///  number of pivots needed to calculate U.
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
    stat = GpuLuWorkspace(handle, m, n, d_a, &work_size);

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
    stat = GpuDeterminant(handle, m, n, d_a, workspace, devIpiv_d, devInfo_d);

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

    // Count number of swaps, if odd multiply determinant by -1
    int cnt = 0;
    for (int i = 0; i < m*n; ++i) {
      if (*(devIpiv_h+i) == 0) break;
      if (*(devIpiv_h+i) != i+1) cnt++;
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

  /// Return the rank of a matrix
  /// Computation all done in GPU
  ///
  /// \param a
  /// An arbitrary matrix
  ///
  /// \return
  /// Rank of the input matrix
  static int Rank(const Matrix<T> &a) {
    // Obtain row echelon form through SVD
    GpuSvdSolver<T> svd;
    svd.Compute(a);

    // Obtain computed sigular vector
    Vector<T> sigular_vector = svd.SingularValues();

    // Count non zero elements of sigular vector
    int rank = 0;
    for (int i = 0; i < sigular_vector.rows(); i++) {
      if (sigular_vector[i] != 0)
        rank++;
    }

    return rank;
  }

  ///  static T FrobeniousNorm(const Matrix<T> &a)
  ///           calculates the norm of the entire matrix
  ///
  ///  \param a
  ///  const Matrix<T> &a : The m x n input matrix to have the
  ///                       frobenious norm found
  ///
  ///  \return
  ///  returns the resulting scalar produced by
  ///  c = (sum(i=0->i=m*n){a[i]^2})^(1/2) where the rhs
  ///  is the norm of the input matrix if treated as a single vector
  ///
  ///  \sa
  ///  \ref static T FrobeniousNorm(const Matrix<T> &a)
  static T FrobeniusNorm(const Matrix<T> &a) {
    int m = a.rows();
    int n = a.cols();
    int incx = 1;
    const T * h_a = &a(0);

    //Traceocate and transfer memories
    T * h_c = reinterpret_cast<T *>(malloc(sizeof(T)));
    T * d_a;  gpuErrchk(cudaMalloc(&d_a, m * n * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_a, h_a, m * n * sizeof(T), cudaMemcpyHostToDevice));

    // Setup and do Frobenious Norm
    cublasHandle_t  handle;
    cublasCreate(&handle);
    cublasStatus_t stat;
    stat = GpuFrobeniusNorm(handle, n * m, incx, d_a, h_c);

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

  /// Return the trace of a matrix
  /// Computation all done in GPU
  ///
  /// \param a
  /// An arbitrary matrix
  ///
  /// \return
  /// Trace of the input matrix
  static T Trace(const Matrix<T> &a) {
    // Get the diagonal vector
    Vector<T> diagonal_vector = a.diagonal();

    // Get the number of elements in diagonal vector
    int m = diagonal_vector.rows();

    // Create host memory
    const T *h_a = &diagonal_vector(0);
    T *h_multiplier = new T[m];
    for (int i = 0; i < m; i++)
      h_multiplier[i] = 1.0;
    T h_result;

    // Create device memory from host memory
    T *d_a;
    T *d_multiplier;
    T *d_result;
    gpuErrchk(cudaMalloc(&d_a, m * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_multiplier, m * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_result, sizeof(T)));

    // Copy host memory over to device
    gpuErrchk(cudaMemcpy(d_a, h_a, m * sizeof(T),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_multiplier, h_multiplier, m * sizeof(T),
                         cudaMemcpyHostToDevice));

    // Create parameters for cublas wraper function
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasOperation_t trans = CUBLAS_OP_T;
    int n = 1;
    T alpha = 1.0;
    T beta = 0.0;
    int lda = m;
    int incx = 1;
    int incy = 1;

    // Do vector summation to obtain trace
    stat = GpuMatrixVectorMul(handle, trans, m, n, &alpha,
                       d_a, lda, d_multiplier, incx, &beta, d_result, incy);

    // Error check
    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "GPU Trace Internal Failure" << std::endl;
      cudaFree(d_a);
      cudaFree(d_multiplier);
      cudaFree(d_result);
      cublasDestroy(handle);
      exit(1);
    }

    // Copy device result over to host
    gpuErrchk(cudaMemcpy(&h_result, d_result, sizeof(T),
                         cudaMemcpyDeviceToHost));

    // Synchonize and clean up
    cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_multiplier);
    cudaFree(d_result);
    delete []h_multiplier;

    // Destroy the handle
    cublasDestroy(handle);

    // Return the result
    return h_result;
  }

  ///  static T DotProduct(const Vector<T> &a, const Vector<T> &b)
  ///                      calculates the dot product of two vectors
  ///
  ///  \param a
  ///  const Vector<T> &a : The 1 x k input vector
  ///  \param b
  ///  const Vector<T> &b : The k , 1 input vector
  ///
  ///  \return
  ///  returns the resulting sumation of
  ///  c = sum(l=0->l=k){a[l] * b[l]}
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
    stat = GpuVectorVectorDot(handle, n, d_a, d_b, h_c);

    // Error Check
    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "GPU Vector Vector Dot Product Internal Failure"
                << std::endl;
      cudaFree(d_a); cudaFree(d_b);
      cublasDestroy(handle);
    }
    cudaDeviceSynchronize();

    // Free memories and return result
    cudaFree(d_a); cudaFree(d_b);
    cublasDestroy(handle);
    return *h_c;
  }

  ///  static Matrix<T> OuterProuct(const Vector<T> &a, const Vector<T> &b)
  ///                   calculates the matrix multiplication of two Vectors.
  ///
  ///  \param a
  ///  const Vector<T> &a : The p x 1 input vector
  ///  \param b
  ///  const Vector<T> &b : The 1 x p input vector
  ///
  ///  \return
  ///  returns the resulting matrix of
  ///  c[i,j] = sum(l=0->l=p){a[i,l] * b[l,j]}
  ///
  ///  \sa
  ///  \ref static Matrix<T> Multiply(const Matrix<T> &a, const Matrix<T> &b)
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
      stat = GpuMatrixMatrixMul(handle, m, n, k, d_a, d_b, d_c);

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

