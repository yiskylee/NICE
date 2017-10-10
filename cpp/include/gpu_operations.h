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

#ifdef CUDA_AND_GPU

#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <unistd.h>
#include <stdexcept>

#include <iostream>

#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"
#include "include/gpu_svd_solver.h"

namespace Nice {

// Abstract class of common matrix operation interface
template <typename T>
class GpuOperations {
 private:
  static GpuUtil<T> *util_;

 public:
  /// This function multiplies an input Matrix and a scalar
  ///
  /// \param a
  /// Input Matrix
  /// \param scalar
  /// Input Scalar of type T
  ///
  /// \return
  /// This function returns a Matrix of type T
  static Matrix<T> Multiply(const Matrix<T> &a, const T &scalar) {
    // Allocate and transfer memory
    int n = a.cols() * a.rows();
    const T * h_a = &a(0);
    Matrix<T> h_c(a.rows(), a.cols());
    T * d_a;

    // Setup GPU memory
    util_->SetupMem(&d_a, h_a, n);

    // Set up and do cublas matrix scalar multiply
    GpuMatrixScalarMul(util_->GetBlasHandle(), n, scalar, d_a);

    // Device sync
    util_->SyncDev();

    // Transfer memories back, clear memories, and return result
    util_->SyncMem(d_a, &h_c(0, 0), n);

    return h_c;
  }

  /// This function multiplies an input Vector and a scalar
  ///
  /// \param a
  /// Input Vector
  /// \param scalar
  /// Input Scalar of type T
  ///
  /// \return
  /// This function returns a Vector of type T
  static Vector<T> Multiply(const Vector<T> &a, const T &scalar) {
      // Allocate and transfer memory
    int n = a.rows();
    const T * h_a = &a(0);
    Vector<T> h_c(a.rows());
    T * d_a;

    // Setup GPU memory
    util_->SetupMem(&d_a, h_a, n);

    GpuMatrixScalarMul(util_->GetBlasHandle(), n, scalar, d_a);

    // Device sync
    util_->SyncDev();

    // Transfer memories back, clear memories, and return result
    util_->SyncMem(d_a, &h_c(0, 0), n);

    return h_c;
  }

  /// This function multiplies an input Matrix and a Matrix
  ///
  /// \param a
  /// Input Matrix of type T
  /// \param b
  /// Input Matrix of type T
  ///
  /// \return
  /// This function returns a Matrix of type T
  /**static Matrix<T> Multiply(const Matrix<T> &a, const Matrix<T> &b) {
    if (a.cols() == b.rows()) {  // Check if matricies k vals are equal
      // Allocate and transfer memories
      int m = a.rows();
      int n = b.cols();
      int k = a.cols();

      const T * h_a = &a(0);
      const T * h_b = &b(0);
      Matrix<T> h_c(m, n);

      T * d_a;
      T * d_b;
      T * d_c;

      // Setup GPU memory
      util_->SetupMem(&d_a, h_a, m * k);
      util_->SetupMem(&d_b, h_b, k * n);
      util_->SetupMem(&d_c, nullptr, m * n, false);

      // Set up and do cublas matrix multiply
      GpuMatrixMatrixMul(util_->GetBlasHandle(), m, n, k, d_a, d_b, d_c);



      // Device sync
      util_->SyncDev();

      // Transfer memories back, clear memrory, and return result
      util_->SyncMem(d_a, nullptr, 0, false);
      util_->SyncMem(d_b, nullptr, 0, false);
      util_->SyncMem(d_c, &h_c(0, 0), m * n);

      return h_c;
    } else {
      std::cerr << "Matricies in gpu matrix multiply's sizes aren't compatible"
                << std::endl;
      exit(1);
    }
  }**/

  /// This function multiplies an input Matrix and a Vector
  ///
  /// \param a
  /// Input Matrix
  /// \param b
  /// Input Vector of type T
  ///
  /// \return
  /// This function returns a Vector of type T
  static Matrix<T> Multiply(const Matrix<T> &a, const Vector<T> &b) {
    if (a.cols() == b.rows() && !a.isZero()) {
      // Allocate and transfer memories
      int m = a.rows();
      int n = b.cols();
      int k = a.cols();

      const T * h_a = &a(0);
      const T * h_x = &b(0);
      Vector<T> h_y(m);

      T * d_a;
      T * d_x;
      T * d_y;

      // Setup GPU memory
      util_->SetupMem(&d_a, h_a, m * k);
      util_->SetupMem(&d_x, h_x, k * n);
      util_->SetupMem(&d_y, nullptr, m, false);

      cublasOperation_t norm = CUBLAS_OP_N;

      T alpha = 1.0;
      T beta = 0.0;
      int lda = m;
      int incx = 1;
      int incy = 1;

      // Set up and do cublas matrix multiply
      GpuMatrixVectorMul(util_->GetBlasHandle(), norm, m, k, &alpha,
                        d_a, lda, d_x, incx, &beta, d_y, incy);

      // Device sync
      util_->SyncDev();
      // Transfer memories back, clear memrory, and return result
      util_->SyncMem(d_a, nullptr, 0, false);
      util_->SyncMem(d_x, nullptr, 0, false);
      util_->SyncMem(d_y, &h_y(0), m);

      return h_y;
    } else if (a.cols() != b.rows()) {
      std::cerr << "Matricies in gpu matrix multiply's sizes aren't compatible"
                << std::endl;
      exit(1);
    } else if (a.isZero() && b.isZero()) {
      std::cerr << "The maxtrix and the vector are empty"
                << std::endl;
      exit(1);
    } else if (a.isZero()) {
      std::cerr << "The maxtrix is empty"
                << std::endl;
      exit(1);
    } else if (b.isZero()) {
      std::cerr << "The vector is empty"
                << std::endl;
      exit(1);
    } else {
      std::cerr << "Unknown error"
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

    T * d_a;
    T * d_b;
    T * d_c;

    // Setup GPU memory
    util_->SetupMem(&d_a, h_a, m * n);
    util_->SetupMem(&d_b, h_b, m * n);
    util_->SetupMem(&d_c, nullptr, m * n, false);

    GpuMatrixAdd(util_->GetBlasHandle(),
                 m, n,
                 &alpha,
                 d_a, lda,
                 &beta,
                 d_b, ldb,
                 d_c, ldc);

    // Device sync
    util_->SyncDev();

    // Transfer memories back, clear memrory, and return result
    util_->SyncMem(d_a, nullptr, 0, false);
    util_->SyncMem(d_b, nullptr, 0, false);
    util_->SyncMem(d_c, &h_c(0, 0), m * n);

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

      T * d_a;
      T * d_b;
      T * d_c;

      // Setup GPU memory
      util_->SetupMem(&d_a, h_a, m * n);
      util_->SetupMem(&d_b, h_b, m * n);
      util_->SetupMem(&d_c, nullptr, m * n, false);

      GpuMatrixAdd(util_->GetBlasHandle(),
                   m, n,
                   &alpha,
                   d_a, lda,
                   &beta,
                   d_b, ldb,
                   d_c, ldc);

      // Device sync
      util_->SyncDev();

      // Transfer memories back, clear memrory, and return result
      util_->SyncMem(d_a, nullptr, 0, false);
      util_->SyncMem(d_b, nullptr, 0, false);
      util_->SyncMem(d_c, &h_c(0, 0), m * n);

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

      T * d_a;
      T * d_b;
      T * d_c;

      // Setup GPU memory
      util_->SetupMem(&d_a, h_a, m * n);
      util_->SetupMem(&d_b, h_b, m * n);
      util_->SetupMem(&d_c, nullptr, m * n, false);

      // Set up and do cublas matrix subtract
      GpuMatrixMatrixSub(util_->GetBlasHandle(), m, n, &alpha, d_a, lda,
                         &beta, d_b, ldb, d_c, ldc);

      // Device sync
      util_->SyncDev();

      // Transfer memories back, clear memrory, and return result
      util_->SyncMem(d_a, nullptr, 0, false);
      util_->SyncMem(d_b, nullptr, 0, false);
      util_->SyncMem(d_c, &h_c(0, 0), m * n);

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

    // Setup GPU memory
    util_->SetupMem(&d_a, h_a, n * n);
    util_->SetupIntMem(&d_ipiv, nullptr, n, false);
    util_->SetupIntMem(&d_info, nullptr, 1, false);

    // Setup cusolver parameters
    int lda = n;
    int nrhs = n;
    int ldb = lda;

    // Setup workspace for LU decomposition
    int workspace_size;
    GpuGetLUDecompWorkspace(util_->GetSolverHandle(),
      n, n, d_a, lda, &workspace_size);

    T *workspace;
    util_->SetupMem(&workspace, nullptr, workspace_size, false);

    // Do LU docomposition
    GpuLUDecomposition(util_->GetSolverHandle(), n, n, d_a, lda,
                       workspace, d_ipiv, d_info);

    util_->SyncMem(workspace, nullptr, 0, false);

    // Create an identity matrix
    Matrix<T> b = Matrix<T>::Identity(n, n);

    // Create host memory
    T *h_b = &b(0);

    // Create device memory needed
    T *d_b;
    util_->SetupMem(&d_b, h_b, n * n);

    // Do lineaer solver
    GpuLinearSolver(util_->GetSolverHandle(),
      CUBLAS_OP_N, n, nrhs, d_a, lda, d_ipiv, d_b,
      ldb, d_info);

    // Device sync
    util_->SyncDev();

    // Transfer memories back, clear memrory, and return result
    util_->SyncMem(d_a, nullptr, 0, false);
    util_->SyncIntMem(d_ipiv, nullptr, 0, false);
    util_->SyncIntMem(d_info, nullptr, 0, false);
    util_->SyncMem(d_b, h_b, n * n);

    // Return the result
    return b;
  }

  static Vector<T> Norm(const Matrix<T> a, const int &p = 2,
                        const int &axis = 0) {
    int m = a.rows();
    int n = a.cols();
    int incx = 1;
    Vector<T> c(n);
    const T * h_a = &a(0);

    // Allocate and transfer memories
    T h_c;
    T * d_a;
    util_->SetupMem(&d_a, h_a, m * n);

    // Setup and do Frobenious Norm
    int iter = 0;
    for (int i = 0; i < n; ++i) {
      GpuFrobeniusNorm(util_->GetBlasHandle(), m, incx, d_a + i * m, &h_c);
      c(iter) = h_c;
      iter++;
    }

    // Free memories and return answer
    util_->SyncMem(d_a, nullptr, 0, false);

    return c;
  }
  static T Determinant(const Matrix<T> &a) {
    int m = a.rows();
    int n = a.cols();
    const T *h_a = &a(0);
    T det;

    // Allocating and transfering memories
    T *d_a;
    util_->SetupMem(&d_a, h_a, m * n);
    int *devIpiv_h = new int[m * n];
    int *devIpiv_d;
    util_->SetupIntMem(&devIpiv_d, nullptr, m * n, false);
    int devInfo_h = 0;
    int *devInfo_d;
    util_->SetupIntMem(&devInfo_d, nullptr, 1, false);
    T *h_c = new T[m * n];

    // Setup and do get LU decomposition buffer
    int work_size = 0;
    GpuLuWorkspace(util_->GetSolverHandle(), m, n, d_a, &work_size);

    // Allocate LU decomposition workspace memory and do LU decomposistion
    T *workspace;
    util_->SetupMem(&workspace, nullptr, work_size, false);
    GpuDeterminant(util_->GetSolverHandle(), m, n, d_a,
      workspace, devIpiv_d, devInfo_d);
    util_->SyncMem(workspace, nullptr, 0, false);

    // Device sync
    util_->SyncDev();

    // Transfer memories back, clear memrory, and return result
    util_->SyncMem(d_a, h_c, m * n);
    util_->SyncIntMem(devIpiv_d, devIpiv_h, m * n);
    util_->SyncIntMem(devInfo_d, &devInfo_h, 1);

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
    delete []devIpiv_h;
    delete []h_c;
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

  static T Norm(const Vector<T> &a) {
    int num_elem = a.rows();
    int incx = 1;
    const T * h_a = &a(0);

    T h_c;
    T * d_a;
    util_->SetupMem(&d_a, h_a, num_elem);

    GpuFrobeniusNorm(util_->GetBlasHandle(), num_elem, incx, d_a, &h_c);

    // Device sync
    util_->SyncDev();

    // Transfer memories back, clear memrory, and return result
    util_->SyncMem(d_a, nullptr, 0, false);
    return h_c;
  }

  static T SquaredNorm(const Vector<T> &a) {
    T norm = Norm(a);
    return norm * norm;
  }

  static T FrobeniusNorm(const Matrix<T> &a) {
    int m = a.rows();
    int n = a.cols();
    int incx = 1;
    const T * h_a = &a(0);

    // Traceocate and transfer memories
    T h_c;
    T * d_a;
    util_->SetupMem(&d_a, h_a, m * n);

    // Setup and do Frobenious Norm
    GpuFrobeniusNorm(util_->GetBlasHandle(), n * m, incx, d_a, &h_c);

    // Device sync
    util_->SyncDev();

    // Transfer memories back, clear memrory, and return result
    util_->SyncMem(d_a, nullptr, 0, false);

    return h_c;
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
    util_->SetupMem(&d_a, h_a, m);
    util_->SetupMem(&d_multiplier, h_multiplier, m);
    util_->SetupMem(&d_result, nullptr, 1, false);

    // Create parameters for cublas wraper function
    cublasOperation_t trans = CUBLAS_OP_T;
    int n = 1;
    T alpha = 1.0;
    T beta = 0.0;
    int lda = m;
    int incx = 1;
    int incy = 1;

    // Do vector summation to obtain trace
    GpuMatrixVectorMul(util_->GetBlasHandle(), trans, m, n, &alpha,
                       d_a, lda, d_multiplier, incx, &beta, d_result, incy);

    // Device sync
    util_->SyncDev();

    // Transfer memories back, clear memrory, and return result
    util_->SyncMem(d_a, nullptr, 0, false);
    util_->SyncMem(d_multiplier, nullptr, 0, false);
    util_->SyncMem(d_result, &h_result, 1);

    delete []h_multiplier;

    // Return the result
    return h_result;
  }

  static T DotProduct(const Vector<T> &a, const Vector<T> &b) {
    int n = a.rows();

    // Allocate and transfer memories
    const T * h_a = &a(0);
    const T * h_b = &b(0);
    T h_c;

    T * d_a;
    T * d_b;
    util_->SetupMem(&d_a, h_a, n);
    util_->SetupMem(&d_b, h_b, n);

    // Setup and do dot product
    GpuVectorVectorDot(util_->GetBlasHandle(), n, d_a, d_b, &h_c);

    // Device sync
    util_->SyncDev();

    // Transfer memories back, clear memrory, and return result
    util_->SyncMem(d_a, nullptr, 0, false);
    util_->SyncMem(d_b, nullptr, 0, false);

    return h_c;
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

      T * d_a;
      T * d_b;
      T * d_c;
      util_->SetupMem(&d_a, h_a, m * k);
      util_->SetupMem(&d_b, h_b, k * n);
      util_->SetupMem(&d_c, nullptr, m * n, false);

      // Setup and do outer product multiply
      GpuMatrixMatrixMul(util_->GetBlasHandle(), m, n, k, d_a, d_b, d_c);

      // Device sync
      util_->SyncDev();

      // Transfer memories back, clear memrory, and return result
      util_->SyncMem(d_a, nullptr, 0, false);
      util_->SyncMem(d_b, nullptr, 0, false);
      util_->SyncMem(d_c, &h_c(0, 0), m * n);

      return h_c;
    } else {
      std::cerr << "Vectors in gpu outer product's sizes aren't compatible"
                << std::endl;
      exit(1);
    }
  }
};

template <typename T>
GpuUtil<T> *GpuOperations<T>::util_ = GpuUtil<T>::GetInstance();

}  // namespace Nice
#endif  // CUDA_AND_GPU
#endif  // CPP_INCLUDE_GPU_OPERATIONS_H_
