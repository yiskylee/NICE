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

#ifndef CPP_INCLUDE_CPU_OPERATIONS_H_
#define CPP_INCLUDE_CPU_OPERATIONS_H_

#include <string>
#include <iostream>
#include <cmath>
#include "include/matrix.h"
#include "include/vector.h"
#include "include/kernel_types.h"
#include "Eigen/SVD"
#include "include/svd_solver.h"
#include "include/util.h"

namespace Nice {

// Abstract class of common matrix operation interface
template<typename T>
class CpuOperations {
 public:
  /// This is a function that calculates the transpose Matrix of the
  /// input Matrix
  ///
  /// \param a
  /// Input Matrix
  ///
  /// \return
  /// This function returns a Matrix of type T
  static Matrix<T> Transpose(const Matrix<T> &a) {
    return a.transpose();  // Return transpose
  }

  /// This is a function that calculates the transpose Vector of the
  /// input Vector
  ///
  /// \param a
  /// Input Vector
  ///
  /// \return
  /// This function returns a Vector of type T
  static Vector<T> Transpose(const Vector<T> &a) {
    return a.transpose();
  }

  /// This is a function that calculates the product Matrix of the input Matrix
  /// and a scalar
  ///
  /// \param a
  /// Input Matrix
  /// \param scalar
  /// Input scalar
  ///
  /// \return
  /// This function returns a Matrix of type T
  static Matrix<T> Multiply(const Matrix<T> &a, const T &scalar) {
    // Scalar-matrix multiplication
    return scalar * a;
  }

  /// This is a funtion that calculates the product Matrix of the two input
  /// Matrices
  ///
  /// \param a
  /// Input Matrix 1
  /// \param b
  /// Input Matrix 2
  ///
  /// \return
  /// This function returns a Matrix of type T
  static Matrix<T> Multiply(const Matrix<T> &a, const Matrix<T> &b) {
    // Matrix-matrix multiplication
    return a * b;
  }
  /// This is a function that adds each element in the matrix to a scalar and
  /// returns the resulting matrix.
  ///
  /// \param a
  /// Input Matrix
  /// \param scalar
  /// Input Scalar
  ///
  /// \return
  /// This function returns a matrix that is the resulatant of adding the
  /// input matrix and scalar.
  ///
  /// \sa
  /// \ref Add(const Matrix<T> &a, const Matrix<T> &b)
  static Matrix<T> Add(const Matrix<T> &a, const T &scalar) {
      // Does not work if matrix is empty.
      if (a.rows() == 0) {
        std::cerr << "MATRICIES ARE EMPTY";
        exit(1);

      // Otherwise, code will run fine.
    } else {
        return (a.array() + scalar);
    }
  }
  /// This is a function that adds two matricies and returns the resulting
  /// matrix.
  ///
  /// \param a
  /// Input Matrix 1
  /// \param b
  /// Input Matrix 2
  ///
  /// \return
  /// This function returns a matrix that is the resulatant of adding two
  /// input matricies.
  ///
  /// \sa
  /// \ref Add(const Matrix<T> &a, const T &scalar)
  static Matrix<T> Add(const Matrix<T> &a, const Matrix<T> &b) {
      // Does not work if matricies are not the same size.
      if ((a.rows() != b.rows()) || (a.cols() != b.cols())) {
        std::cerr << "MATRICIES ARE NOT THE SAME SIZE";
        exit(1);

      // Does not work if matricies are empty.
    } else if (a.rows() == 0) {
        std::cerr << "MATRICIES ARE EMPTY";
        exit(1);

      // Otherwise, code will run fine.
    } else {
        return a + b;
    }
  }
  static Matrix<T> Subtract(const Matrix<T> &a, const T &scalar) {
    // Matrix-scalar subtraction
    if (a.rows() == 0 || a.cols() == 0) {
      std::cerr << "EMPTY MATRIX AS ARGUEMENT!";
      exit(1);
    }
    return (a.array() - scalar);
  }
  static Matrix<T> Subtract(const Matrix<T> &a, const Matrix<T> &b) {
    // Matrix-matrix subtraction
    if ((a.rows() != b.rows()) || (a.cols() != b.cols())) {
      std::cerr << "MATRICES ARE NOT THE SAME SIZE!";
      exit(1);  // Exits the program
    } else if (b.rows() == 0 || b.cols() == 0 || a.rows() == 0
        || a.cols() == 0) {
      std::cerr << "EMPTY MATRIX AS ARGUMENT!";
      exit(1);  // Exits the program
    }
    return a - b;
  }

  /// This is a function that calculates the "logical or" of the two input
  /// Matrices
  ///
  /// \param a
  /// Input Matrix 1
  /// \param b
  /// Input Matrix 2
  ///
  /// \return
  /// This function returns a Matrix of type bool
  static Matrix<bool> LogicalOr(const Matrix<bool> &a, const Matrix<bool> &b) {
    // Returns the resulting matrix that is created by running a logical or
    // operation on the two input matrices
    if ((a.rows() != b.rows()) || (a.cols() != b.cols())) {
      std::cerr << "MATRICES ARE NOT THE SAME SIZE!";
      exit(1);  // Exits the program
    } else if (b.rows() == 0 || b.cols() == 0 || a.rows() == 0
        || a.cols() == 0) {
      std::cerr << "EMPTY MATRIX AS ARGUMENT!";
      exit(1);  // Exits the program
    }
    return (a.array() || b.array());
  }

  /// This is a funtion that returns the "logical not" of the input Matrix
  ///
  /// \param a
  /// Input Matrix 1
  ///
  /// \return
  /// This funtion returns a Matrix of type bool
  static Matrix<bool> LogicalNot(const Matrix<bool> &a) {
    Matrix<bool> b = a.replicate(1, 1);
    int r;
    // Iterate through the copied matrix
    for (r = 0; r < b.rows(); ++r) {
      for (int c = 0; c < b.cols(); ++c) {
        b(r, c) = !b(r, c);
      }
    }
    if (b.rows() == 0 || b.cols() == 0) {
      std::cerr << "EMPTY MATRIX AS ARGUMENT!";
      exit(1);  // Exits the program
    }
    return b;
  }

  /// This is a function that calculates the "logical and" of the two input
  /// Matrices
  ///
  /// \param a
  /// Input Matrix 1
  /// \param b
  /// Input Matrix 2
  ///
  /// \return
  /// This function returns a Matrix of type bool
  static Matrix<bool> LogicalAnd(const Matrix<bool> &a, const Matrix<bool> &b) {
    // Checks to see that the number of rows and columns are the same
    if ((a.rows() != b.rows()) || (a.cols() != b.cols())) {
      std::cerr << "MATRICES ARE NOT THE SAME SIZE!";
      exit(1);  // Exits the program
    }
    return (a.array() && b.array());
    // Will return a matrix due to implicit conversion
  }

  /// This is a function that returns the inverse of a matrix.
  ///
  /// \param a
  /// Input Matrix
  ///
  /// \return
  /// This function returns a matrix that is the inverse of the input matrix.
  static Matrix<T> Inverse(const Matrix<T> &a) {
      // If the matrix is empty, it should not check for inverse.
      if (a.cols() == 0) {
        std::cerr << "MATRIX IS EMPTY";
        exit(1);
      // If the matrix is not sqaure it will not produce an inverse.
    } else if (a.cols() != a.rows()) {
        std::cerr << "MATRIX IS NOT A SQUARE MATRIX!";
        exit(1);

      // If the determinant of a matrix is 0, it does not have an inverse.
    } else if (a.determinant() == 0) {
        std::cerr << "MATRIX DOES NOT HAVE AN INVERSE (DETERMINANT IS ZERO)!";
        exit(1);

      // If this point is reached then an inverse of the matrix exists.
    } else {
      return a.inverse();
    }
  }

  /// static Vector <T> Norm( const Matrix <T> &a,
  /// const int &p = 2, const int &axis = 0) calculates the norm of
  /// the values in an m x n dependent of the input p and axis.
  /// The norm is returned in the form of a vector. If the axis is 0,
  /// the norm will be calulated column wise and the size of the
  /// output vector will be dependent on n. If the axis is 1, the
  /// norm will be calculated row-wise and the size of the vector
  /// will be dependent on m.
  ///
  /// \param a
  /// const Matrix <T> &a
  /// \param b
  /// \const int &p
  /// \param c
  /// \const int &axis
  ///
  /// \return
  /// Vector <T>
  static Vector<T> Norm(const Matrix<T> &a,
                      const int &p = 2,
                      const int &axis = 0) {
    int num_rows = a.rows();
    int num_cols = a.cols();
    float norm_value = 0;
    if (axis == 0) {
    Vector<T> norm(num_cols);
     for (int j = 0; j < num_cols; j++) {
      for (int i = 0; i < num_rows; i++)
         norm_value += pow(a(i, j), p);
       norm(j) = pow(norm_value, (1.0/p));
       norm_value = 0;
     }
     return norm;
    } else if (axis == 1) {
     Vector<T> norm(num_rows);
     for (int i = 0; i < num_rows; i++) {
      for (int j = 0; j < num_cols; j++)
         norm_value += pow(a(i, j), p);
       norm(i) = pow(norm_value, (1.0/p));
       norm_value = 0;
     }
     return norm;
    } else {
      std::cerr << "Axis must be zero or one!";
      exit(1);
    }
}

  static T Determinant(const Matrix<T> &a);
  /// static int Rank(const Matrix <T> &a) is a function that returns
  ///                                      the rank of a m x n matrix
  /// \param a
  /// Matrix<T> &a
  ///
  /// \return
  /// This function returns an int value of the matrix's rank.

  static int Rank(const Matrix<T> &a) {
    // Rank of a matrix
    SvdSolver<T> svd;
    return svd.Rank(a);
  }

  /// This is a function that returns the frobenius norm of Matrix a
  ///
  /// \param a
  ///
  /// \return
  /// This function returns a value of type T
  static T FrobeniusNorm(const Matrix<T> &a) {
    if (a.rows() == 0 || a.cols() == 0) {
      std::cerr << "EMPTY MATRIX AS ARGUMENT!";
      exit(-1);  // Exits the program
    } else {
      return a.norm();
    }
  }

  /// This is a function that returns the sum of the diagonal coefficiants of a
  /// Matrix
  ///
  /// \param a
  /// Input Matrix
  ///
  /// \return
  /// This function returns a value of type T

  static T Trace(const Matrix<T> &a) {
    // Trace of a matrix
    return a.trace();
  }

  /// This is a function that calculates the dot product of two vectors.
  ///
  /// \param a
  /// Input Vector 1
  /// \param b
  /// Input Vector 2
  ///
  /// \return
  /// This function returns a value of type T which is the dot product of
  /// the two vectors.
  static T DotProduct(const Vector<T> &a, const Vector<T> &b) {
      // Checks to see if the size of the two vectors are not the same
      if (a.size() != b.size()) {
        std::cerr << "VECTORS ARE NOT THE SAME SIZE!";
        exit(1);

      // Checks to see if both vectors contain at least one element
      // Only one vector is checked because it is known that both
      // vectors are the same size
    } else if (a.size() == 0) {
      std::cerr << "VECTORS ARE EMPTY!";
      exit(1);

      // If this point is reached then calculating the dot product
      // of the two vectors is valid
    } else {
        return (a.dot(b));
    }
  }
  /// This is a function that calculates the "Outer Product of the input Vectors
  ///
  /// \param a
  /// Input Vector 1
  /// \param b
  /// Input Vector 2
  ///
  /// \return
  /// This function returns a Matrix of type T
  static Matrix<T> OuterProduct(const Vector<T> &a, const Vector<T> &b) {
    // This function returns the outer product of he two passed in vectors
    if (a.size() == 0 || b.size() == 0) {
      std::cerr << "EMPTY VECTOR AS ARGUMENT!";
      exit(1);
    }
    return a * b.transpose();
  }

  /// This is a function that calculates the "logical and" of the two input
  /// Vectors
  ///
  /// \param a
  /// Input Vector 1
  /// \param b
  /// Input Vector 2
  ///
  /// \return
  /// This function returns a Vector of type bool
  static Vector<bool> LogicalAnd(const Vector<T> &a, const Vector<T> &b) {
  if ((a.rows() != b.rows()) || (a.cols() != b.cols())) {
        std::cerr << "MATRICES ARE NOT THE SAME SIZE!";
        exit(1);  // Exits the program
      }
      return (a.array() && b.array());
  }
  /// This is a function that calculates the "logical or" of the two input
  /// Vectors
  ///
  /// \param a
  /// Input Vector 1
  /// \param b
  /// Input Vector 2
  ///
  /// \return
  /// This function returns a Vector of type bool
  static Vector<bool> LogicalOr(const Vector<bool> &a, const Vector<bool> &b) {
    // Returns the resulting vector that is created by running a logical or
    // operation on the two input vectors
    if (a.size() != b.size()) {
      std::cerr << "VECTORS ARE NOT THE SAME SIZE!";
      exit(1);  // Exits the program
    } else if (a.size() == 0 || b.size() == 0) {
      std::cerr << "EMPTY VECTOR AS ARGUMENT!";
      exit(1);  // Exits the program
    }
    return (a.array() || b.array());
  }

  /// This is a funtion that returns the "logical not" of the input Vector
  ///
  /// \param a
  /// Input Vector 1
  ///
  /// \return
  /// This funtion returns a Vector of type bool
  static Vector<bool> LogicalNot(const Vector<bool> &a) {
    Vector<bool> b = a.replicate(1, 1);
    int i;
    // Iterate through vector
    for (i = 0; i < b.size(); ++i) {
      b(i) = !b(i);
    }
    if (a.size() == 0) {
      std::cerr << "EMPTY VECTOR AS ARGUMENT!";
      exit(1);  // Exits the program
    }
    return b;
  }
  /// statix Matrix <T> Normalize(const Matrix <T> &a, const int &p
  /// =2, const int &axis = 0) normalizes a m x n matrix by element.
  ///
  /// \param a
  /// const Matrix<T> &a
  /// \param b
  /// const int &p = 2
  /// \param c
  /// const int &axis = 0
  ///
  /// \return
  /// Matrix <T>
  /// \sa
  /// \ref Norm
  static Matrix<T> Normalize(const Matrix<T> &a, const int &p = 2,
                                                  const int &axis = 0) {
    int num_rows = a.rows();
    int num_cols = a.cols();
    Matrix<T> b(num_rows, num_cols);
    if (axis == 0) {
     b = a.transpose().array().colwise() / Norm(a, p, axis).array();
     return b.transpose();
    } else if (axis == 1) {
     b = a.array().colwise() / Norm(a, p, axis).array();
     return b;
    } else {
     std::cerr << "Axis must be zero or one!";
     exit(1);
    }
  }
  /// Generates a kernel matrix from an input data_matrix
  /// \param data_matrix
  /// Input matrix whose rows represent samples and columns represent features
  /// \param kernel_type
  /// Kernel type could be chosen from Gaussian, Linear and Polynomial
  /// \param constant
  /// In Gaussian kernel, this is sigma;
  /// In Polynomial kernel, this is constant c
  /// In Linear kernel, this is c as well
  /// \return
  /// An nxn kernel matrix where n is the number of samples in data_matrix
  static Matrix<T> GenKernelMatrix(
      const Matrix<T> &data_matrix,
      const KernelType kernel_type = kGaussianKernel,
      const float constant = 1.0) {
    int num_samples = data_matrix.rows();
    // An n x n kernel matrix
    Matrix<T> kernel_matrix(num_samples, num_samples);
    if (kernel_type == kGaussianKernel) {
      float sigma = constant;
      // This for loop generates the kernel matrix using Gaussian kernel
      for (int i = 0; i < num_samples; i++)
        for (int j = 0; j < num_samples; j++) {
          // Calculate the the norm of (x_i - x_j) for all (i, j) pairs
          float i_j_dist = (data_matrix.row(i) - data_matrix.row(j)).norm();
          kernel_matrix(i, j) = exp(-i_j_dist / (2 * sigma * sigma));
        }
    }
    return kernel_matrix;
  }

  /// Generates a degree matrix D from an input kernel matrix
  /// It also generates D^(-1/2)
  /// \param kernel_matrix
  /// Input matrix: a squared kernel matrix
  /// \param degree_matrix
  /// Output degree matrix D
  /// \param degree_matrix_to_the_minus_half
  /// Output matrix D^(-1/2)
  static void GenDegreeMatrix(
      const Matrix<T> &kernel_matrix,
      Matrix<T> *degree_matrix,
      Matrix<T> *degree_matrix_to_the_minus_half) {
    // Generate the diagonal vector d_i and degree matrix D
    Vector<T> d_i = kernel_matrix.rowwise().sum();
    *degree_matrix = d_i.asDiagonal();
    // Generate matrix D^(-1/2)
    *degree_matrix_to_the_minus_half = d_i.array().sqrt().unaryExpr(
        std::ptr_fun(util::reciprocal<T>)).matrix().asDiagonal();
  }
};
}  // namespace Nice
#endif  // CPP_INCLUDE_CPU_OPERATIONS_H_
