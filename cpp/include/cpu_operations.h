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
#include "include/matrix.h"
#include "include/vector.h"

namespace Nice {

// Abstract class of common matrix operation interface
template<typename T>
class CpuOperations {
 public:
  static Matrix<T> Transpose(const Matrix<T> &a) {
    return a.transpose();  // Return transpose
  }
  static Vector<T> Transpose(const Vector<T> &a) {
    return a.transpose();
  }
  static Matrix<T> Multiply(const Matrix<T> &a, const T &scalar) {
    // Scalar-matrix multiplication
    return scalar * a;
  }
  static Matrix<T> Multiply(const Matrix<T> &a, const Matrix<T> &b) {
    // Matrix-matrix multiplication
    return a * b;
  }
  static Matrix<T> Add(const Matrix<T> &a, const T &scalar);
  static Matrix<T> Add(const Matrix<T> &a, const Matrix<T> &b);
  static Matrix<T> Subtract(const Matrix<T> &a, const T &scalar);
  static Matrix<T> Subtract(const Matrix<T> &a, const Matrix<T> &b);
  static Matrix<bool> LogicalOr(const Matrix<bool> &a, const Matrix<bool> &b) {
    // Returns the resulting matrix that is created by running a logical or
    // operation on the two input matrices
    if ((a.rows() != b.rows()) || (a.cols() != b.cols())) {
      std::cerr << std::endl << "ERROR: MATRICES ARE NOT THE SAME SIZE!"
                << std::endl << std::endl;
      exit(1);  // Exits the program
    } else if (b.rows() == 0 || b.cols() == 0 || a.rows() == 0
        || a.cols() == 0) {
      std::cerr << std::endl << "ERROR: EMPTY MATRIX AS ARGUMENT!" << std::endl
                << std::endl;
      exit(1);  // Exits the program
    }
    return (a.array() || b.array());
  }
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
      std::cerr << std::endl << "ERROR: EMPTY MATRIX AS ARGUMENT!" << std::endl
                << std::endl;
      exit(1);  // Exits the program
    }
    return b;
  }
  static Matrix<bool> LogicalAnd(const Matrix<bool> &a, const Matrix<bool> &b) {
    // This function returns the logical AND of two boolean matrices
    // Checks to see that the number of rows and columns are the same
    if ((a.rows() != b.rows()) || (a.cols() != b.cols())) {
      std::cerr << "/nERROR: MATRICES ARE NOT THE SAME SIZE!/n/n";
      exit(1);  // Exits the program
    }
    return (a.array() && b.array());
    // Will return a matrix due to implicit conversion
  }
  static Matrix<T> Inverse(const Matrix<T> &a);
  static Matrix<T> Norm(const int &p = 2, const int &axis = 0);
  static T Determinant(const Matrix<T> &a);
  static T Rank(const Matrix<T> &a);
  static T FrobeniusNorm(const Matrix<T> &a);
  static T Trace(const Matrix<T> &a) {
    // Trace of a matrix
    return a.trace();
  }
  static T DotProduct(const Vector<T> &a, const Vector<T> &b);

  static Matrix<T> OuterProduct(const Vector<T> &a, const Vector<T> &b) {
    // This function returns the outer product of he two passed in vectors
    if (a.size() == 0 || b.size() == 0) {
      std::cerr << std::endl << "ERROR: EMPTY VECTOR AS ARGUMENT!" << std::endl
                << std::endl;
      exit(1);
    }
    return a * b.transpose();
  }
  static Vector<T> LogicalAnd(const Vector<T> &a, const Vector<T> &b);
  static Vector<bool> LogicalOr(const Vector<bool> &a, const Vector<bool> &b) {
    // Returns the resulting vector that is created by running a logical or
    // operation on the two input vectors
    if (a.size() != b.size()) {
      std::cerr << std::endl << "ERROR: VECTORS ARE NOT THE SAME SIZE!"
                << std::endl << std::endl;
      exit(1);  // Exits the program
    } else if (a.size() == 0 || b.size() == 0) {
      std::cerr << std::endl << "ERROR: EMPTY VECTOR AS ARGUMENT!" << std::endl
                << std::endl;
      exit(1);  // Exits the program
    }
    return (a.array() || b.array());
  }
  static Vector<bool> LogicalNot(const Vector<bool> &a) {
    Vector<bool> b = a.replicate(1, 1);
    int i;
    // Iterate through vector
    for (i = 0; i < b.size(); ++i) {
      b(i) = !b(i);
    }
    if (a.size() == 0) {
      std::cerr << std::endl << "ERROR: EMPTY VECTOR AS ARGUMENT!" << std::endl
                << std::endl;
      exit(1);  // Exits the program
    }
    return b;
  }
};
}  // namespace Nice
#endif  // CPP_INCLUDE_CPU_OPERATIONS_H_
