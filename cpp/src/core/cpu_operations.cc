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

#include "include/cpu_operations.h"
#include <unistd.h>
#include <iostream>
#include "Eigen/Dense"
#include "include/matrix.h"
#include "include/vector.h"

namespace Nice {

// This function returns the transpose of a matrix
template<typename T>
Matrix<T> CpuOperations<T>::Transpose(const Matrix<T> &a) {
  return a.transpose();  // Return transpose
}

template<typename T>
Vector<T> CpuOperations<T>::Transpose(const Vector<T> &a) {
  return a.transpose();
}

// Scalar-matrix multiplication
template<typename T>
Matrix<T> CpuOperations<T>::Multiply(const Matrix<T> &a, const T &scalar) {
  return scalar * a;
}

// Matrix-matrix multiplication
template<typename T>
Matrix<T> CpuOperations<T>::Multiply(const Matrix<T> &a, const Matrix<T> &b) {
  return a * b;
}

// Trace of a matrix
template<typename T>
T CpuOperations<T>::Trace(const Matrix<T> &a) {
  return a.trace();
}

// This function returns the logical AND of two boolean matrices
template<typename T>
Matrix<bool> CpuOperations<T>::LogicalAnd(const Matrix<bool> &a,
                                          const Matrix<bool> &b) {
  // Checks to see that the number of rows and columns are the same
  if ((a.rows() != b.rows()) || (a.cols() != b.cols())) {
    std::cout << std::endl << "ERROR: MARTRICES ARE NOT THE SAME SIZE!"
    << std::endl << std::endl;
    exit(-1);  // Exits the program
  }
  return (a.array() && b.array());
  // Will return a matrix due to implicit conversion
}

template class CpuOperations<int>;
template class CpuOperations<float>;
template class CpuOperations<double>;
template class CpuOperations<bool>;
}  //  namespace Nice

