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
#include <stdexcept>
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


// Returns the resulting matrix that is created by running a logical or
// operation on the two input matrices
template<typename T>
Matrix<bool> CpuOperations<T>::LogicalOr(const Matrix<bool> &a,
                                        const Matrix<bool> &b) {
  if ((a.rows() != b.rows()) || (a.cols() != b.cols())) {
    throw std::invalid_argument("ERROR: MATRICES ARE NOT THE SAME SIZE!");
  }
  return (a.array() || b.array());
}

// Returns the resulting vector that is created by running a logical or
// operation on the two input vectors
template<typename T>
Vector<bool> CpuOperations<T>::LogicalOr(const Vector<bool> &a,
                                        const Vector<bool> &b) {
  if ( a.size() != b.size() ) {
    throw std::invalid_argument("ERROR: VECTORS ARE NOT THE SAME SIZE!");
  }
  return (a.array() || b.array());
}

template<typename T>
Matrix<bool> CpuOperations<T>::LogicalNot(const Matrix<bool> &a) {
  Matrix<bool> b = a.replicate(1, 1);
  int r;
  // Iterate through the copied matrix
  for (r = 0; r < b.rows(); ++r) {
    for (int c = 0; c < b.cols(); ++c) {
      b(r, c) = !b(r, c);
    }
  }
  if (b.rows() == 0 || b.cols() == 0) {
    throw std::invalid_argument("ERROR: EMPTY MATRIX AS ARGUMENT!");
  }
  return b;
}

template<typename T>
Vector<bool> CpuOperations<T>::LogicalNot(const Vector<bool> &a) {
  Vector<bool> b = a.replicate(1, 1);
  int i;
  // Iterate through vector
  for (i = 0; i < b.size(); ++i) {
    b(i) = !b(i);
  }
  if (a.size() == 0) {
    throw std::invalid_argument("ERROR: EMPTY VECTOR AS ARGUMENT!");
    }
  return b;
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

/*
// Rank of a matrix
template<typename T>
T CpuOperations<T>::Rank(const Matrix<T> &a) {
    return a.rank();
}
*/

// This function returns the dot product of two vectors
template<typename T>
T CpuOperations<T>::DotProduct(const Vector<T> &a,
                               const Vector<T> &b) {
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

// This function returns the logical AND of two boolean matrices
template<typename T>
Matrix<bool> CpuOperations<T>::LogicalAnd(const Matrix<bool> &a,
                                          const Matrix<bool> &b) {
  // Checks to see that the number of rows and columns are the same
  if ((a.rows() != b.rows()) || (a.cols() != b.cols())) {
    std::cout << std::endl << "ERROR: MATRICES ARE NOT THE SAME SIZE!"
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

