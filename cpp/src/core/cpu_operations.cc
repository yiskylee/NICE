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
Matrix<bool>CpuOperations<T>::LogicalOr(const Matrix<bool> &a,
                                        const Matrix<bool> &b) {
  if(( a.rows() != b.rows() ) || ( a.cols() != b.cols() )) {
    throw std::invalid_argument("ERROR: MATRICES ARE NOT THE SAME SIZE!");
  }
  return (a.array() || b.array());
}

// Returns the resulting vector that is created by running a logical or
// operation on the two input vectors
template<typename T>
Vector<bool>CpuOperations<T>::LogicalOr(const Vector<bool> &a,
                                        const Vector<bool> &b) {
  if(( a.rows() != b.rows() ) || ( a.cols() != b.cols() )) {
    throw std::invalid_argument("ERROR: VECTORS ARE NOT THE SAME SIZE!");
  }
  return (a.array() || b.array());
}

template class CpuOperations<int>;
template class CpuOperations<float>;
template class CpuOperations<double>;

}  //  namespace Nice
