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

template<typename T>
Matrix<bool> CpuOperations<T>::LogicalNot(const Matrix<bool> &a) {
  Matrix<bool> b = a.replicate(1,1);
  //Iterate through the copied matrix
  for(int r = 0; r < b.rows(); ++r) {
    for(int c = 0; c < b.cols(); ++c) {
      if(b(r,c) != 0 && b(r,c) != 1) {
    	  throw std::invalid_argument("Empty Matrix as Argument!");
      }
      b(r,c) = !b(r,c);
    }
  }
  return b;
}

template<typename T>
Vector<bool> CpuOperations<T>::LogicalNot(const Vector<bool> &a) {
  Vector<bool> b = a.replicate(1,1);
  //Iterate through vector
  for(int i = 0; i < b.size(); ++i) {
    b(i) = !b(i);
  }
  return b;
}

template class CpuOperations<int>;
template class CpuOperations<float>;
template class CpuOperations<double>;
template class CpuOperations<bool>;

}  //  namespace Nice
