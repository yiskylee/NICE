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

#ifndef CPP_SRC_CORE_COMMONMATRIXOPERATIONSINTERFACE_H_
#define CPP_SRC_CORE_COMMONMATRIXOPERATIONSINTERFACE_H_

#include "Matrix.h"

namespace nice {

// Forward declaration
template <typename T>
class Matrix;

// Abstract class of common matrix operation interface
template <typename T>
class CommonMatrixOperationInterface {
 public:
  virtual void Svd() = 0;
  virtual Matrix<T> Multiply(const Matrix<T> &m) = 0;
  virtual Matrix<T> Add(const Matrix<T> &m) = 0;
  virtual Matrix<T> Substract(const Matrix<T> &m) = 0;
  virtual void Transpose() = 0;
  virtual Matrix<T> Inverse() = 0;
  virtual Matrix<T> Norm(const int &p, const int &axis) = 0;
  virtual Matrix<T> Multiply(const T &v) = 0;
  virtual Matrix<T> Add(const T &v) = 0;
  virtual Matrix<T> Substract(const T &v) = 0;
};

}  // namespace nice

#endif  // CPP_SRC_CORE_COMMONMATRIXOPERATIONSINTERFACE_H_

