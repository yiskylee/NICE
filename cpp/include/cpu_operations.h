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
#include "include/matrix.h"
#include "include/vector.h"


namespace Nice {

// Abstract class of common matrix operation interface
template<typename T>
class CpuOperations {
 public:
  static Matrix<T> Transpose(const Matrix<T> &a);
  static Vector<T> Transpose(const Vector<T> &a);
  static Matrix<T> Multiply(const Matrix<T> &a, const T &scalar);
  static Matrix<T> Multiply(const Matrix<T> &a, const Matrix<T> &b);
  static Matrix<T> Add(const Matrix<T> &a, const T &scalar);
  static Matrix<T> Add(const Matrix<T> &a, const Matrix<T> &b);
  static Matrix<T> Subtract(const Matrix<T> &a, const T &scalar);
  static Matrix<T> Subtract(const Matrix<T> &a, const Matrix<T> &b);
  static Matrix<T> LogicalAnd(const Matrix<T> &a, const Matrix<T> &b);
  static Matrix<T> LogicalNot(const Matrix<T> &a, const Matrix<T> &b);
  static Matrix<T> Inverse(const Matrix<T> &a);
  static Matrix<T> Norm(const int &p = 2, const int &axis = 0);
  static T Determinant(const Matrix<T> &a);
  static T Rank(const Matrix<T> &a);
  static T FrobeniusNorm(const Matrix<T> &a);
  static T Trace(const Matrix<T> &a);
  static T DotProduct(const Vector<T> &a, const Vector<T> &b);
  static Matrix<T> OuterProduct(const Vector<T> &a, const Vector<T> &b);
  static Vector<T> LogicalAnd(const Vector<T> &a, const Vector<T> &b);
  static Vector<T> LogicalOr(const Vector<T> &a, const Vector<T> &b);
  static Vector<T> LogicalNot(const Vector<T> &a, const Vector<T> &b);
};
}  // namespace Nice

#endif  // CPP_INCLUDE_CPU_OPERATIONS_H_
