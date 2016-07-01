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

#include "include/svd_solver.h"
#include <iostream>
#include "Eigen/Dense"
#include "Eigen/SVD"
#include "include/matrix.h"
#include "include/vector.h"

namespace Nice {

template<typename T>
SvdSolver<T>::SvdSolver()
:
svd_() {}

template<typename T>
void SvdSolver<T>::Compute(const Matrix<T> &a) {
  svd_.compute(a, Eigen::ComputeFullU|Eigen::ComputeFullV);
}

template<typename T>
Matrix<T> SvdSolver<T>::MatrixU() const {
  return svd_.matrixU();
}

template<typename T>
Matrix<T> SvdSolver<T>::MatrixV() const {
  return svd_.matrixV();
}

template<typename T>
Vector<T> SvdSolver<T>::SingularValues() const {
  return svd_.singularValues();
}

template<typename T>
int SvdSolver<T>::Rank(const Matrix<T> &a) {
  Compute(a);
  return svd_.rank();
}


template class SvdSolver<float>;
template class SvdSolver<double>;

}  //  namespace Nice
