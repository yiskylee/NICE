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

#ifndef CPP_INCLUDE_SVD_SOLVER_H_
#define CPP_INCLUDE_SVD_SOLVER_H_

#include "include/matrix.h"
#include "include/vector.h"

#include "Eigen/SVD"


namespace Nice {

// Abstract class of svd solver
template<typename T>
class SvdSolver {
 private:
  Eigen::JacobiSVD<Matrix<T>> svd_;

 public:
  SvdSolver()
  :
  svd_() {}

  /// Computation function that perform the actual SVD decomposition
  ///
  /// \param a
  /// An arbitrary matrix
  ///
  /// \return
  /// Void
  void Compute(const Matrix<T> &a) {
    svd_.compute(a, Eigen::ComputeFullU|Eigen::ComputeFullV);
  }

  /// Return the matrix U after SVD decomposition
  ///
  /// \param
  /// Void
  ///
  /// \return
  /// Matrix U
  Matrix<T> MatrixU() const {
    return svd_.matrixU();
  }

  /// Return the matrix V after SVD decomposition
  ///
  /// \param
  /// Void
  ///
  /// \return
  /// Matrix V
  Matrix<T> MatrixV() const {
    return svd_.matrixV();
  }

  /// Return the singular values  after SVD decomposition
  ///
  /// \param
  /// Void
  ///
  /// \return
  /// Vector S
  Vector<T> SingularValues() const {
    return svd_.singularValues();
  }

  /// Return the rank of a matrix through SVD decomposition
  ///
  /// \param a
  /// An arbitrary matrix
  ///
  /// \return
  /// Matrix rank
  int Rank(const Matrix<T> &a) {
    Compute(a);
    return svd_.rank();
  }
};

}  // namespace Nice

#endif  // CPP_INCLUDE_SVD_SOLVER_H_

