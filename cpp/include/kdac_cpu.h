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

// Kernel Dimension Alternative Clustering (KDACCPU)
// Please refer to the paper published in PAMI by Liu, Dy and Jordan at:
// http://people.eecs.berkeley.edu/~jordan/papers/niu-dy-jordan-pami.pdf
// We try to follow naming conventions in the paper as much as possible.
// The lower cased variable names is the same as in the paper, and the
// upper cased matrix variable names in the paper are converted to lower
// case suffixed with "_matrix". For example:
// matrix U in the paper is named u_matrix in this implementation.

#ifndef CPP_INCLUDE_KDAC_CPU_H
#define CPP_INCLUDE_KDAC_CPU_H

#include "include/kdac.h"


namespace Nice {
template<class T>
class KDACCPU: public KDAC<T> {
 public:
  using KDAC<T>::kernel_type_;
  using KDAC<T>::x_matrix_;
  using KDAC<T>::constant_;
  using KDAC<T>::n_;
  using KDAC<T>::wl_deltaxij_proj_matrix_;
  using KDAC<T>::kij_matrix_;

  /// This is the default constructor for KDAC
  KDACCPU() = default;

  ~KDACCPU() = default;

  KDACCPU(const KDACCPU &rhs) {}
//  KDACCPU &operator=(const KDACCPU &rhs) {}

  void OptimizeW() {
    KDAC<T>::GenGammaMatrix();
    KDAC<T>::OptimizeW();
  }

  // Generate the term exp(-wTA_ijw) for different w, and put every kij into
  // a nxn matrix kij_matrix
  // Find out more at https://github.com/yiskylee/NICE/wiki
  void GenKij(const Vector<T> &w_l) {
    if (kernel_type_ == kGaussianKernel) {
      // -1 / 2 * sigma ^2
      T denom = -1.f / (2 * constant_ * constant_);
      for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
          Vector<T> delta_x_ij = x_matrix_.row(i) - x_matrix_.row(j);
          T projection = w_l.dot(delta_x_ij);
          wl_deltaxij_proj_matrix_(i, j) = projection;
          kij_matrix_(i, j) = std::exp(denom * projection * projection);
        }
      }
    }
  }


};
}  // namespace NICE

#endif  // CPP_INCLUDE_KDAC_CPU_H