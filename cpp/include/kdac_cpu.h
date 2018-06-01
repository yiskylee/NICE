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
  using KDAC<T>::phi_of_alpha_;
  using KDAC<T>::phi_of_zero_;
  using KDAC<T>::phi_of_zero_prime_;
  using KDAC<T>::gamma_matrix_;
  using KDAC<T>::x_matrix_;
  using KDAC<T>::kernel_type_;
  using KDAC<T>::g_of_w_;
  using KDAC<T>::new_g_of_w_;
  using KDAC<T>::kij_matrix_;
  using KDAC<T>::profiler_;
  using KDAC<T>::n_;
  using KDAC<T>::d_;
  using KDAC<T>::alpha_;
  using KDAC<T>::constant_;

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
          kij_matrix_(i, j) = std::exp(denom * projection * projection);
        }
      }
    }
  }

  Vector<T> GenWGradient(const Vector<T> &w_l) {
    bool output = false;
    Vector<T> w_gradient = Vector<T>::Zero(d_);
//    Matrix<T> kij_matrix = GenKij(w_l);
    if (kernel_type_ == kGaussianKernel) {
      T denom = -1.f / (2 * constant_ * constant_);
      for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
          // scalar_term is -gamma_ij * 1/sigma^2 * g(w) * exp(-waw/2sigma^2)
          Vector<T> delta_x_ij = x_matrix_.row(i) - x_matrix_.row(j);
          T projection = w_l.dot(delta_x_ij);
          T kij = std::exp(denom * projection * projection);
          T scalar_term = -gamma_matrix_(i, j) * g_of_w_(i, j) *
              kij / (constant_ * constant_);
          // wl.dot(delta_x_ij)delta_x_ij is the Aijw term in equation 13
          w_gradient += scalar_term * w_l.dot(delta_x_ij) * delta_x_ij;
          // XILI
          if (output) {
            if (i < 5 && j < 5) {

              T temp = w_l.dot(delta_x_ij);
//            std::cout << "(" << i << ", " << j << "): \n";
//            util::Print(w_l, "w_l");
//            util::Print(delta_x_ij, "delta_x_ij");
              std::cout << "(" << i << ", " << j << "): "
                        << "gamma: " << gamma_matrix_(i, j)
                        << ", g_of_w: " << g_of_w_(i, j)
                        << ", kij: " << kij
                        << ", projection: " << projection
                        << ", scalar: " << scalar_term
                        << ", w*delta: " << temp << std::endl;
              if (i == 0 && j == 1)
                util::Print(delta_x_ij, "delta01");
            }
          }
          // XILI
        }
      }
    }
    return w_gradient;
  }

};
}  // namespace NICE

#endif  // CPP_INCLUDE_KDAC_CPU_H