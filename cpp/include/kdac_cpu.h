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


 private:
  Matrix<T> waw_matrix_;
  Matrix<T> waf_matrix_;
  Matrix<T> faf_matrix_;

  void InitYW() {
    KDAC<T>::InitYW();
    // Coefficients for calculating phi
    waw_matrix_ = Matrix<T>::Zero(n_, n_);
    waf_matrix_ = Matrix<T>::Zero(n_, n_);
    faf_matrix_ = Matrix<T>::Zero(n_, n_);
  }

  void InitXYW(const Matrix <T> &input_matrix,
               const Matrix <T> &y_matrix) {
    KDAC<T>::InitXYW(input_matrix, y_matrix);
    // Coefficients for calculating phi
    waw_matrix_ = Matrix<T>::Zero(n_, n_);
    waf_matrix_ = Matrix<T>::Zero(n_, n_);
    faf_matrix_ = Matrix<T>::Zero(n_, n_);
  }

  void OptimizeW() {
    KDAC<T>::GenGammaMatrix();
    KDAC<T>::GenGofW();
    KDAC<T>::OptimizeW();
  }

  Matrix<T> GenAij(int i, int j) {
    Vector<T> delta_x_ij = x_matrix_.row(i) - x_matrix_.row(j);
    return delta_x_ij * delta_x_ij.transpose();
  }

  // Generate phi(alpha), phi(0) and phi'(0) for LineSearch
  // If this is the first time to generate phi(), then w_l_changed is true
  // Or if the w_l is negated because phi'(0) is negative,
  // then w_l_changed is true
  // If w_l_changed is true, generate phi(0) and phi'(0), otherwise
  // when we are only computing phi(alpha) with a different alpha in the loop
  // of the LineSearch, the w_l_changed is false and we do not generate
  // new waw, waf and faf
  void GenPhi(const Vector<T> &w_l,
              const Vector<T> &gradient,
              bool w_l_changed) {
    // Count number of times GenPhi is called inside one OptimizeW()
    if (kernel_type_ == kGaussianKernel) {
      profiler_["gen_phi"].Start();
      float alpha_square = alpha_ * alpha_;
      float sqrt_one_minus_alpha = std::sqrt(1 - alpha_square);
      float denom = -1.f / (2 * constant_ * constant_);
      phi_of_alpha_ = 0;
      if (w_l_changed) {
        GenPhiCoeff(w_l, gradient);
        phi_of_zero_ = 0;
        phi_of_zero_prime_ = 0;
      }
//      Matrix<T> kij_matrix =
//          denom * ( (faf_matrix_ - waw_matrix_) * alpha_square +
//              2 * waf_matrix_ * sqrt_one_minus_alpha * alpha_ +
//              waw_matrix_);
//      phi_of_alpha_ =
//          (kij_matrix.array().exp() * gamma_matrix_.array()).
//              matrix().sum();
//      if (w_l_changed) {
//        Matrix<T> kij_matrix = denom * waw_matrix_.array().exp().matrix();
//        phi_of_zero_ =
//            (kij_matrix.array() * gamma_matrix_.array()).
//            matrix().sum();
//        phi_of_zero_prime_ = (kij_matrix.array() * waf_matrix_.array() *
//            gamma_matrix_.array() * 2 * denom).
//            matrix().sum();
//      }
      for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
          T waw = waw_matrix_(i, j);
          T waf = waf_matrix_(i, j);
          T faf = faf_matrix_(i, j);
          T kij = exp(denom * ((faf - waw) * alpha_square +
              2 * waf * sqrt_one_minus_alpha * alpha_ + waw));
          phi_of_alpha_ += gamma_matrix_(i, j) * kij;
          if (w_l_changed) {
            T kij = exp(denom * waw);
            phi_of_zero_ += gamma_matrix_(i, j) * kij;
            phi_of_zero_prime_ +=
                gamma_matrix_(i, j) * denom * 2 * waf * kij;
          }
        }
      }
      profiler_["gen_phi"].Record();
    }
  }

  // TODO: make GenPhiOFAlpha return phi(0), instead of modifying class member
  // variable
  T GenPhiOfAlpha(const Vector<T> &w_l) {
    // TODO: Optimize g(w)
    T phi_of_alpha = 0;
    if (kernel_type_ == kGaussianKernel) {
      T denom = -1.f / (2 * constant_ * constant_);
      for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
          Vector<T> delta_x_ij = x_matrix_.row(i) - x_matrix_.row(j);
          T projection = w_l.dot(delta_x_ij);
          T kij = exp(denom * projection * projection);
          // g_of_w_(i,j) is the exp(-waw/2sigma^2) for all previously genreated
          // w columns (see equation 12)
          phi_of_alpha += gamma_matrix_(i, j) * kij * g_of_w_(i, j);
        }
      }
    }
    return phi_of_alpha;
  }

  void GenPhiCoeff(const Vector<T> &w_l, const Vector<T> &gradient) {
    // Three terms used to calculate phi of alpha
    // They only change if w_l or gradient change
    for (int i = 0; i < n_; i++) {
      for (int j = 0; j < n_; j++) {
        Vector<T> delta_x_ij =
            x_matrix_.row(i) - x_matrix_.row(j);
        T delta_w = w_l.transpose() * delta_x_ij;
        T delta_f = delta_x_ij.transpose() * gradient;
        waw_matrix_(i, j) = delta_w * delta_w;
        waf_matrix_(i, j) = delta_w * delta_f;
        faf_matrix_(i, j) = delta_f * delta_f;
//        waw_matrix_(i, j) = w_l.transpose() * a_matrix_ij * w_l;
//        waf_matrix_(i, j) = w_l.transpose() * a_matrix_ij * gradient;
//        faf_matrix_(i, j) = gradient.transpose() * a_matrix_ij * gradient;
      }
    }
  }

  Vector<T> GenWGradient(const Vector<T> &w_l) {
    profiler_["gen_grad"].Start();
    Vector<T> w_gradient = Vector<T>::Zero(d_);
    float sigma_sq = constant_ * constant_;
    if (kernel_type_ == kGaussianKernel) {
      for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
          Vector<T> delta_x_ij =
              x_matrix_.row(i) - x_matrix_.row(j);
          T delta_w = w_l.dot(delta_x_ij);
          T waw = delta_w * delta_w;
          T exp_term = exp(-waw / (2.0 * sigma_sq));
          T gamma = gamma_matrix_(i, j);
          T g_of_w = g_of_w_(i, j);
          w_gradient += -exp_term * gamma * g_of_w / sigma_sq *
              delta_w * delta_x_ij;
//          T exp_term = exp(static_cast<T>(-w_l.transpose() * a_matrix_ij * w_l)
//                               / (2.0 * pow(constant_, 2)));
//          w_gradient += -(gamma_matrix_(i, j)) * (g_of_w_(i, j))
//              * exp_term * a_matrix_ij * w_l / pow(constant_, 2);
        }
      }
    }
    profiler_["gen_grad"].Record();
    return w_gradient;
  }

  void UpdateGOfW(const Vector<T> &w_l) {
    profiler_["update_g_of_w"].Start();
    float sigma_sq = pow(constant_, 2);
    for (int i = 0; i < n_; i++) {
      for (int j = 0; j < n_; j++) {
        if (kernel_type_ == kGaussianKernel) {
          Vector<T> delta_x_ij =
              x_matrix_.row(i) - x_matrix_.row(j);
          T delta_w = w_l.transpose() * delta_x_ij;
          T waw = delta_w * delta_w;
          T exp_term = exp(-waw / (2.0 * sigma_sq));
          g_of_w_(i, j) *= exp_term;
        }
      }
    }
    profiler_["update_g_of_w"].Record();
  }
};
}  // namespace NICE

#endif  // CPP_INCLUDE_KDAC_CPU_H