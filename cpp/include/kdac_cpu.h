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
template<typename T>
class KDACCPU: public KDAC<T> {
 public:
  /// This is the default constructor for KDAC
  /// Number of clusters c and reduced dimension q will be both set to 2
  KDACCPU() {}

  ~KDACCPU() {}

  KDACCPU(const KDACCPU &rhs) {}
//  KDACCPU &operator=(const KDACCPU &rhs) {}


 private:
  Matrix<T> waw_matrix_;
  Matrix<T> waf_matrix_;
  Matrix<T> faf_matrix_;

  void Init(const Matrix<T> &input_matrix) {
    KDAC<T>::Init(input_matrix);
    // Coefficients for calculating phi
    waw_matrix_ = Matrix<T>::Zero(this->n_, this->n_);
    waf_matrix_ = Matrix<T>::Zero(this->n_, this->n_);
    faf_matrix_ = Matrix<T>::Zero(this->n_, this->n_);
  }

  // Initialization for generating alternative views with a given Y
  void Init(const Matrix<T> &input_matrix, const Matrix<T> &y_matrix) {
    KDAC<T>::Init(input_matrix, y_matrix);
    // Coefficients for calculating phi
    waw_matrix_ = Matrix<T>::Zero(this->n_, this->n_);
    waf_matrix_ = Matrix<T>::Zero(this->n_, this->n_);
    faf_matrix_ = Matrix<T>::Zero(this->n_, this->n_);
  }



  void OptimizeW(void) {
    KDAC<T>::GenGammaMatrix();
    KDAC<T>::GenGofW();
    KDAC<T>::OptimizeW();
  }

  Matrix<T> GenAij(int i, int j) {
    Vector<T> delta_x_ij = this->x_matrix_.row(i) - this->x_matrix_.row(j);
    return delta_x_ij * delta_x_ij.transpose();
  }

  void GenPhiCoeff(const Vector<T> &w_l, const Vector<T> &gradient) {
    // Three terms used to calculate phi of alpha
    // They only change if w_l or gradient change
    for (int i = 0; i < this->n_; i++) {
      for (int j = 0; j < this->n_; j++) {
        Vector<T> delta_x_ij =
            this->x_matrix_.row(i) - this->x_matrix_.row(j);
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
    if (this->kernel_type_ == kGaussianKernel) {
      this->profiler_.gen_phi.Start();
      float alpha_square = pow(this->alpha_, 2);
      float sqrt_one_minus_alpha = pow((1 - alpha_square), 0.5);
      float denom = -1 / (2 * pow(this->constant_, 2));
      this->phi_of_alpha_ = 0;
      if (w_l_changed) {
        GenPhiCoeff(w_l, gradient);
        this->phi_of_zero_ = 0;
        this->phi_of_zero_prime_ = 0;
      }
//      Matrix<T> kij_matrix =
//          denom * ( (faf_matrix_ - waw_matrix_) * alpha_square +
//              2 * waf_matrix_ * sqrt_one_minus_alpha * this->alpha_ +
//              waw_matrix_);
//      this->phi_of_alpha_ =
//          (kij_matrix.array().exp() * this->gamma_matrix_.array()).
//              matrix().sum();
//      if (w_l_changed) {
//        Matrix<T> kij_matrix = denom * waw_matrix_.array().exp().matrix();
//        this->phi_of_zero_ =
//            (kij_matrix.array() * this->gamma_matrix_.array()).
//            matrix().sum();
//        this->phi_of_zero_prime_ = (kij_matrix.array() * waf_matrix_.array() *
//            this->gamma_matrix_.array() * 2 * denom).
//            matrix().sum();
//      }
      for (int i = 0; i < this->n_; i++) {
        for (int j = 0; j < this->n_; j++) {
          T waw = waw_matrix_(i, j);
          T waf = waf_matrix_(i, j);
          T faf = faf_matrix_(i, j);
          T kij = exp(denom * ((faf - waw) * alpha_square +
              2 * waf * sqrt_one_minus_alpha * this->alpha_ + waw));
          this->phi_of_alpha_ += this->gamma_matrix_(i, j) * kij;
          if (w_l_changed) {
            T kij = exp(denom * waw);
            this->phi_of_zero_ += this->gamma_matrix_(i, j) * kij;
            this->phi_of_zero_prime_ +=
                this->gamma_matrix_(i, j) * denom * 2 * waf * kij;
          }
        }
      }
      this->profiler_.gen_phi.Record();
    }
  }

  Vector<T> GenWGradient(const Vector<T> &w_l) {
    this->profiler_.gen_grad.Start();
    Vector<T> w_gradient = Vector<T>::Zero(this->d_);
    float sigma_sq = pow(this->constant_, 2);
    if (this->kernel_type_ == kGaussianKernel) {
      for (int i = 0; i < this->n_; i++) {
        for (int j = 0; j < this->n_; j++) {
          Vector<T> delta_x_ij =
              this->x_matrix_.row(i) - this->x_matrix_.row(j);
          T delta_w = w_l.transpose() * delta_x_ij;
          T waw = delta_w * delta_w;
          T exp_term = exp(-waw / (2.0 * sigma_sq));
          T gamma = this->gamma_matrix_(i, j);
          T g_of_w = this->g_of_w_(i, j);
          w_gradient += -exp_term * gamma * g_of_w / sigma_sq *
              delta_w * delta_x_ij;
//          T exp_term = exp(static_cast<T>(-w_l.transpose() * a_matrix_ij * w_l)
//                               / (2.0 * pow(this->constant_, 2)));
//          w_gradient += -(this->gamma_matrix_(i, j)) * (this->g_of_w_(i, j))
//              * exp_term * a_matrix_ij * w_l / pow(this->constant_, 2);
        }
      }
    }
    this->profiler_.gen_grad.Record();
    return w_gradient;
  }

  void UpdateGOfW(const Vector<T> &w_l) {
    this->profiler_.update_g_of_w.Start();
    float sigma_sq = pow(this->constant_, 2);
    for (int i = 0; i < this->n_; i++) {
      for (int j = 0; j < this->n_; j++) {
        if (this->kernel_type_ == kGaussianKernel) {
          Vector<T> delta_x_ij =
              this->x_matrix_.row(i) - this->x_matrix_.row(j);
          T delta_w = w_l.transpose() * delta_x_ij;
          T waw = delta_w * delta_w;
          T exp_term = exp(-waw / (2.0 * sigma_sq));
          this->g_of_w_(i, j) *= exp_term;
        }
      }
    }
    this->profiler_.update_g_of_w.Record();
  }
};
}  // namespace NICE

#endif  // CPP_INCLUDE_KDAC_CPU_H
