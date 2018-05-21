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

// Kernel Dimension Alternative Clustering (KDAC base class)
// Please refer to the paper published in PAMI by Liu, Dy and Jordan at:
// http://people.eecs.berkeley.edu/~jordan/papers/niu-dy-jordan-pami.pdf
// We try to follow naming conventions in the paper as much as possible.
// The lower cased variable names is the same as in the paper, and the
// upper cased matrix variable names in the paper are converted to lower
// case suffixed with "_matrix". For example:
// matrix U in the paper is named u_matrix in this implementation.

#ifndef CPP_INCLUDE_KDAC_H
#define CPP_INCLUDE_KDAC_H

#include <functional>
#include <vector>
#include <cmath>
#include <valarray>
#include <tgmath.h>
#include <numeric>
#include <iomanip>
#include <map>
#include "include/matrix.h"
#include "include/vector.h"
#include "include/cpu_operations.h"
#include "include/gpu_operations.h"
#include "include/svd_solver.h"
#include "include/kmeans.h"
#include "Eigen/Core"
#include "include/util.h"
#include "include/kernel_types.h"
#include "include/acl.h"

namespace Nice {
template<typename T>
class KDAC : public ACL<T> {
 public:
  using ACL<T>::c_;
  using ACL<T>::q_;
  using ACL<T>::n_;
  using ACL<T>::d_;
  using ACL<T>::lambda_;
  using ACL<T>::alpha_;
  using ACL<T>::kernel_type_;
  using ACL<T>::constant_;
  using ACL<T>::u_converge_;
  using ACL<T>::w_converge_;
  using ACL<T>::u_w_converge_;
  using ACL<T>::threshold1_;
  using ACL<T>::threshold2_;
  using ACL<T>::x_matrix_;
  using ACL<T>::w_matrix_;
  using ACL<T>::pre_w_matrix_;
  using ACL<T>::u_matrix_;
  using ACL<T>::pre_u_matrix_;
  using ACL<T>::verbose_;
  using ACL<T>::debug_;
  using ACL<T>::max_time_exceeded_;
  using ACL<T>::max_time_;
  using ACL<T>::method_;
  using ACL<T>::mode_;
  using ACL<T>::clustering_result_;
  using ACL<T>::u_matrix_normalized_;
  using ACL<T>::y_matrix_;
  using ACL<T>::y_matrix_temp_;
  using ACL<T>::d_i_;
  using ACL<T>::l_matrix_;
  using ACL<T>::h_matrix_;
  using ACL<T>::k_matrix_y_;
  using ACL<T>::k_matrix_;
  using ACL<T>::d_matrix_;
  using ACL<T>::d_matrix_to_the_minus_half_;
  using ACL<T>::d_ii_;
  using ACL<T>::didj_matrix_;
  using ACL<T>::gamma_matrix_;
  using ACL<T>::profiler_;
  using ACL<T>::GenDegreeMatrix;
  using ACL<T>::GenKernelMatrix;
  using ACL<T>::OptimizeU;
  using ACL<T>::RunKMeans;
  using ACL<T>::InitYW;
  using ACL<T>::CheckMaxTime;
  using ACL<T>::OutputProgress;
  using ACL<T>::InitXYW;
  using ACL<T>::vectorization_;

  /// This is the default constructor for KDAC
  /// Number of clusters c and reduced dimension q will be both set to 0
  KDAC() :
      y_matrix_tilde_(),
      g_of_w_(),
      phi_of_alpha_(0),
      phi_of_zero_(0),
      phi_of_zero_prime_(0),
      first_time_gen_u_(true)
  {
    method_ = "KDAC";
  }

  ~KDAC() {}
  KDAC(const KDAC &rhs) {}



  void Fit(const Matrix <T> &input_matrix) {
    InitX(input_matrix);
    // When there is no Y, it is the the first round when the second term
    // lambda * HSIC is zero, we do not need to optimize W, and we directly
    // go to kmeans where Y_0 is generated. And both u and v are converged.

    // If this is the first time to generate matrix U
    // then we just use the input X matrix to generate the
    // kernel matrix
    // Because Fit(X) is called when there is no w_matrix generated
    // the kernel matrix is then just the Kernel of X itself
    GenKernelMatrix(x_matrix_);
    GenDegreeMatrix();
    OptimizeU();
    RunKMeans();
  }

  // Fit() with an empty param list can only be run when the X and Y already
  // exist from the previous round of computation
  void Fit() {
    // Only changes w_matrix and y_tilde matrix
    profiler_["fit"].Start();
    profiler_["exit_timer"].Start();
    PROFILE(InitYW(), profiler_["init"]);
    while (!u_w_converge_ && !max_time_exceeded_) {
      pre_u_matrix_ = u_matrix_;
      pre_w_matrix_ = w_matrix_;
      // When Fit() is called, we already have a w matrix
      // we project X to subspace W (n * d to d * q)
      // Generate the kernel matrix based on kernel type from projected X
      Matrix <T> projected_matrix = x_matrix_ * w_matrix_;
      GenKernelMatrix(projected_matrix);
      GenDegreeMatrix();
      PROFILE(OptimizeU(), profiler_["u"]);
      PROFILE(OptimizeW(), profiler_["w"]);

      u_converge_ = util::CheckConverged(u_matrix_, pre_u_matrix_,
                                         threshold2_);
      w_converge_ = util::CheckConverged(w_matrix_, pre_w_matrix_,
                                         threshold2_);
      u_w_converge_ = u_converge_ && w_converge_;
      CheckMaxTime();

      if (verbose_)
        OutputProgress();
    }
    PROFILE(RunKMeans(), profiler_["kmeans"]);
    profiler_["fit"].Stop();
  }

  /// This function creates an alternative clustering result
  /// \param input_matrix
  /// The input matrix of n samples and d features where each row
  /// represents a sample
  /// \param y_matrix
  /// The binary matrix of n x (c0 + c1 + c2 + ...) which represent
  /// the previous cluster assignments
  /// \return
  /// It only generates the clustering result but does not returns it
  /// Users can use Predict() to get the clustering result returned
  void Fit(const Matrix <T> &input_matrix, const Matrix <T> &y_matrix) {
    // This is called when we have existing labels Y
    // now we are generating an alternative view with a
    // given Y_previous by doing Optimize both W and U until they converge
    // KDAC method follows the pseudo code in Algorithm 1 in the paper
    // ISM method follows the Appendix I in Chieh's paper
//    PROFILE(InitXYW(input_matrix, y_matrix), profiler_["init"]);
    profiler_["fit"].Start();
    profiler_["exit_timer"].Start();
    InitXYW(input_matrix, y_matrix);
    while (!u_w_converge_ && !max_time_exceeded_) {
      pre_u_matrix_ = u_matrix_;
      pre_w_matrix_ = w_matrix_;
      // So this is the first time to generate matrix U
      // the kernel matrix is then just the Kernel of X itself
      // After U is generated, W is generated following it
      // then we use the projection matrix XW to generate kernel matrix
      if (first_time_gen_u_) {
        GenKernelMatrix(x_matrix_);
        first_time_gen_u_ = false;
      } else {
        Matrix<T> projected_matrix = x_matrix_ * w_matrix_;
        GenKernelMatrix(projected_matrix);
      }
      GenDegreeMatrix();
      PROFILE(OptimizeU(), profiler_["u"]);
      PROFILE(OptimizeW(), profiler_["w"]);
      u_converge_ =
          util::CheckConverged(u_matrix_, pre_u_matrix_, threshold2_);
      w_converge_ =
          util::CheckConverged(w_matrix_, pre_w_matrix_, threshold2_);
      u_w_converge_ = u_converge_ && w_converge_;
      CheckMaxTime();
      if (verbose_)
        OutputProgress();
    }
    PROFILE(RunKMeans(), profiler_["kmeans"]);
    profiler_["fit"].Stop();
  }

 protected:
  Matrix <T> y_matrix_tilde_;  // The kernel matrix for Y
  Matrix <T> g_of_w_;  // g(w) for updating gradient
  // in formula 5
  T phi_of_alpha_, phi_of_zero_, phi_of_zero_prime_;
  // If this is the first time to generate a matrix U
  // If this is true, it means the kernel matrix is generated directly from
  // the input matrix X
  bool first_time_gen_u_;

  virtual void InitX(const Matrix <T> &input_matrix) {
    ACL<T>::InitX(input_matrix);
  }

  void InitW() {
    // When the user does not initialize W using SetW()
    // W matrix is initilized to be a cut-off identity matrix in KDAC method
    if (w_matrix_.cols() == 0) {
      w_matrix_ = Matrix<T>::Identity(d_, q_);
    }
  }

  // Initialization for all Y related data structures
  virtual void InitY(const Matrix<T> &y_matrix) {
    ACL<T>::InitY(y_matrix);
    // Generate Y tilde matrix in equation 5 from kernel matrix of Y
    y_matrix_tilde_ = h_matrix_ * k_matrix_y_ * h_matrix_;
  }

  Vector <T> GenOrthogonal(const Matrix <T> &space,
                           const Vector <T> &vector) {
    Vector <T> projection = Vector<T>::Zero(space.rows());
    for (int j = 0; j < space.cols(); j++) {
      // projection = (v * u / u^2) * u
      projection += (vector.dot(space.col(j)) /
          space.col(j).squaredNorm()) * space.col(j);
    }
    return vector - projection;
  }

  Vector <T> GenOrthonormal(const Matrix <T> &space,
                            const Vector <T> &vector) {
    util::CheckFinite(space, "space");
    Vector <T> ortho_vector = GenOrthogonal(space, vector);
    util::CheckFinite(ortho_vector, "ortho_vector");
    T norm = ortho_vector.norm();
    if(norm == 0) {
      return ortho_vector;
    }
    return ortho_vector.array() / norm;
  }

  void GenGammaMatrix(void) {
    // didj matrix contains the element (i, j) that equal to d_i * d_j
    didj_matrix_ = d_i_ * d_i_.transpose();
    // Generate the Gamma matrix in equation 5, which is a constant since
    // we have U fixed. Note that instead of generating one element of
    // gamma_ij on the fly as in the paper, we generate the whole gamma matrix
    // at one time and then access its entry of (i, j)
    // This is an element-wise operation
    gamma_matrix_ = ((u_matrix_ * u_matrix_.transpose()).array() /
        didj_matrix_.array()).matrix() - y_matrix_tilde_ * lambda_;
    if (debug_) {
      util::CheckFinite(u_matrix_, "u_matrix_");
      util::CheckFinite(didj_matrix_, "didj_matrix_");
      util::CheckFinite(y_matrix_tilde_, "y_matrix_tilde_");
      std::string out_path =
          "/home/xiangyu/Dropbox/git_project/NICE/python/debug/output/";
//      util::ToFile(gamma_matrix_,
//                   out_path + "gamma_kdac_" + mode_ + "_"
//                       + std::to_string(outer_iter_num_) + "_"
//                       + std::to_string(inner_iter_num_) + ".csv");
    }
  }

  void GenGofW() {
    // After gamma_matrix is generated, we are optimizing gamma * kij as in 5
    // g_of_w is g(w_l) that is multiplied by g(w_(l+1)) in each iteration
    // of changing l.
    // Note that here the g_of_w is a n*n matrix because it contains A_ij
    // g_of_w(i, j) corresponding to exp(-w_T * A_ij * w / 2sigma^2)
    // When l = 0, g_of_w is 1
    // when l = 1, g_of_w is 1 .* g(w_1)
    // when l = 2, g_of_w is 1 .* g(w_1) .* g(w_2)...
    g_of_w_ = Matrix<T>::Constant(this->n_, this->n_, 1);
  }

  virtual void OptimizeW() {
    // We optimize each column in the W matrix
    for (int l = 0; l < w_matrix_.cols(); l++) {
      Vector <T> w_l;
      // Number of iterations in converging w_l
      // Get orthogonal to make w_l orthogonal to vectors from w_0 to w_(l-1)
      // when l is not 0
      if (l == 0) {
        w_l = w_matrix_.col(l);
        w_l = w_l.array() / w_l.norm();
      } else {
        w_l = GenOrthonormal(w_matrix_.leftCols(l), w_matrix_.col(l));
      }
      // Search for the w_l that maximizes formula 5
      // The initial objective is set to the lowest number
      T objective = std::numeric_limits<T>::lowest();
      bool w_l_converged = false;
      while (!w_l_converged) {
        Vector <T> grad_f_vertical;
        T pre_objective = objective;
        // Calculate the w gradient in equation 13, then find the gradient
        // that is vertical to the space spanned by w_0 to w_l
        Vector <T> grad_f = GenWGradient(w_l);
        grad_f_vertical =
            GenOrthonormal(w_matrix_.leftCols(l + 1), grad_f);
        LineSearch(grad_f_vertical, &w_l, &objective);
        w_l = std::sqrt(1.0 - alpha_ * alpha_) * w_l + alpha_ * grad_f_vertical;
        Matrix<T> leftCols = w_matrix_.leftCols(l+1);
        util::CheckFinite(leftCols, "leftCols");
        util::CheckFinite(grad_f, "grad_f");
        util::CheckFinite(grad_f_vertical, "grad_f_vertical");
        util::CheckFinite(w_l, "w_l");
        w_matrix_.col(l) = w_l;
        w_l_converged =
            util::CheckConverged(objective, pre_objective, threshold2_);
      }
      UpdateGOfW(w_l);
      // TODO: Need to learn about if using Vector<T> &w_l = w_matrix_.col(l)
      if (verbose_)
        std::cout << "Column " << l + 1 << " cost: " << objective << " | ";
//      std::cout << objective << ", ";
    }
    if (verbose_)
      std::cout << "W Optimized" << std::endl;

    profiler_["gen_phi"].SumRecords();
    profiler_["gen_grad"].SumRecords();
    profiler_["update_g_of_w"].SumRecords();
  }

  void LineSearch(const Vector <T> &gradient,
                  Vector <T> *w_l, T *objective) {
    alpha_ = 1.0;
    float a1 = 0.1;
    float rho = 0.8;

    if (kernel_type_ == kGaussianKernel) {
      GenPhi(*w_l, gradient, true);
      if (phi_of_zero_prime_ < 0) {
        *w_l = -(*w_l);
        GenPhi(*w_l, gradient, true);
      }
      while ((phi_of_alpha_ < phi_of_zero_ + alpha_ * a1 * phi_of_zero_prime_)) {
        alpha_ = alpha_ * rho;
        GenPhi(*w_l, gradient, false);
      }
      *objective = phi_of_alpha_;
    }
  }

  Matrix <T> GetYTilde() { return y_matrix_tilde_; }

  void CheckFiniteOptimizeW() {
    util::CheckFinite(didj_matrix_, "didj");
    util::CheckFinite(gamma_matrix_, "Gamma");
    util::CheckFinite(w_matrix_, "W");
  }


  virtual void UpdateGOfW(const Vector <T> &w_l) = 0;

  virtual void GenPhi(const Vector <T> &w_l,
                      const Vector <T> &gradient,
                      bool w_l_changed) = 0;

  virtual void GenPhiCoeff(const Vector <T> &w_l, const Vector <T> &gradient) = 0;
  virtual Vector <T> GenWGradient(const Vector <T> &w_l) = 0;

};
}  // namespace NICE

#endif  // CPP_INCLUDE_KDAC_H
