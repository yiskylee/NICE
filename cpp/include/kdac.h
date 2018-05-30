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
  using ACL<T>::d_matrix_to_the_minus_half_;
  using ACL<T>::d_ii_;
  using ACL<T>::didj_matrix_;
  using ACL<T>::gamma_matrix_;
  using ACL<T>::profiler_;
  using ACL<T>::GenDegreeMatrix;
  using ACL<T>::GenKernelMatrix;
  using ACL<T>::RunKMeans;
  using ACL<T>::InitYW;
  using ACL<T>::CheckMaxTime;
  using ACL<T>::OutputProgress;
  using ACL<T>::InitXYW;
  using ACL<T>::vectorization_;
  using ACL<T>::CheckFiniteOptimizeU;

  /// This is the default constructor for KDAC
  /// Number of clusters c and reduced dimension q will be both set to 0
  KDAC() :
      y_matrix_tilde_(),
      g_of_w_(),
      new_g_of_w_(),
      phi_of_alpha_(0),
      phi_of_zero_(0),
      phi_of_zero_prime_(0)
  {
    method_ = "KDAC";
  }

  ~KDAC() {}
  KDAC(const KDAC &rhs) {}

  Matrix <T> GetYTilde() { return y_matrix_tilde_; }

  void OutputCostChieh() {

    Matrix<T> hdkdh = h_matrix_ * d_matrix_to_the_minus_half_ * k_matrix_ *
        d_matrix_to_the_minus_half_ * h_matrix_;
    Matrix<T> uhdkdhu = u_matrix_.transpose() * hdkdh * u_matrix_;
    Matrix<T> yhdkdhy = y_matrix_.transpose() * hdkdh * y_matrix_;

    T hsic_first_term = uhdkdhu.trace();
    T hsic_second_term = yhdkdhy.trace();

    std::cout << "["
              << hsic_first_term << ", "
              << hsic_second_term<< "], \n ";
  }

  void OutputCostDonglin() {
    Matrix<T> udkdu = u_matrix_.transpose() * d_matrix_to_the_minus_half_ *
        k_matrix_ *
        d_matrix_to_the_minus_half_ * u_matrix_;
    Matrix<T> yhkhy = y_matrix_.transpose() * h_matrix_ *
        k_matrix_ *
        h_matrix_ * y_matrix_;

    T hsic_first_term = udkdu.trace();
    T hsic_second_term = yhkhy.trace();

    std::cout << "["
              << hsic_first_term << ", "
              << hsic_second_term<< "], \n ";
  }

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

    // When Fit() is called, the U matrix is already generated
    // So we optimize W first
    PROFILE(OptimizeW(), profiler_["w"]);
    if (verbose_)
      OutputProgress();

    while (!u_w_converge_ && !max_time_exceeded_) {
      pre_u_matrix_ = u_matrix_;
      pre_w_matrix_ = w_matrix_;

//      Matrix<T> projected_matrix = x_matrix_ * w_matrix_;
//      GenKernelMatrix(projected_matrix);
//
//      // XILI
//      Matrix<T> diff = (k_matrix_ - g_of_w_);
//      util::Print(diff.norm(), "diff");
//      // XILI

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
    GenKernelMatrix(x_matrix_);
    GenDegreeMatrix();
    PROFILE(OptimizeU(), profiler_["u"]);
    PROFILE(OptimizeW(), profiler_["w"]);
    if (verbose_)
      OutputProgress();

    while (!u_w_converge_ && !max_time_exceeded_) {
      pre_u_matrix_ = u_matrix_;
      pre_w_matrix_ = w_matrix_;
//      // We first generate U and W for the first time before the loop
//      // Now with W matrix generated, we use projection matrix
//      Matrix<T> projected_matrix = x_matrix_ * w_matrix_;
//      GenKernelMatrix(projected_matrix);
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
  // new g(w) to hold g(w) * temporary exp(-waw/2sigma^2)
  // new_g_of_w becomes g_of_w when converged w is found
  Matrix <T> new_g_of_w_;
  // in formula 5
  T phi_of_alpha_, phi_of_zero_, phi_of_zero_prime_;

  virtual void InitX(const Matrix <T> &input_matrix) {
    ACL<T>::InitX(input_matrix);
    g_of_w_ = Matrix<T>::Constant(n_, n_, 1);
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
    util::CheckFinite(space, "space", true);
    Vector <T> ortho_vector = GenOrthogonal(space, vector);
    util::CheckFinite(ortho_vector, "ortho_vector");
    T norm = ortho_vector.norm();
    if (std::abs(norm) < 10e-3) {
      std::cerr << "Vector is close to zero" << std::endl;
      exit(1);
    }
    return ortho_vector.array() / norm;
  }

  void GenGammaMatrix() {
    // Generate the Gamma matrix in equation 5, which is a constant since
    // we have U fixed. Note that instead of generating one element of
    // gamma_ij on the fly as in the paper, we generate the whole gamma matrix
    // at one time and then access its entry of (i, j)
    // This is an element-wise operation

    // Donglin's Gamma Matrix
    gamma_matrix_ = ((u_matrix_ * u_matrix_.transpose()).array() /
        didj_matrix_.array()).matrix() - y_matrix_tilde_ * lambda_;

//    if (debug_) {
//      util::CheckFinite(u_matrix_, "u_matrix_");
//      util::CheckFinite(didj_matrix_, "didj_matrix_");
//      util::CheckFinite(y_matrix_tilde_, "y_matrix_tilde_");
//      std::string out_path =
//          "/home/xiangyu/Dropbox/git_project/NICE/python/debug/output/";
//      util::ToFile(gamma_matrix_,
//                   out_path + "gamma_kdac_" + mode_ + "_"
//                       + std::to_string(outer_iter_num_) + "_"
//                       + std::to_string(inner_iter_num_) + ".csv");
//    }
  }

  void OptimizeU() {
    l_matrix_ = k_matrix_.array() / didj_matrix_.array();
    Eigen::SelfAdjointEigenSolver <Matrix<T>> solver(l_matrix_);
    Vector <T> eigen_values = solver.eigenvalues().real();
    std::vector <T>
        v(eigen_values.data(), eigen_values.data() + eigen_values.size());
    std::vector <size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&v](size_t t1, size_t t2) { return v[t1] > v[t2]; });
    u_matrix_ = Matrix<T>::Zero(n_, c_);
    Vector <T> eigen_vector = Vector<T>::Zero(n_);
    for (int i = 0; i < c_; i++) {
      eigen_vector = solver.eigenvectors().col(idx[i]).real();
      u_matrix_.col(i) = eigen_vector;
    }
    if (verbose_)
      std::cout << "U Optimized\n";
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
      } else {
        w_l = GenOrthonormal(w_matrix_.leftCols(l), w_matrix_.col(l));
        w_matrix_.col(l) = w_l;
      }
      // Search for the w_l that maximizes formula 5
      // The initial objective is set to the lowest number
      phi_of_alpha_ = std::numeric_limits<T>::lowest();
      bool w_l_converged = false;

      while (!w_l_converged) {
        // Calculate the w gradient in equation 13, then find the gradient
        // that is vertical to the space spanned by w_0 to w_l
        // XILI
        Vector <T> grad_f = GenWGradient(w_l);
//        if (l == 2)
//          grad_f = GenWGradient(w_l, true);
//        else
//        Matrix<T> leftCols = w_matrix_.leftCols(l + 1);
//        util::Print(leftCols, "leftCols");
//        util::Print(grad_f, "grad_f");
//        // XILI
        Vector <T> grad_f_vertical =
            GenOrthonormal(w_matrix_.leftCols(l + 1), grad_f);

        //XILI
//        if (l == 2) {
//          std::cout << "Round " << i++ << std::endl;
//          Vector<T> w0 = w_matrix_.col(0);
//          Vector<T> w1 = w_matrix_.col(1);
//          Vector<T> w2 = w_matrix_.col(2);
//          std::cout << "w0 dot w1: " << w0.dot(w1) << std::endl;
//          std::cout << "w0 dot w0: " << w0.dot(w0) << std::endl;
//          std::cout << "w1 dot w1: " << w1.dot(w1) << std::endl;
//          std::cout << "w0 dot w2: " << w0.dot(w2) << std::endl;
//          std::cout << "w1 dot w2: " << w1.dot(w2) << std::endl;
//          std::cout << "w2 dot w2: " << w2.dot(w2) << std::endl;
//          util::Print(w2, "w2");
//          util::Print(grad_f, "grad_f");
//          util::Print(grad_f_vertical, "grad_f_vertical");
//        }
        //XILI
        // Line search a good alpha and update w_l
        LineSearch(grad_f, grad_f_vertical, &w_l);
//        w_l = std::sqrt(1.0 - alpha_ * alpha_) * w_l + alpha_ * grad_f_vertical;
        w_matrix_.col(l) = w_l;
        w_l_converged =
            util::CheckConverged(phi_of_alpha_, phi_of_zero_, threshold2_);
      }
      // After w_l is converged, the new_g_of_w becomes the current g_of_w
      g_of_w_ = new_g_of_w_;


      // TODO: Get rid of UpdateGOfW because it is already calculated in the
      // GenPhiOfAlpha function when w_l is converged, and the final g_of_w
      // equals to the kernel matrix of XW, so when we optimizeU, we don't need
      // to re-generate Kernel Matrix k_matrix_
//      UpdateGOfW(w_l);

      // TODO: Need to learn about if using Vector<T> &w_l = w_matrix_.col(l)
      if (verbose_)
        std::cout << "Column " << l + 1 << " cost: " << phi_of_alpha_ << " | ";

      //XILI
//      if (l == 2) {
//        Vector<T> w0 = w_matrix_.col(0);
//        Vector<T> w1 = w_matrix_.col(1);
//        Vector<T> w2 = w_matrix_.col(2);
//        std::cout << "w0 dot w1: " << w0.dot(w1) << std::endl;
//        std::cout << "w0 dot w0: " << w0.dot(w0) << std::endl;
//        std::cout << "w1 dot w1: " << w1.dot(w1) << std::endl;
//        std::cout << "w0 dot w2: " << w0.dot(w2) << std::endl;
//        std::cout << "w1 dot w2: " << w1.dot(w2) << std::endl;
//        std::cout << "w2 dot w2: " << w2.dot(w2) << std::endl;
//      }
      //XILI
    }
    // After all w columns have been updated, g_of_w becomes kernel
    // matrix, we reset g_of_w to 1 matrix before OptimizeW in the next
    // iteration
    k_matrix_ = g_of_w_;
    g_of_w_ = Matrix<T>::Constant(n_, n_, 1);


    profiler_["gen_phi"].SumRecords();
    profiler_["gen_grad"].SumRecords();
    profiler_["update_g_of_w"].SumRecords();

    if (verbose_)
      std::cout << "W Optimized\n";
  }

  void LineSearch(const Vector<T> gradient,
                  const Vector<T> &gradient_vertical,
                  Vector<T> *w_l) {
    alpha_ = 1;
    float a1 = 0.1;
    float rho = 0.8;
    float alpha_square = alpha_ * alpha_;
    float sqrt_one_minus_alpha = std::sqrt(1 - alpha_square);

    if (kernel_type_ == kGaussianKernel) {
      if (phi_of_alpha_ == std::numeric_limits<T>::lowest()) {
        // When w_l is just initialized, we don't have phi(alpha)
        // We first generate phi(alpha), and make it become the previous
        // objective: phi(0)
        phi_of_zero_ = GenPhiOfAlpha(*w_l);
      } else {
        // When we have already generated phi(alpha),
        // we directly make phi(0) equal to the phi(alpha) from last iteration
        phi_of_zero_ = phi_of_alpha_;
      }
      // phi'(0) is always generated using the gradient of current w_l
      phi_of_zero_prime_ = gradient.dot(gradient_vertical);

      Vector<T> new_w_l = *w_l * sqrt_one_minus_alpha +
          gradient_vertical * alpha_;
      phi_of_alpha_ = GenPhiOfAlpha(new_w_l);

      while (phi_of_alpha_ <
          phi_of_zero_ + a1 * alpha_ * phi_of_zero_prime_) {
        alpha_ *= rho;
        alpha_square = alpha_ * alpha_;
        sqrt_one_minus_alpha = std::sqrt(1-alpha_square);
        new_w_l = *w_l * sqrt_one_minus_alpha + gradient_vertical * alpha_;
        phi_of_alpha_ = GenPhiOfAlpha(new_w_l);
      }

      // Once we have found the alpha, the corresponding new_w_l becomes the
      // current w_l
      *w_l = new_w_l;
//      // TODO: Make phi(0), phi'(0), and phi(alpha) as local variable
      // TODO: Put gradient inside LineSearch
    }
  }



  void CheckFiniteOptimizeW() {
    util::CheckFinite(didj_matrix_, "didj");
    util::CheckFinite(gamma_matrix_, "Gamma");
    util::CheckFinite(w_matrix_, "W");
  }

  virtual void UpdateGOfW(const Vector <T> &w_l) = 0;
  virtual T GenPhiOfAlpha(const Vector<T> &w_l) = 0;
  virtual Vector <T> GenWGradient(const Vector <T> &w_l, bool output=false) = 0;

};
}  // namespace NICE

#endif  // CPP_INCLUDE_KDAC_H