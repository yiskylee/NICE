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
#include "include/kdac_profiler.h"

namespace Nice {
template<typename T>
class KDAC {
 public:
  /// This is the default constructor for KDAC
  /// Number of clusters c and reduced dimension q will be both set to 2
  KDAC() :
      c_(0),
      q_(0),
      n_(0),
      d_(0),
      lambda_(1.0),
      alpha_(1.0),
      kernel_type_(kGaussianKernel),
      constant_(1.0),
      u_converge_(false),
      w_converge_(false),
      u_w_converge_(false),
      threshold1_(0.0001),
      threshold2_(0.001),
      x_matrix_(),
      w_matrix_(),
      pre_w_matrix_(),
      y_matrix_(),
      y_matrix_temp_(),
      y_matrix_tilde_(),
      d_matrix_(),
      d_matrix_to_the_minus_half_(),
      d_ii_(),
      d_i_(),
      didj_matrix_(),
      k_matrix_(),
      k_matrix_y_(),
      u_matrix_(),
      pre_u_matrix_(),
      u_matrix_normalized_(),
      l_matrix_(),
      h_matrix_(),
      gamma_matrix_(),
      g_of_w_(),
      clustering_result_(),
      verbose_(false),
      debug_(false),
      max_time_exceeded_(false),
      max_time_(10),
      method_("KDAC"),
      mode_("pyInterface"),
      vectorization_(true),
      first_time_gen_u_(true),
      cost_vector_(),
      eigen_vals_(),
      outer_iter_num_(0),
      inner_iter_num_(0) {}

  ~KDAC() {}
  KDAC(const KDAC &rhs) {}

  // Set the number of clusters c
  void SetC(int c) { c_ = c; }

  // Set user-defined W
  void SetW(const Matrix <T> &w_matrix) {
    if (w_matrix.cols() != q_ || w_matrix.rows() != d_) {
      std::cerr << "Matrix W must be a " << d_ << " * " << q_ << " matrix\n";
      exit(1);
    }
    w_matrix_ = w_matrix;
  }

  // Set lambda for HSIC
  void SetLambda(float lambda) { lambda_ = lambda; }

  /// Set the reduced dimension q
  void SetQ(int q) { q_ = q; }

  /// Set thresholds
  void SetThreshold1(float thresh1) { threshold1_ = thresh1; }

  void SetThreshold2(float thresh2) { threshold2_ = thresh2; }

  /// Set time limit before breaking out of the program
  void SetMaxTime(int max_time) { max_time_ = max_time; }

  /// Set the kernel type: kGaussianKernel, kPolynomialKernel, kLinearKernel
  /// And set the constant associated the kernel
  void SetKernel(KernelType kernel_type, float constant) {
    kernel_type_ = kernel_type;
    constant_ = constant;
  }

  void SetVerbose(bool verbose) { verbose_ = verbose; }

  void SetVectorization(bool vectorization) { vectorization_ = vectorization; }

  void SetDebug(bool debug) { debug_ = debug; }

  void SetMethod(std::string method) { method_ = method; }

  void SetMode(std::string mode) { mode_ = mode; }

  int GetD(void) { return d_; }

  int GetN(void) { return n_; }

  int GetQ(void) { return q_; }

  int GetC(void) { return c_; }

  Matrix <T> GetU(void) { return u_matrix_; }

  Matrix <T> GetW(void) { return w_matrix_; }

  Matrix <T> GetUNormalized(void) { return u_matrix_normalized_; }

  Matrix <T> GetL(void) { return l_matrix_; }

  Matrix <T> GetDMatrix(void) { return d_matrix_; }

  Matrix <T> GetDToTheMinusHalf(void) { return d_matrix_to_the_minus_half_; }

  Matrix <T> GetK(void) {
    return k_matrix_;
  }

  Matrix <T> GetY(void) { return y_matrix_; }

  Matrix <T> GetYTilde(void) { return y_matrix_tilde_; }

  Matrix <T> GetGamma(void) { return gamma_matrix_; }

  KDACProfiler GetProfiler(void) { return profiler_; }

  void OutputProgress() {
    if (u_converge_ && !w_converge_)
      std::cout << "U Converged | W Not Converged" << std::endl;
    else if (!u_converge_ && w_converge_)
      std::cout << "U Not Converged | W Converged" << std::endl;
    else if (!u_converge_ && !w_converge_)
      std::cout << "U Not Converged | W Not Converged" << std::endl;
    else
      std::cout << "U Converged | W Converged" << std::endl;
  }

  void OutputConfigs() {
    std::cout << "c: " << c_ << std::endl;
    std::cout << "q: " << q_ << std::endl;
    std::cout << "n: " << n_ << std::endl;
    std::cout << "d: " << d_ << std::endl;
    std::cout << "lambda: " << lambda_ << std::endl;
    std::cout << "alpha: " << alpha_ << std::endl;
    std::cout << "sigma: " << constant_ << std::endl;
    std::cout << "threshold1: " << threshold1_ << std::endl;
    std::cout << "threshold2: " << threshold2_ << std::endl;
    std::cout << "vectorizaion: " << vectorization_ << std::endl;
    std::cout << "method: " << method_ << std::endl;
    std::cout << "mode: " << mode_ << std::endl;
    std::cout << "verbose: " << verbose_ << std::endl;
    std::cout << "debug: " << debug_ << std::endl;
  }

  /// This function checks if the total execution time exceeds a pre-defined
  /// value
  void CheckMaxTime() {
    profiler_["exit_timer"].Stop();
    // If the whole program runs for more than 20 hours, it returns
    if (profiler_["exit_timer"].vec_.back() / 1e3 > max_time_) {
      std::cout << "Exceeds maximum time limit: " << max_time_ << std::endl;
      max_time_exceeded_ = true;
    }
  }

  /// This function can be used when the user is not satisfied with
  /// the previous clustering results and want to discard the result from
  /// the last run so she can re-run Fit with new parameters
  void DiscardLastRun() {
    if (debug_)
      util::Print(y_matrix_, "y_matrix_before");
    Matrix <T> y_matrix_new = Matrix<T>::Zero(n_, y_matrix_.cols() - c_);
    y_matrix_new = y_matrix_.leftCols(y_matrix_.cols() - c_);
    y_matrix_ = y_matrix_new;
    if (debug_)
      util::Print(y_matrix_, "y_matrix_after");
  }

  /// This function creates the first clustering result
  /// \param input_matrix
  /// The input matrix of n samples and d features where each row
  /// represents a sample
  /// \return
  /// It only generates the clustering result but does not returns it
  /// Users can use Predict() to get the clustering result returned
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
    // If vectorization is enabled,
    // Q matrix is generated along with the Kernel Matrix
    if (method_ == "ISM" && vectorization_)
      GenKernelAndQMatrix(x_matrix_);
    else
      GenKernelMatrix(x_matrix_);
    GenDegreeMatrix();
    OptimizeU();
    RunKMeans();
  }

  // Fit() with an empty param list can only be run when the X and Y already
  // exist from the previous round of computation
  void Fit(void) {
    // Only changes w_matrix and y_tilde matrix
    profiler_["fit"].Start();
    profiler_["exit_timer"].Start();
    PROFILE(InitYW(), profiler_["init"]);

    if (method_ == "KDAC") {
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
    } else if (method_ == "ISM") {
      outer_iter_num_ = 0;
      Vector<T> pre_eigen_vals;
      while (!u_w_converge_ && !max_time_exceeded_ && outer_iter_num_ < 20) {
        if (verbose_)
          std::cout << "\nOuter Loop " << outer_iter_num_ << std::endl;
        pre_u_matrix_ = u_matrix_;
        pre_w_matrix_ = w_matrix_;
        pre_eigen_vals = eigen_vals_;
        // When Fit() is called, we already have a w matrix
        // we project X to subspace W (n * d to d * q)
        // Generate the kernel matrix based on kernel type from projected X
        Matrix <T> projected_matrix = x_matrix_ * w_matrix_;
        // Q matrix is already generated in Fit(X) based on X
        // so there is no need to do it here using the projected_matrix
        GenKernelMatrix(projected_matrix);
        GenDegreeMatrix();
        PROFILE(OptimizeWISM(), profiler_["w"]);
        PROFILE(OptimizeU(), profiler_["u"]);
//        u_converge_ = util::CheckConverged(u_matrix_, pre_u_matrix_,
//                                           threshold2_);
        // we only check if w matrix converges, u matrix does not matter as
        // there are multiple local minimum points
        u_converge_ = true;
        w_converge_ = util::CheckConverged(eigen_vals_,
                                           pre_eigen_vals, threshold1_);
        if (verbose_) {
          T change1 = static_cast<T>((w_matrix_ - pre_w_matrix_).norm()) /
              static_cast<T>(pre_w_matrix_.norm());
          T change2 = static_cast<T>((eigen_vals_ - pre_eigen_vals).norm()) /
              static_cast<T>(pre_eigen_vals.norm());
          std::cout << "change in W | eigen vals: " << change1 << " | " << change2 << std::endl;
        }

        T mean_cost = cost_vector_.mean();
        Vector <T> temp = cost_vector_.array() - mean_cost;
        temp = temp.array() * temp.array();
        T std = sqrt(temp.sum() / temp.rows());
        T min = cost_vector_.minCoeff();
        T max = cost_vector_.maxCoeff();
        T magnitude = fabs(min) > fabs(max) ? fabs(min) : fabs(max);
        w_converge_ |= std < magnitude * 0.01;
        u_w_converge_ = u_converge_ && w_converge_;
        CheckMaxTime();

        if (verbose_)
          OutputProgress();
        outer_iter_num_++;
      }
      if (outer_iter_num_ >= 20 && verbose_) {
        std::cout << "Reached 20 iterations" << std::endl;
      }
    }
    PROFILE(RunKMeans(), profiler_["kmeans"]);

    profiler_["fit"].Stop();
  }
//    while (!u_w_converge_ && !max_time_exceeded_ && iter_num < 20) {
//      pre_u_matrix_ = u_matrix_;
//      pre_w_matrix_ = w_matrix_;
//      // When Fit() is called, we already have a w matrix
//      // we project X to subspace W (n * d to d * q)
//      // Generate the kernel matrix based on kernel type from projected X
//      Matrix<T> projected_matrix = x_matrix_ * w_matrix_;
//      GenKernelMatrix(projected_matrix);
//      GenDegreeMatrix();
//      if (method_ == "KDAC") {
//        PROFILE(OptimizeU(), profiler_["u"]);
//        PROFILE(OptimizeW(), profiler_["w"]);
//      } else if (method_ == "ISM") {
//        // When we are doing ISM, I am following Chieh's step to
//        // optimize W first
//        PROFILE(OptimizeWISM(), profiler_["w"]);
//        PROFILE(OptimizeU(), profiler_["u"]);
//      } else {
//        std::cerr << "Use either KDAC or ISM as the method, exiting\n";
//        exit(1);
//      }
//      u_converge_ = util::CheckConverged(u_matrix_, pre_u_matrix_, threshold2_);
//      w_converge_ = util::CheckConverged(w_matrix_, pre_w_matrix_, threshold2_);
//      u_w_converge_ = u_converge_ && w_converge_;
//      if (verbose_)
//        OutputProgress();
//      iter_num ++;
//    }
//    if (iter_num >= 20) {
//      std::cout << "Reached 20 iterations" << std::endl;
//    }
//    PROFILE(RunKMeans(), profiler_["kmeans"]);
//    if (verbose_)
//      std::cout << "Kmeans Done\n";
//    profiler_["fit"].Stop();

  /// This function creates an alternative clustering result
  /// Must be called after \ref Fit(const Matrix<T> &input_matrix)
  /// when the first clustering result is generated
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
    if (method_ == "KDAC") {
      profiler_["fit"].Start();
      profiler_["exit_timer"].Start();
      InitX(input_matrix);
      while (!u_w_converge_ && !max_time_exceeded_) {
        pre_u_matrix_ = u_matrix_;
        pre_w_matrix_ = w_matrix_;
        // When Fit(X, y) is called, we don't already have a w matrix
        // the kernel matrix is then just the Kernel of X itself
        // If vectorization is enabled,
        // Q matrix is generated along with the Kernel Matrix
        GenKernelMatrix(x_matrix_);
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
    } else if (method_ == "ISM") {
      Fit(input_matrix);
      Fit();
    } else {
      std::cerr << "Use either KDAC or ISM as the method, exiting\n";
      exit(1);
    }
  }

  /// Running Predict() after Fit() returns
  /// the current clustering result as a Vector of T
  /// \return
  /// A NICE vector of T that specifies the clustering result
  Vector <T> Predict(void) {
    if (clustering_result_.rows() == 0) {
      std::cerr << "Fit() must be run before Predict(), exiting" << std::endl;
      exit(1);
    } else {
      return clustering_result_;
    }
  }

 protected:
  int c_;  // cluster number c
  int q_;  // reduced dimension q
  int n_;  // number of samples in input data X
  int d_;  // input data X dimension d
  float lambda_;  // Learning rate lambda
  float alpha_;  // Alpha in W optimization
  KernelType kernel_type_;  // The kernel type of the kernel matrix
  float constant_;  // In Gaussian kernel, this is sigma;
  // In Polynomial kernel, this is the polynomial order
  // In Linear kernel, this is c as well
  bool u_converge_;  // If matrix U reaches convergence, false by default
  bool w_converge_;  // If matrix W reaches convergence, false by default
  bool u_w_converge_;  // If matrix U and W both converge, false by default
  T threshold1_;  // threshold for column convergence
  T threshold2_;  // threshold for matrix convergence
  Matrix <T> x_matrix_;  // Input matrix X (n by d)
  Matrix <T> w_matrix_;  // Transformation matrix W (d by q, q < d).
  // Initialized to (d by d) of I
  Matrix <T> pre_w_matrix_;  // W matrix from last iteration,
  // to check convergence
  Matrix <T> y_matrix_;  // Labeling matrix Y (n by (c0 + c1 + c2 + ..))
  Matrix <T> y_matrix_temp_;  // The matrix that holds the current Y_i
  Matrix <T> y_matrix_tilde_;  // The kernel matrix for Y
  Matrix <T> d_matrix_;  // Diagonal degree matrix D (n by n)
  Matrix <T> d_matrix_to_the_minus_half_;  // D^(-1/2) matrix
  Vector <T> d_ii_;  // The diagonal vector of the matrix D
  Vector <T> d_i_;  // The diagonal vector of the matrix D^(-1/2)
  Matrix <T> didj_matrix_;  // The matrix whose element (i, j) equals to
  // di * dj - the ith and jth element from vector d_i_
  Matrix <T> k_matrix_;  // Kernel matrix K (n by n)
  Matrix <T> k_matrix_y_;  // Kernel matrix for Y (n by n)
  Matrix <T> u_matrix_;  // Embedding matrix U (n by c)
  Matrix <T> pre_u_matrix_;  // The U from last iteration, to check convergence
  Matrix <T> u_matrix_normalized_;  // Row-wise normalized U
  Matrix <T> l_matrix_;  // D^(-1/2) * K * D^(-1/2)
  Matrix <T> h_matrix_;  // Centering matrix (n by n)
  Matrix <T> psi_matrix_;  // Psi used in vectorized ISM
  Matrix <T> gamma_matrix_;  // The nxn gamma matrix used in gamma_ij
  Matrix <T> g_of_w_;  // g(w) for updating gradient
  // in formula 5
  Vector <T> clustering_result_;  // Current clustering result
  T phi_of_alpha_, phi_of_zero_, phi_of_zero_prime_;
  // A map container that contains timers for different functions
  KDACProfiler profiler_;
  // Set to true for debug use
  bool verbose_;
  bool debug_;
  bool max_time_exceeded_;
  // Maximum time before exiting, 72000 seconds by default
  int max_time_;
  // Either KDAC or ISM
  std::string method_;

  // Used for switching between gtest mode and python mode
  std::string mode_;

  // Vetorize ISM or not
  bool vectorization_;
  // Q matrix when we do ISM vectorization
  Matrix <T> q_matrix_;
  // Q matrix transpose
  Matrix <T> qt_matrix_;
  // If this is the first time to generate a matrix U
  // If this is true, it means the kernel matrix is generated directly from
  // the input matrix X
  bool first_time_gen_u_;
  // A vector storing every cost in each iteration in WISM
  Vector<T> cost_vector_;
  // Eigen values to determin if W_{k} has converged in ISM
  Vector<T> eigen_vals_;

  // Outer and inner iteration counter for debugging
  int outer_iter_num_;
  int inner_iter_num_;

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
    return ortho_vector.array() / ortho_vector.norm();
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
    util::CheckFinite(u_matrix_, "u_matrix_");
    util::CheckFinite(didj_matrix_, "didj_matrix_");
    util::CheckFinite(y_matrix_tilde_, "y_matrix_tilde_");
    if (debug_) {
      std::string out_path =
          "/home/xiangyu/Dropbox/git_project/NICE/python/debug/output/";
      util::ToFile(gamma_matrix_,
                   out_path + "gamma_kdac_" + mode_ + "_"
                       + std::to_string(outer_iter_num_) + "_"
                       + std::to_string(inner_iter_num_) + ".csv");
    }
  }

  void GenGofW(void) {
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

  /// Generates a degree matrix D from an input kernel matrix
  /// It also generates D^(-1/2) and two diagonal vectors
  void GenDegreeMatrix(void) {
    // Generate the diagonal vector d_i and degree matrix D
    d_ii_ = k_matrix_.rowwise().sum();
    d_matrix_ = d_ii_.asDiagonal();
    // Generate matrix D^(-1/2)
    d_i_ = d_ii_.array().sqrt().unaryExpr(std::ptr_fun(util::reciprocal < T > ));
    d_matrix_to_the_minus_half_ = d_i_.asDiagonal();
  }

  /// Generate the Kernel Matrix based on the current W
  /// Currently only support the gaussian kernel
  /// This is a slight violation of google coding style because
  /// input_matrix is not modified in any way but we cannot use const
  /// because the row() function can only be applied on a non-const variable
  void GenKernelMatrix(Matrix <T> &input) {
    if (kernel_type_ == kGaussianKernel) {
      float sigma_sq = constant_ * constant_;
      for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
          Vector <T> delta_ij = input.row(i) - input.row(j);
          T i_j_dist = delta_ij.squaredNorm();
          k_matrix_(i, j) = exp(-i_j_dist / (2 * sigma_sq));
        }
      }
    }
  }

  void GenKernelAndQMatrix(Matrix <T> &input) {
    if (kernel_type_ == kGaussianKernel) {
      float sigma_sq = constant_ * constant_;
      int q_i = 0;
      for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
          Vector <T> delta_ij = input.row(i) - input.row(j);
          T i_j_dist = delta_ij.squaredNorm();
          k_matrix_(i, j) = exp(-i_j_dist / (2 * sigma_sq));
          q_matrix_.row(q_i) = delta_ij;
          q_i++;
        }
      }
      qt_matrix_ = q_matrix_.transpose();
    }
  }

  // Check if q is not bigger than c
  void ValidateParams() {
    bool validated = true;
    if (n_ == 0 || d_ == 0 || q_ == 0 || c_ == 0) {
      std::cerr << "One or more of n,d,q,c values are not correctly set"
                << std::endl;
      validated = false;
    }
    if (q_ > d_) {
      std::cerr << "Reduced dimension q cannot > dimension d" << std::endl;
      validated = false;
    }
    if (method_ != "KDAC" && method_ != "ISM") {
      std::cerr << "Use either KDAC or ISM as the method, exiting\n";
      validated = false;
    }

    if (!validated) {
      OutputConfigs();
      exit(1);
    }

  }

  /// This function runs KMeans on the normalized U
  void RunKMeans() {
    KMeans <T> kms;
    kms.SetNInit(20);
    T eps = std::numeric_limits<T>::min();
    // Add a very small number to the l2 norm of each row in case it is 0
    u_matrix_normalized_ = u_matrix_.array().colwise() /
        (u_matrix_.rowwise().norm().array() + eps);

    if (debug_) {
      std::string out_path =
          "/home/xiangyu/Dropbox/git_project/NICE/python/debug/output/";
      util::ToFile(u_matrix_, out_path + "u_" + mode_ + ".csv");
    }

    kms.Fit(u_matrix_normalized_, c_);
    clustering_result_ = kms.GetLabels();

    if (y_matrix_.cols() == 0) {
      // When this is calculating Y0
      y_matrix_ = Matrix<T>::Zero(n_, c_);
      for (int i = 0; i < n_; i++)
        y_matrix_(i, clustering_result_(i)) = 1;
    } else {
      // When this is to calculate Y_i and append it to Y_[0~i-1]
      y_matrix_temp_ = Matrix<T>::Zero(n_, c_);
      for (int i = 0; i < n_; i++)
        y_matrix_temp_(i, clustering_result_(i)) = 1;
      Matrix <T> y_matrix_new(n_, y_matrix_.cols() + c_);
      y_matrix_new << y_matrix_, y_matrix_temp_;
      y_matrix_ = y_matrix_new;
      // Reset the y_matrix_temp holder to zero
    }

    if (verbose_)
      std::cout << "Kmeans Done\n";
  }

  Matrix <T> GenPhiOfW(T *objective) {
    if (vectorization_) {
      // Vectorization solution, where we convert the conventional for loop
      // solution to matrix multiplications
      float sigma_sq = pow(constant_, 2);
      Matrix <T> ddt = d_i_ * d_i_.transpose();
      Matrix <T> tau = ddt.cwiseProduct(psi_matrix_).cwiseProduct(k_matrix_);
      if (debug_) {
        std::string out_path =
            "/home/xiangyu/Dropbox/git_project/NICE/python/debug/output/";
        util::ToFile(tau, out_path + "tau_" + mode_ + "_"
                         + std::to_string(outer_iter_num_) + "_"
                         + std::to_string(inner_iter_num_) + ".csv");
      }

      *objective = -tau.sum();
//      Vector<T> test_vector = Vector<T>::Constant(n_*n_, 1);
      Eigen::Map <Vector<T>> tau_map(tau.data(), tau.size());
//      Vector<T> tau_vector = tau_map;
//      qt_matrix_.rowwise() *= test_vector.transpose();
      Matrix <T> phi_w0 = (q_matrix_.array().colwise() * tau_map.array()).matrix().transpose() * q_matrix_;
      Matrix <T> phi_w = phi_w0.array() / sigma_sq;
      return phi_w;
    } else {
      Matrix <T> ddt = d_i_ * d_i_.transpose();
      gamma_matrix_ = ddt.cwiseProduct(psi_matrix_);
      // For loop solution
      float sigma_sq = pow(constant_, 2);
      Matrix <T> phi_w = Matrix<T>::Zero(d_, d_);
      for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
          Vector <T> delta_x_ij =
              this->x_matrix_.row(i) - this->x_matrix_.row(j);
          Matrix <T> a_ij = delta_x_ij * delta_x_ij.transpose();
          Matrix <T> waw = w_matrix_.transpose() * a_ij * w_matrix_;
//          phi_w = phi_w + a_ij * ((gamma_matrix_(i, j) / sigma_sq) *
//              exp(-waw.trace() / (2.0 * sigma_sq)));
          T value = gamma_matrix_(i, j) * exp(-waw.trace() / (2.0 * sigma_sq));
          phi_w = phi_w + a_ij * value;
          *objective -= value;
        }
      }
      phi_w /= sigma_sq;
      return phi_w;
    }
  }

  void UpdateW(const Matrix <T> &phi_w) {
    Eigen::EigenSolver <Matrix<T>> solver(phi_w);
    Vector <T> eigen_values = solver.eigenvalues().real();
    Vector <T> eigen_values_img = solver.eigenvalues().imag();
    Vector <T> eigen_vectors_img = solver.eigenvectors().imag();
    if (eigen_values_img.sum() != 0) {
//      util::ToFile(phi_w, "/home/xiangyu/Dropbox/git_project/NICE/python/debug/phi.csv");
//      util::ToFile(eigen_vectors_img, "/home/xiangyu/Dropbox/git_project/NICE/python/debug/img.csv");
//      std::cerr << "Writing phi to disk" << std::endl;
//      std::cerr << "Imaginary Parts Exist! But not exiting..." << std::endl;
//      exit(1);
    }

    // Key-value sort for eigen values
    std::vector <T>
        v(eigen_values.data(), eigen_values.data() + eigen_values.size());
    std::vector <size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&v](size_t t1, size_t t2) { return v[t1] < v[t2]; });

//    for (int i = 0; i < eigen_values.size(); i++)
//      std::cout << v[idx[i]] << "\t";
//    std::cout << std::endl;

    for (int i = 0; i < q_; i++) {
      w_matrix_.col(i) = solver.eigenvectors().col(idx[i]).real();
      eigen_vals_(i) = eigen_values(idx[i]);
    }
  }

  void OptimizeU(void) {
    l_matrix_ = h_matrix_ * d_matrix_to_the_minus_half_ *
        k_matrix_ * d_matrix_to_the_minus_half_ * h_matrix_;

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

    CheckFiniteOptimizeU();
    if (verbose_)
      std::cout << "U Optimized\n";
  }

  virtual void OptimizeWISM(void) {
    profiler_["update_psi"].Start();
    Matrix <T> pre_w_matrix;
    // eigen_vals is the eigen values of matrix Phi(W), used to determine if
    // updating W_{k} has converged
    Vector<T> pre_eigen_vals;
    psi_matrix_ = h_matrix_ * (u_matrix_ * u_matrix_.transpose() -
        k_matrix_y_ * lambda_) * h_matrix_;
    profiler_["update_psi"].Record();
    bool converge = false;
    inner_iter_num_ = 0;
    int max_iter = 10;
    cost_vector_ = Vector<T>::Zero(max_iter);
    while (!converge) {
      pre_w_matrix = w_matrix_;

      pre_eigen_vals = eigen_vals_;

      T objective = 0.0;
      profiler_["update_phi"].Start();
      Matrix <T> phi_w = GenPhiOfW(&objective);
      profiler_["update_phi"].Record();
      cost_vector_(inner_iter_num_) = objective;
      profiler_["update_w"].Start();
      UpdateW(phi_w);
      profiler_["update_w"].Record();

      if (verbose_) {
        std::cout << "Inner Loop " << inner_iter_num_ << ", ";
        std::cout << "Cost: " << objective << ", ";
      }
      if (debug_) {
        std::string out_path =
            "/home/xiangyu/Dropbox/git_project/NICE/python/debug/output/";
        util::ToFile(phi_w, out_path + "phi_w_" + mode_ + "_" + std::to_string(outer_iter_num_) + "_" + std::to_string(inner_iter_num_) + ".csv");
        util::ToFile(w_matrix_, out_path + "w_" + mode_ + "_" + std::to_string(outer_iter_num_) + "_" + std::to_string(inner_iter_num_) + ".csv");
      }

//      converge = util::CheckConverged(w_matrix_, pre_w_matrix, threshold1_);
      converge = util::CheckConverged(eigen_vals_, pre_eigen_vals, threshold1_);


      if (verbose_) {
        T change1 = static_cast<T>((w_matrix_ - pre_w_matrix).norm()) /
            static_cast<T>(pre_w_matrix.norm());
        T change2 = static_cast<T>((eigen_vals_ - pre_eigen_vals).norm()) /
            static_cast<T>(pre_eigen_vals.norm());
        std::cout << "change in W | eigen vals: " << change1 << " | " << change2 << std::endl;
      }

      if (!converge) {
        profiler_["update_k"].Start();
        Matrix <T> projected_matrix = x_matrix_ * w_matrix_;
        GenKernelMatrix(projected_matrix);
        profiler_["update_k"].Record();
        profiler_["update_d"].Start();
        GenDegreeMatrix();
        profiler_["update_d"].Record();
      }


      inner_iter_num_ += 1;
      if (inner_iter_num_ >= max_iter) {
        break;
      }
    }
    profiler_["update_psi"].SumRecords();
    profiler_["update_phi"].SumRecords();
    profiler_["update_w"].SumRecords();
    profiler_["update_k"].SumRecords();
    profiler_["update_d"].SumRecords();

    if (verbose_) {
      if (converge) {
        std::cout << "WISM Converged" << std::endl;
      } else {
        std::cout << "Not converging after " << max_iter
                  << " iterations, but we jump out of the loop anyway\n";
      }
    }
  }

  virtual void OptimizeW(void) {
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

        w_l = sqrt(1.0 - pow(alpha_, 2)) * w_l + alpha_ * grad_f_vertical;
        w_matrix_.col(l) = w_l;
        w_l_converged =
            util::CheckConverged(objective, pre_objective, threshold2_);
      }
      UpdateGOfW(w_l);
      // TODO: Need to learn about if using Vector<T> &w_l = w_matrix_.col(l)
      if (verbose_)
        std::cout << "Column " << l + 1 << " cost: " << objective << " | ";
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

  void CheckFiniteOptimizeU(void) {
    util::CheckFinite(k_matrix_, "Kernel");
    util::CheckFinite(d_matrix_to_the_minus_half_, "d_matrix_to_minus_half");
    util::CheckFinite(l_matrix_, "L");
    util::CheckFinite(u_matrix_, "U");
  }

  void CheckFiniteOptimizeW(void) {
    util::CheckFinite(didj_matrix_, "didj");
    util::CheckFinite(gamma_matrix_, "Gamma");
    util::CheckFinite(w_matrix_, "W");
  }

  void InitW() {
    // When the user does not initialize W using SetW()
    // W matrix is initilized to be a cut-off identity matrix in KDAC mode
    // and a d x q zero matrix in ISM mode
    // eigen_vals to bookkeep convergence is also initialized.
    if (w_matrix_.cols() == 0) {
      if (method_ == "KDAC") {
        w_matrix_ = Matrix<T>::Identity(d_, q_);
      } else if (method_ == "ISM") {
        w_matrix_ = Matrix<T>::Zero(d_, q_);
        eigen_vals_ = Vector<T>::Zero(q_);
      }
    }
  }

  // Initialization when Fit() is called, only need to update w and y_tilde
  virtual void InitYW() {
    InitY(y_matrix_);
    InitW();

    // Need to reset those states because the previous Fit(X) or Fit(X, y) might
    // change some of them to true
    max_time_exceeded_ = false;
    u_converge_ = false;
    w_converge_ = false;
    u_w_converge_ = false;

  }

  // Initialize all data structures related to the input matrix X
  virtual void InitX(const Matrix <T> &input_matrix) {
    x_matrix_ = input_matrix;
    n_ = input_matrix.rows();
    d_ = input_matrix.cols();
    ValidateParams();
    h_matrix_ = Matrix<T>::Identity(n_, n_)
        - Matrix<T>::Constant(n_, n_, 1) / static_cast<T>(n_);

    // kernel matrix
    k_matrix_ = Matrix<T>::Zero(n_, n_);
    u_matrix_ = Matrix<T>::Zero(n_, c_);

    // Generate Q matrix if using ISM method and vectorization is true
    if (method_ == "ISM" && vectorization_) {
      q_matrix_ = Matrix<T>::Zero(n_ * n_, d_);
    }
  }

  // Initialization for generating alternative views with a given Y
  virtual void InitXYW(const Matrix <T> &input_matrix,
                    const Matrix <T> &y_matrix) {
    InitX(input_matrix);
    InitY(y_matrix);
    InitW();
  }

  // Initialization for all Y related data structures
  virtual void InitY(const Matrix<T> &y_matrix) {
    // The following is only called when we have y matrix
    y_matrix_ = y_matrix;
    // Generate the kernel for the label matrix Y: K_y
    k_matrix_y_ = y_matrix_ * y_matrix_.transpose();
    // Generate Y tilde matrix in equation 5 from kernel matrix of Y
    y_matrix_tilde_ = h_matrix_ * k_matrix_y_ * h_matrix_;
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
