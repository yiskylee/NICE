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
      c_(2),
      q_(2),
      n_(0),
      d_(0),
      lambda_(1),
      alpha_(1.0),
      kernel_type_(kGaussianKernel),
      constant_(1.0),
      u_converge_(false),
      w_converge_(false),
      u_w_converge_(false),
      threshold1_(0.01),
      threshold2_(0.01),
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
      vectorization_(false) {}

  ~KDAC() {}
  KDAC(const KDAC &rhs) {}

  // Set the number of clusters c
  void SetC(int c) { c_ = c; }

  // Set user-defined W
  void SetW(const Matrix<T> &w_matrix) {
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

  void SetDebug(bool debug) { debug_ = debug; }

  void SetMethod(std::string method) { method_ = method; }

  int GetD(void) { return d_; }

  int GetN(void) { return n_; }

  int GetQ(void) { return q_; }

  int GetC(void) { return c_; }

  Matrix<T> GetU(void) { return u_matrix_; }

  Matrix<T> GetW(void) { return w_matrix_; }

  Matrix<T> GetUNormalized(void) { return u_matrix_normalized_; }

  Matrix<T> GetL(void) { return l_matrix_; }

  Matrix<T> GetDMatrix(void) { return d_matrix_; }

  Matrix<T> GetDToTheMinusHalf(void) { return d_matrix_to_the_minus_half_; }

  Matrix<T> GetK(void) {
    GenKernelMatrix();
    return k_matrix_;
  }

  Matrix<T> GetY(void) { return y_matrix_; }

  Matrix<T> GetYTilde(void) { return y_matrix_tilde_; }

  Matrix<T> GetGamma(void) { return gamma_matrix_; }

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

  /// This function can be used when the user is not satisfied with
  /// the previous clustering results and want to discard the result from
  /// the last run so she can re-run Fit with new parameters
  void DiscardLastRun() {
    if (debug_)
      util::Print(y_matrix_, "y_matrix_before");
    Matrix<T> y_matrix_new = Matrix<T>::Zero(n_, y_matrix_.cols() - c_);
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
  void Fit(const Matrix<T> &input_matrix) {
    profiler_.fit.Start();
    profiler_.exit_timer.Start();
    PROFILE(Init(input_matrix), profiler_.init);
    // When there is no Y, it is the the first round when the second term
    // lambda * HSIC is zero, we do not need to optimize W, and we directly
    // go to kmeans where Y_0 is generated. And both u and v are converged.
    PROFILE(OptimizeU(), profiler_.u);
    PROFILE(RunKMeans(), profiler_.kmeans);
    profiler_.fit.Stop();
  }


  // Fit() with an empty param list can only be run when the X and Y already
  // exist from the previous round of computation
  void Fit(void) {
    // Only changes w_matrix and y_tilde matrix
    profiler_.fit.Start();
    profiler_.exit_timer.Start();
    PROFILE(Init(), profiler_.init);
    while (!u_w_converge_ && !max_time_exceeded_) {
      pre_u_matrix_ = u_matrix_;
      pre_w_matrix_ = w_matrix_;
      PROFILE(OptimizeU(), profiler_.u);
      if (method_ == "KDAC") {
        PROFILE(OptimizeW(), profiler_.w);
      } else if (method_ == "ISM") {
        PROFILE(OptimizeWISM(), profiler_.w);
      } else {
        std::cerr << "Use either KDAC or ISM as the method, exiting" << std::endl;
        exit(1);
      }
      u_converge_ = util::CheckConverged(u_matrix_, pre_u_matrix_, threshold2_);
      w_converge_ = util::CheckConverged(w_matrix_, pre_w_matrix_, threshold2_);
      u_w_converge_ = u_converge_ && w_converge_;
      if (verbose_)
        OutputProgress();
    }
    PROFILE(RunKMeans(), profiler_.kmeans);
    if (verbose_)
      std::cout << "Kmeans Done" << std::endl;
    profiler_.fit.Stop();
  }

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
  void Fit(const Matrix<T> &input_matrix, const Matrix<T> &y_matrix) {
    // This is called when we have exsiting labels Y
    // now we are generating an alternative view with a
    // given Y_previous by doing Optimize both W and U until they converge
    // Following the pseudo code in Algorithm 1 in the paper
    profiler_.fit.Start();
    profiler_.exit_timer.Start();
    PROFILE(Init(input_matrix, y_matrix), profiler_.init);
    while (!u_w_converge_ && !max_time_exceeded_) {
      pre_u_matrix_ = u_matrix_;
      pre_w_matrix_ = w_matrix_;
      PROFILE(OptimizeU(), profiler_.u);
      if (method_ == "KDAC") {
        PROFILE(OptimizeW(), profiler_.w);
      } else if (method_ == "ISM") {
        PROFILE(OptimizeWISM(), profiler_.w);
      } else {
        std::cerr << "Use either KDAC or ISM as the method, exiting" << std::endl;
        exit(1);
      }
      u_converge_ = util::CheckConverged(u_matrix_, pre_u_matrix_, threshold2_);
      w_converge_ = util::CheckConverged(w_matrix_, pre_w_matrix_, threshold2_);
      u_w_converge_ = u_converge_ && w_converge_;
      if (verbose_)
        OutputProgress();
    }
    PROFILE(RunKMeans(), profiler_.kmeans);
    if (verbose_)
      std::cout << "Kmeans Done" << std::endl;
    profiler_.fit.Stop();
  }

  /// Running Predict() after Fit() returns
  /// the current clustering result as a Vector of T
  /// \return
  /// A NICE vector of T that specifies the clustering result
  Vector<T> Predict(void) {
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
  Matrix<T> x_matrix_;  // Input matrix X (n by d)
  Matrix<T> w_matrix_;  // Transformation matrix W (d by q, q < d).
  // Initialized to (d by d) of I
  Matrix<T> pre_w_matrix_;  // W matrix from last iteration,
  // to check convergence
  Matrix<T> y_matrix_;  // Labeling matrix Y (n by (c0 + c1 + c2 + ..))
  Matrix<T> y_matrix_temp_;  // The matrix that holds the current Y_i
  Matrix<T> y_matrix_tilde_;  // The kernel matrix for Y
  Matrix<T> d_matrix_;  // Diagonal degree matrix D (n by n)
  Matrix<T> d_matrix_to_the_minus_half_;  // D^(-1/2) matrix
  Vector<T> d_ii_;  // The diagonal vector of the matrix D
  Vector<T> d_i_;  // The diagonal vector of the matrix D^(-1/2)
  Matrix<T> didj_matrix_;  // The matrix whose element (i, j) equals to
  // di * dj - the ith and jth element from vector d_i_
  Matrix<T> k_matrix_;  // Kernel matrix K (n by n)
  Matrix<T> k_matrix_y_;  // Kernel matrix for Y (n by n)
  Matrix<T> u_matrix_;  // Embedding matrix U (n by c)
  Matrix<T> pre_u_matrix_;  // The U from last iteration, to check convergence
  Matrix<T> u_matrix_normalized_;  // Row-wise normalized U
  Matrix<T> l_matrix_;  // D^(-1/2) * K * D^(-1/2)
  Matrix<T> h_matrix_;  // Centering matrix (n by n)
  Matrix<T> gamma_matrix_;  // The nxn gamma matrix used in gamma_ij
  Matrix<T> g_of_w_;  // g(w) for updating gradient
  // in formula 5
  Vector<T> clustering_result_;  // Current clustering result
  T phi_of_alpha_, phi_of_zero_, phi_of_zero_prime_;
  // A struct contains timers for different functions
  KDACProfiler profiler_;
  // Set to true for debug use
  bool verbose_;
  bool debug_;
  bool max_time_exceeded_;
  // Maximum time before exiting, 72000 seconds by default
  int max_time_;
  // Either KDAC or ISM
  std::string method_;
  // Vetorize ISM or not
  bool vectorization_;


  Vector<T> GenOrthogonal(const Matrix<T> &space,
                          const Vector<T> &vector) {
    Vector<T> projection = Vector<T>::Zero(space.rows());
    for (int j = 0; j < space.cols(); j++) {
      // projection = (v * u / u^2) * u
      projection += (vector.dot(space.col(j)) /
          space.col(j).squaredNorm()) * space.col(j);
    }
    return vector - projection;
  }

  Vector<T> GenOrthonormal(const Matrix<T> &space,
                           const Vector<T> &vector) {
    util::CheckFinite(space, "space");
    Vector<T> ortho_vector = GenOrthogonal(space, vector);
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
    d_i_ = d_ii_.array().sqrt().unaryExpr(std::ptr_fun(util::reciprocal<T>));
    d_matrix_to_the_minus_half_ = d_i_.asDiagonal();
  }

  /// Generate the Kernel Matrix based on the current W
  void GenKernelMatrix() {
    // Project X to subspace W (n * d to d * q)
    // Generate the kernel matrix based on kernel type from projected X
    Matrix<T> projected_x_matrix = x_matrix_ * w_matrix_;
    if (kernel_type_ == kGaussianKernel) {
      float sigma_sq = constant_ * constant_;
      for (int i = 0; i < n_; i++)
        for (int j = 0; j < n_; j++) {
          Vector<T> delta_ij =
              projected_x_matrix.row(i) - projected_x_matrix.row(j);
          T i_j_dist = delta_ij.norm();
          k_matrix_(i, j) = exp(-i_j_dist / (2 * sigma_sq));
        }
    }
  }

  // Check if q is not bigger than c
  void CheckQD() {
    if (q_ >= d_) {
      std::cerr << "Reduced dimension q cannot >= dimension d" << std::endl;
      exit(1);
    }
  }

  /// This function runs KMeans on the normalized U
  void RunKMeans() {
    KMeans<T> kms;
    T eps = std::numeric_limits<T>::min();
    // Add a very small number to the l2 norm of each row in case it is 0
    u_matrix_normalized_ = u_matrix_.array().colwise() /
        (u_matrix_.rowwise().norm().array() + eps);
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
      Matrix<T> y_matrix_new(n_, y_matrix_.cols() + c_);
      y_matrix_new << y_matrix_, y_matrix_temp_;
      y_matrix_ = y_matrix_new;
      // Reset the y_matrix_temp holder to zero
    }
  }

  void OptimizeU(void) {
    GenKernelMatrix();
    // Generate degree matrix from the kernel matrix
    // d_i is the diagonal vector of degree matrix D
    // This is a reference to how to directly generate D^(-1/2)
    // Vector<T> d_i = k_matrix_.rowwise().sum().array().sqrt().unaryExpr(
    //     std::ptr_fun(util::reciprocal<T>));
    // d_matrix_ = d_i.asDiagonal();

    // Generate D and D^(-1/2)
    GenDegreeMatrix();
    l_matrix_ = d_matrix_to_the_minus_half_ *
        k_matrix_ * d_matrix_to_the_minus_half_;
    SvdSolver<T> solver;
    solver.Compute(l_matrix_);
    // Generate a u matrix from SVD solver and then use Normalize
    // to normalize its rows
    u_matrix_ = solver.MatrixU().leftCols(c_);
    CheckFiniteOptimizeU();
    if (verbose_)
      std::cout << "U Optimized" << std::endl;
  }

  virtual void OptimizeWISM(void) {
    if (vectorization_) {
      // Vectorization solution, where we convert the conventional for loop
      // solution to matrix multiplications
      Matrix<T> psi = h_matrix_ * (u_matrix_ * u_matrix_.transpose() -
                      k_matrix_y_ * lambda_) * h_matrix_;
    } else {
      // For loop solution
      float sigma_sq = pow(constant_, 2);
      Matrix<T> phi_w = Matrix<T>::Zero(d_, d_);
      for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
          Vector<T> delta_x_ij =
              this->x_matrix_.row(i) - this->x_matrix_.row(j);
          Matrix<T> a_ij = delta_x_ij * delta_x_ij.transpose();
          Matrix<T> waw = w_matrix_.transpose() * a_ij * w_matrix_;
          T value = gamma_matrix_(i, j) * exp(-waw.trace() / (2.0 * sigma_sq));
          phi_w = phi_w + a_ij * value;
        }
      }
      Eigen::EigenSolver<Matrix<T>> solver(phi_w);

      Vector<T> eigen_values = solver.eigenvalues().real();
      Vector<T> eigen_values_img = solver.eigenvalues().imag();
      if (eigen_values_img.sum() != 0)
        std::cout << "Imaginary Parts Exist!" << std::endl;

      // Key-value sort for eigen values
      std::vector<T>
          v(eigen_values.data(), eigen_values.data() + eigen_values.size());
      std::vector<size_t> idx(v.size());
      std::iota(idx.begin(), idx.end(), 0);
      std::sort(idx.begin(), idx.end(),
                [&v](size_t t1, size_t t2) { return v[t1] < v[t2]; });
      for (int i = 0; i < q_; i++)
        w_matrix_.col(i) = solver.eigenvectors().col(idx[i]).real();
    }
  }

  virtual void OptimizeW(void) {
    // We optimize each column in the W matrix
    for (int l = 0; l < w_matrix_.cols(); l++) {
      Vector<T> w_l;
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
        Vector<T> grad_f_vertical;
        T pre_objective = objective;
        // Calculate the w gradient in equation 13, then find the gradient
        // that is vertical to the space spanned by w_0 to w_l
        Vector<T> grad_f = GenWGradient(w_l);
        grad_f_vertical =
            GenOrthonormal(w_matrix_.leftCols(l + 1), grad_f);
        LineSearch(grad_f_vertical, &w_l, &objective);
        w_l = sqrt(1.0 - pow(alpha_, 2)) * w_l + alpha_ * grad_f_vertical;
        w_matrix_.col(l) = w_l;
        w_l_converged =
            util::CheckConverged(objective, pre_objective, threshold1_);
      }
      UpdateGOfW(w_l);
      // TODO: Need to learn about if using Vector<T> &w_l = w_matrix_.col(l)
      if (verbose_)
        std::cout << "Column " << l+1 << " cost: " << objective << " | ";
    }
    if (verbose_)
      std::cout << "W Optimized" << std::endl;

    profiler_.exit_timer.Stop();

    // If the whole program runs for more than 20 hours, it returns
    if (profiler_.exit_timer.vec_.back() / 1e3 > max_time_) {
      std::cout << "Exceeds maximum time limit. " << std::endl;
      max_time_exceeded_ = true;
    }

    profiler_.gen_phi.SumRecords();
    profiler_.gen_grad.SumRecords();
    profiler_.update_g_of_w.SumRecords();
  }

  void LineSearch(const Vector<T> &gradient,
                  Vector<T> *w_l, T *objective) {
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

  // Initialization when Fit() is called, only need to update w and y_tilde
  void Init() {
    w_matrix_ = Matrix<T>::Identity(d_, d_);
    k_matrix_y_ = y_matrix_ * y_matrix_.transpose();
    y_matrix_tilde_ = h_matrix_ * k_matrix_y_ * h_matrix_;
    u_converge_ = false;
    w_converge_ = false;
    u_w_converge_ = false;
    max_time_exceeded_ = false;
  }

  // Used only in Fit(const Matrix<T> &input_matrix)
  virtual void Init(const Matrix<T> &input_matrix) {
    x_matrix_ = input_matrix;
    n_ = input_matrix.rows();
    d_ = input_matrix.cols();
    CheckQD();
    // When the user does not initialize W using SetW()
    // W matrix is initilized to be an identity matrix
    if (w_matrix_.cols() == 0) {
      if (method_ == "KDAC") {
        w_matrix_ = Matrix<T>::Identity(d_, q_);
      } else if (method_ == "ISM") {
        w_matrix_ = Matrix<T>::Zero(d_, q_);
      }
    }

    h_matrix_ = Matrix<T>::Identity(n_, n_)
        - Matrix<T>::Constant(n_, n_, 1) / static_cast<T>(n_);
    // kernel matrix
    k_matrix_ = Matrix<T>::Zero(n_, n_);
    max_time_exceeded_ = false;
  }

  // Initialization for generating alternative views with a given Y
  virtual void Init(const Matrix<T> &input_matrix,
                    const Matrix<T> &y_matrix) {
    x_matrix_ = input_matrix;
    n_ = input_matrix.rows();
    d_ = input_matrix.cols();
    CheckQD();

    // When the user does not initialize W using SetW()
    // W matrix is initilized to be an identity matrix
    if (w_matrix_.cols() == 0) {
      if (method_ == "KDAC") {
        w_matrix_ = Matrix<T>::Identity(d_, q_);
      } else if (method_ == "ISM") {
        w_matrix_ = Matrix<T>::Zero(d_, q_);
      }
    }
    h_matrix_ = Matrix<T>::Identity(n_, n_)
        - Matrix<T>::Constant(n_, n_, 1) / static_cast<T>(n_);
    y_matrix_ = y_matrix;
    // kernel matrix
    k_matrix_ = Matrix<T>::Zero(n_, n_);
    // Generate the kernel for the label matrix Y: K_y
    k_matrix_y_ = y_matrix_ * y_matrix_.transpose();
    // Generate Y tilde matrix in equation 5 from kernel matrix of Y
    y_matrix_tilde_ = h_matrix_ * k_matrix_y_ * h_matrix_;
    u_converge_ = false;
    w_converge_ = false;
    u_w_converge_ = false;
    max_time_exceeded_ = false;
  }

  virtual void UpdateGOfW(const Vector<T> &w_l) = 0;

  virtual void GenPhi(const Vector<T> &w_l,
              const Vector<T> &gradient,
              bool w_l_changed) = 0;

  virtual void GenPhiCoeff(const Vector<T> &w_l, const Vector<T> &gradient) = 0;
  virtual Vector<T> GenWGradient(const Vector<T> &w_l) = 0;



};
}  // namespace NICE

#endif  // CPP_INCLUDE_KDAC_H
