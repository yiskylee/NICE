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

// Alternative CLustering (ACL), base class for KDAC and ISM
#ifndef CPP_INCLUDE_ACL_H
#define CPP_INCLUDE_ACL_H

#include "include/util.h"
#include "include/kernel_types.h"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/acl_profiler.h"
#include "include/kmeans.h"


namespace Nice {
template<class T>
class ACL {
 public:
  /// Default constructor for ACL
  /// It contains common variables used in both KDAC and ISM
  ACL() :
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
      threshold2_(0.01),
      x_matrix_(),
      w_matrix_(),
      pre_w_matrix_(),
      u_matrix_(),
      pre_u_matrix_(),
      verbose_(false),
      debug_(false),
      max_time_exceeded_(false),
      max_time_(100),
      method_(""),
      mode_(""),
      clustering_result_(),
      u_matrix_normalized_(),
      y_matrix_(),
      y_matrix_temp_(),
      d_i_(),
      l_matrix_(),
      h_matrix_(),
      k_matrix_y_(),
      k_matrix_(),
      d_matrix_(),
      d_matrix_to_the_minus_half_(),
      d_ii_(),
      didj_matrix_(),
      gamma_matrix_(),
      profiler_(),
      vectorization_(true),
      u_eigenvals_(),
      pre_u_eigenvals_()
  {
    profiler_["fit"].SetName("fit");
    profiler_["exit_timer"].SetName("exit_timer");
    profiler_["init"].SetName("init");
    profiler_["u"].SetName("u");
    profiler_["w"].SetName("w");
    profiler_["kmeans"].SetName("kmeans");
    profiler_["gen_phi"].SetName("gen_phi");
    profiler_["gen_grad"].SetName("gen_grad");
    profiler_["update_g_of_w"].SetName("update_g_of_w");
    profiler_["update_psi"].SetName("update_psi");
    profiler_["update_phi"].SetName("update_phi");
    profiler_["update_w"].SetName("update_w");
    profiler_["update_k"].SetName("update_k");
    profiler_["update_d"].SetName("update_d");
  }

  ~ACL() {}
  ACL(const ACL &rhs) {}

  /// This function creates the first clustering result
  /// \param input_matrix
  /// The input matrix of n samples and d features where each row
  /// represents a sample
  /// \return
  /// It only generates the clustering result but does not returns it
  /// Users can use Predict() to get the clustering result returned
  virtual void Fit(const Matrix <T> &input_matrix) = 0;

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
  virtual void Fit(const Matrix <T> &input_matrix, const Matrix <T> &y_matrix)
      = 0;


  /// This function creates an alternative clustering result
  /// Must be called after \ref Fit(const Matrix<T> &input_matrix)
  /// when the first clustering result is generated
  virtual void Fit() = 0;

  /// Running Predict() after Fit() returns
  /// the current clustering result as a Vector of T
  /// \return
  /// A NICE vector of T that specifies the clustering result
  Vector <T> Predict() {
    if (clustering_result_.rows() == 0) {
      std::cerr << "Fit() must be run before Predict(), exiting" << std::endl;
      exit(1);
    } else {
      return clustering_result_;
    }
  }

  // Initialize all data structures related to the input matrix X
  virtual void InitX(const Matrix <T> &input_matrix) {
    if (input_matrix.rows() == 0 || input_matrix.cols() == 0) {
      std::cerr << "X matrix is not initialized\n";
      exit(1);
    }

    n_ = input_matrix.rows();
    d_ = input_matrix.cols();
    ValidateParams();
    h_matrix_ = Matrix<T>::Identity(n_, n_)
        - Matrix<T>::Constant(n_, n_, 1) / static_cast<T>(n_);

    Matrix<T> h_matrix_2 = Matrix<T>::Identity(d_, d_)
        - Matrix<T>::Constant(d_, d_, 1) / static_cast<T>(d_);

    if (input_matrix.rowwise().mean().norm() != 0 ||
        input_matrix.colwise().mean().norm() != 0) {
      std::cout << "Input matrix is not scaled\n";
      x_matrix_ = h_matrix_ * input_matrix * h_matrix_2;
    } else {
      x_matrix_ = input_matrix;
    }


    if (x_matrix_.rowwise().mean().norm() != 0 ||
        x_matrix_.colwise().mean().norm() != 0) {
      std::cout << "Still not scaled\n";
      std::cout << x_matrix_.rowwise().mean().norm() << std::endl;
      std::cout << x_matrix_.colwise().mean().norm() << std::endl;
    }

    k_matrix_ = Matrix<T>::Zero(n_, n_);
    u_matrix_ = Matrix<T>::Zero(n_, c_);

    u_eigenvals_ = Vector<T>::Zero(c_);

  }

  virtual void InitY(const Matrix<T> &y_matrix) {
    if (y_matrix.rows() == 0 || y_matrix.cols() == 0) {
      std::cerr << "Y matrix is not initialized\n";
      exit(1);
    }
    // The following is only called when we have y matrix
    y_matrix_ = y_matrix;
    // Generate the kernel for the label matrix Y: K_y
    k_matrix_y_ = y_matrix_ * y_matrix_.transpose();
  }
  virtual void InitW() = 0;

  // Initialization for generating alternative views with a given Y
  virtual void InitXYW(const Matrix <T> &input_matrix,
                       const Matrix <T> &y_matrix) {
    InitX(input_matrix);
    InitY(y_matrix);
    InitW();
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


  virtual void OutputConfigs() {
    std::cout << "c: " << c_ << std::endl;
    std::cout << "q: " << q_ << std::endl;
    std::cout << "n: " << n_ << std::endl;
    std::cout << "d: " << d_ << std::endl;
    std::cout << "lambda: " << lambda_ << std::endl;
    std::cout << "alpha: " << alpha_ << std::endl;
    std::cout << "sigma: " << constant_ << std::endl;
    std::cout << "threshold1: " << threshold1_ << std::endl;
    std::cout << "threshold2: " << threshold2_ << std::endl;
    std::cout << "method: " << method_ << std::endl;
    std::cout << "mode: " << mode_ << std::endl;
    std::cout << "verbose: " << verbose_ << std::endl;
    std::cout << "debug: " << debug_ << std::endl;
  }

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

  void SetVectorization(bool vectorization) { vectorization_ = vectorization; }

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

  void SetMode(std::string mode) { mode_ = mode; }

  int GetD() { return d_; }

  int GetN() { return n_; }

  int GetQ() { return q_; }

  int GetC() { return c_; }

  Matrix <T> GetU() { return u_matrix_; }

  Matrix <T> GetW() { return w_matrix_; }

  Matrix <T> GetUNormalized() { return u_matrix_normalized_; }

  Matrix <T> GetL() { return l_matrix_; }

  Matrix <T> GetDMatrix() { return d_matrix_; }

  Matrix <T> GetDToTheMinusHalf() { return d_matrix_to_the_minus_half_; }

  Matrix <T> GetK() { return k_matrix_; }

  Matrix <T> GetY(void) { return y_matrix_; }

  Matrix <T> GetGamma(void) { return gamma_matrix_; }

  ACLProfiler GetProfiler(void) { return profiler_; }

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

  void CheckFiniteOptimizeU() {
    util::CheckFinite(k_matrix_, "Kernel");
    util::CheckFinite(d_matrix_to_the_minus_half_, "d_matrix_to_minus_half");
    util::CheckFinite(l_matrix_, "L");
    util::CheckFinite(u_matrix_, "U");
  }


  /// Generates a degree matrix D from an input kernel matrix
  /// It also generates D^(-1/2) and two diagonal vectors
  void GenDegreeMatrix() {
    // Generate the diagonal vector d_i and degree matrix D
    d_ii_ = k_matrix_.rowwise().sum();
    d_matrix_ = d_ii_.asDiagonal();
    // Generate matrix D^(-1/2)
    d_i_ = d_ii_.array().sqrt();

    // didj matrix contains the element (i, j) that equal to d_i * d_j
    didj_matrix_ = d_i_ * d_i_.transpose();

//    d_i_ = d_ii_.array().sqrt().unaryExpr(std::ptr_fun(util::reciprocal < T > ));
    d_matrix_to_the_minus_half_ =
        d_i_.unaryExpr(std::ptr_fun(util::reciprocal < T > )).asDiagonal();

    // XILI Debug
//    util::Print(d_ii_, "d_ii_");
//    util::Print(d_i_, "d_i_");
//    util::Print(d_matrix_to_the_minus_half_.block(0,0,10,10), "D-1/2");
    // XILI Debug

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
  Matrix <T> pre_w_matrix_;  // W matrix from last iteration
  Matrix <T> u_matrix_;  // Embedding matrix U (n by c)
  Matrix <T> pre_u_matrix_;  // The U from last iteration, to check convergence
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
  Vector <T> clustering_result_;  // Current clustering result
  Matrix <T> u_matrix_normalized_;  // Row-wise normalized U
  Matrix <T> y_matrix_;  // Labeling matrix Y (n by (c0 + c1 + c2 + ..))
  Matrix <T> y_matrix_temp_;  // The matrix that holds the current Y_i
  Vector <T> d_i_;  // The diagonal vector of the matrix D^(-1/2)
  Matrix <T> l_matrix_;  // D^(-1/2) * K * D^(-1/2)
  Matrix <T> h_matrix_;  // Centering matrix (n by n)
  Matrix <T> k_matrix_y_;  // Kernel matrix for Y (n by n)
  Matrix <T> k_matrix_;  // Kernel matrix K (n by n)
  Matrix <T> d_matrix_;  // Diagonal degree matrix D (n by n)
  Matrix <T> d_matrix_to_the_minus_half_;  // D^(-1/2) matrix
  Vector <T> d_ii_;  // The diagonal vector of the matrix D
  Matrix <T> didj_matrix_;  // The matrix whose element (i, j) equals to
  // di * dj - the ith and jth element from vector d_i_
  Matrix <T> gamma_matrix_;  // The nxn gamma matrix used in gamma_ij
  // A map container that contains timers for different functions
  ACLProfiler profiler_;
  // Vetorize ISM or not
  bool vectorization_;
  // Eigenvalues of L matrix to track convergence of U matrix
  Vector<T> u_eigenvals_;
  Vector<T> pre_u_eigenvals_;


  /// This function runs KMeans on the normalized U
  void RunKMeans() {
    KMeans<T> kms;
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

 private:
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

    if (kernel_type_ != kGaussianKernel) {
      std::cerr << "kernel_type_: " << kernel_type_ << " does not exist\n";
      validated = false;
    }

    if (!validated) {
      OutputConfigs();
      exit(1);
    }
  }




};

}  // namespace NICE

#endif  // CPP_INCLUDE_ACL_H
