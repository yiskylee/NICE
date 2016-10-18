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

// Kernel Dimension Alternative Clustering (KDAC)
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
#include <X11/Xlib.h>
#include <valarray>
#include <tgmath.h>
#include "include/matrix.h"
#include "include/vector.h"
#include "include/cpu_operations.h"
#include "include/svd_solver.h"
#include "include/kmeans.h"
#include "include/spectral_clustering.h"
#include "Eigen/Core"
#include "include/util.h"
#include "include/kernel_types.h"
#include "include/stop_watch.h"

#define PROFILE(func, timer, time)\
  timer.Start();\
  func;\
  timer.Stop();\
  time += timer.DiffInMs();\


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
      alpha_(0.1),
      kernel_type_(kGaussianKernel),
      constant_(1.0),
      u_converge_(false),
      w_converge_(false),
      u_w_converge_(false),
      threshold_(0.01),
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
      a_matrix_list_(),
      clustering_result_() {}

  ~KDAC() {}
  KDAC(const KDAC &rhs) {}
  KDAC &operator=(const KDAC &rhs) {}

  void Print(const Vector<T> &vector, std::string name) {
//    std::cout.precision(2);
//    std::cout << std::scientific;
    std::cout << name << std::endl;
    for (int i = 0; i < vector.rows(); i++) {
      std::cout << vector(i) << " ";
    }
    std::cout << std::endl;
  }

  void Print(const Matrix<T> &matrix, std::string name) {
//    std::cout.precision(2);
//    std::cout << std::scientific;
    std::cout << name << std::endl;
    std::cout << matrix << " ";
    std::cout << std::endl;
  }

  void Print(const T &scalar, std::string name) {
//    std::cout.precision(2);
//    std::cout << std::scientific;
    std::cout << name << std::endl;
    std::cout << scalar << std::endl;
  }


  // Set the number of clusters c
  void SetC(int c) {
    c_ = c;
  }

  // Set lambda for HSIC
  void SetLambda(float lambda) {
    lambda_ = lambda;
  }

  /// Set the reduced dimension q
  void SetQ(int q) {
    q_ = q;
  }

  int GetD(void) {
    return d_;
  }

  int GetN(void) {
    return n_;
  }

  int GetQ(void) {
    return q_;
    // To test if Nice4Py reflects changes
    // return 88;
  }

  int GetC(void) {
    return c_;
  }

  Matrix<T> GetU(void) {
    return u_matrix_;
  }

  Matrix<T> GetW(void) {
    return w_matrix_;
  }

  Matrix<T> GetUNormalized(void) {
    return u_matrix_normalized_;
  }

  Matrix<T> GetL(void) {
    return l_matrix_;
  }

  Matrix<T> GetDMatrix(void) {
    return d_matrix_;
  }

  Matrix<T> GetDToTheMinusHalf(void) {
    return d_matrix_to_the_minus_half_;
  }

  Matrix<T> GetK(void) {
    return k_matrix_;
  }

  Matrix<T> GetY(void) {
    return y_matrix_;
  }

  std::vector<Matrix<T>> GetAList(void) {
    return a_matrix_list_;
  }

  Matrix<T> GetYTilde(void) {
    return y_matrix_tilde_;
  }

  Matrix<T> GetGamma(void) {
    return gamma_matrix_;
  }

  double GetTimeInit(void) {
    return time_init_;
  }

  double GetTimeU(void) {
    return time_u_;
  }

  double GetTimeW(void) {
    return time_w_;
  }

  double GetTimeKMeans(void) {
    return time_kmeans_;
  }

  double GetTimeFit(void) {
    return time_fit_;
  }

  int GetNumItersFit(void) {
    return num_iters_fit_;
  }

  int GetNumItersWMatrix(void) {
    return num_iters_w_matrix_.sum();
  }



  /// Set the kernel type: kGaussianKernel, kPolynomialKernel, kLinearKernel
  /// And set the constant associated the kernel
  void SetKernel(KernelType kernel_type, float constant) {
    kernel_type_ = kernel_type;
    constant_ = constant;
  }

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
    Vector<T> ortho_vector = GenOrthogonal(space, vector);
    return ortho_vector.array() / ortho_vector.norm();
  }

  /// This function creates the first clustering result
  /// \param input_matrix
  /// The input matrix of n samples and d features where each row
  /// represents a sample
  /// \return
  /// It only generates the clustering result but does not returns it
  /// Users can use Predict() to get the clustering result returned
  void Fit(const Matrix<T> &input_matrix) {
    Init(input_matrix);
    // When there is no Y, it is the the first round when the second term
    // lambda * HSIC is zero, we do not need to optimize W, and we directly
    // go to kmeans where Y_0 is generated. And both u and v are converged.
    OptimizeU();
    RunKMeans();
  }


  // Fit() with an empyt param list can only be run when the X and Y already
  // exist from the previous round of computation
  void Fit(void) {
    Init();
    while (!u_w_converge_) {
      pre_u_matrix_ = u_matrix_;
      pre_w_matrix_ = w_matrix_;
      OptimizeU();
      OptimizeW();
      u_converge_ = CheckConverged(u_matrix_, pre_u_matrix_, threshold_);
      w_converge_ = CheckConverged(w_matrix_, pre_w_matrix_, threshold_);
      u_w_converge_ = u_converge_ && w_converge_;
    }
    RunKMeans();
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
    timer_fit_.Start();
    PROFILE(Init(input_matrix, y_matrix), timer_init_, time_init_);

    while (!u_w_converge_) {
      pre_u_matrix_ = u_matrix_;
      pre_w_matrix_ = w_matrix_;
      PROFILE(OptimizeU(), timer_u_, time_u_);
      PROFILE(OptimizeW(), timer_w_, time_w_);
      u_converge_ = CheckConverged(u_matrix_, pre_u_matrix_, threshold_);
      w_converge_ = CheckConverged(w_matrix_, pre_w_matrix_, threshold_);
      u_w_converge_ = u_converge_ && w_converge_;
      num_iters_fit_ ++;
    }
    PROFILE(RunKMeans(), timer_kmeans_, time_kmeans_);
    timer_fit_.Stop();
    time_fit_ = timer_fit_.DiffInMs();
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

 private:
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
  T threshold_;  // To determine convergence
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
  Matrix<T> gamma_matrix_;  // The gamma matrix used in gamma_ij in formula 5
  std::vector<Matrix<T>> a_matrix_list_;  // An n*n list that contains all of
  // the A_ij matrix
  Vector<T> clustering_result_;  // Current clustering result
  Matrix<T> waw_matrix_;
  Matrix<T> waf_matrix_;
  Matrix<T> faf_matrix_;

  double time_fit_;
  StopWatch timer_fit_;
  double time_u_;
  StopWatch timer_u_;
  double time_w_;
  StopWatch timer_w_;
  double time_init_;
  StopWatch timer_init_;
  double time_kmeans_;
  StopWatch timer_kmeans_;
  int num_iters_fit_;
  // Record the number of iterations for each column in w_matrix to converge
  Vector<T> num_iters_w_matrix_;


  // Initialization for generating the first clustering result
  void Init(const Matrix<T> &input_matrix) {
    x_matrix_ = input_matrix;
    n_ = input_matrix.rows();
    d_ = input_matrix.cols();
    // w_matrix_ is I initially
    w_matrix_ = Matrix<T>::Identity(d_, d_);
    num_iters_w_matrix_ = Vector<T>::Zero(q_);
  }

  // Initialization for generating alternative views when Y is already generated
  // from last run
  void Init() {
    h_matrix_ = Matrix<T>::Identity(n_, n_)
        - Matrix<T>::Constant(n_, n_, 1) / static_cast<T>(n_);
    y_matrix_temp_ = Matrix<T>::Zero(n_, c_);
    // Generate the kernel for the label matrix Y: K_y
    k_matrix_y_ = y_matrix_ * y_matrix_.transpose();
    // Generate Y tilde matrix in equation 5 from kernel matrix of Y
    y_matrix_tilde_ = h_matrix_ * k_matrix_y_ * h_matrix_;
    InitAMatrixList();
    // Coefficients for calculating phi
    waw_matrix_ = Matrix<T>::Zero(n_, n_);
    waf_matrix_ = Matrix<T>::Zero(n_, n_);
    faf_matrix_ = Matrix<T>::Zero(n_, n_);
    u_converge_ = false;
    w_converge_ = false;
    u_w_converge_ = false;
  }

  // Initialization for generating alternative views with a given Y
  void Init(const Matrix<T> &input_matrix, const Matrix<T> &y_matrix) {
    Init(input_matrix);
    CheckQD();
    h_matrix_ = Matrix<T>::Identity(n_, n_)
        - Matrix<T>::Constant(n_, n_, 1) / static_cast<T>(n_);
    y_matrix_temp_ = Matrix<T>::Zero(n_, c_);
    y_matrix_ = y_matrix;
    // Generate the kernel for the label matrix Y: K_y
    k_matrix_y_ = y_matrix_ * y_matrix_.transpose();
    // Generate Y tilde matrix in equation 5 from kernel matrix of Y
    y_matrix_tilde_ = h_matrix_ * k_matrix_y_ * h_matrix_;
    InitAMatrixList();
    // Coefficients for calculating phi
    waw_matrix_ = Matrix<T>::Zero(n_, n_);
    waf_matrix_ = Matrix<T>::Zero(n_, n_);
    faf_matrix_ = Matrix<T>::Zero(n_, n_);
    u_converge_ = false;
    w_converge_ = false;
    u_w_converge_ = false;
    time_fit_ = 0;
    time_u_ = 0;
    time_w_ = 0;
    time_init_ = 0;
    time_kmeans_ = 0;
    num_iters_fit_ = 0;
    num_iters_w_matrix_ = Vector<T>::Zero(q_);
  }

  void InitAMatrixList(void) {
    a_matrix_list_.resize(n_ * n_);
    for (int i = 0; i < n_; i++) {
      for (int j = 0; j < n_; j++) {
        Vector<T> delta_x_ij = x_matrix_.row(i) - x_matrix_.row(j);
        Matrix<T> a_ij = CpuOperations<T>::OuterProduct(delta_x_ij, delta_x_ij);
        a_matrix_list_[i * n_ + j] = a_ij;
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
    CheckFinite(u_matrix_normalized_, "normalized_u");
    kms.Fit(u_matrix_normalized_, c_);
    clustering_result_ = kms.GetLabels();
    if (y_matrix_.cols() == 0) {
      // When this is calculating Y0
      y_matrix_ = Matrix<T>::Zero(n_, c_);
      for (int i = 0; i < n_; i++)
        y_matrix_(i, clustering_result_(i)) = 1;
    } else {
      // When this is to calculate Y_i and append it to Y_[0~i-1]
      for (int i = 0; i < n_; i++)
        y_matrix_temp_(i, clustering_result_(i)) = 1;
      Matrix<T> y_matrix_new(n_, y_matrix_.cols() + c_);
      y_matrix_new << y_matrix_, y_matrix_temp_;
      y_matrix_ = y_matrix_new;
      // Reset the y_matrix_temp holder to zero
      y_matrix_temp_.setZero();
    }
  }

  void UpdateGOfW(Matrix<T> &g_of_w, const Vector<T> &w_l) {
    for (int i = 0; i < n_; i++) {
      for (int j = 0; j < n_; j++) {
        if (kernel_type_ == kGaussianKernel) {
          g_of_w(i, j) = g_of_w(i, j) *
              static_cast<T>(-w_l.transpose() * a_matrix_list_[i * n_ + j] *
                  w_l) / static_cast<T>(2 * pow(constant_, 2));
        }
      }
    }
  }

  void CheckFinite(const Matrix<T> &matrix, std::string name) {
    if (!matrix.allFinite()) {
      std::cout << name << " not finite: " << std::endl << matrix << std::endl;
      exit(1);
    }
  }

  void CheckFinite(const Vector<T> &vector, std::string name) {
    if (!vector.allFinite()) {
      std::cout << name << " not finite: " << std::endl << vector << std::endl;
      exit(1);
    }
  }


  Vector<T> GenWGradient(const Matrix<T> &g_of_w, const Vector<T> &w_l) {
    Vector<T> w_gradient = Vector<T>::Zero(d_);
    if (kernel_type_ == kGaussianKernel) {
      for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
          Matrix<T> &a_matrix_ij = a_matrix_list_[i * n_ + j];
          T exp_term = exp(static_cast<T>(-w_l.transpose() * a_matrix_ij * w_l)
                               / (2.0 * pow(constant_, 2)));
          w_gradient += -gamma_matrix_(i, j) * g_of_w(i, j) * exp_term *
              a_matrix_ij * w_l / pow(constant_, 2);
//          w_gradient += -gamma_matrix_(i, j) * g_of_w(i, j) *
//              exp( (-w_l.transpose() * a_matrix_ij * w_l) /
//                  (2 * pow(constant_, 2)) ) * a_matrix_ij * w_l;

        }
      }

    }
    return w_gradient;
  }

  void OptimizeU(void) {
    // If this is the first round and second,
    // then we use the full X to initialize the
    // U matrix, Otherwise, we project X to subspace W (n * d to n * q)
    Matrix<T> projected_x_matrix = x_matrix_ * w_matrix_;
    CheckFinite(projected_x_matrix, "projected_x_matrix");
    // Generate the kernel matrix based on kernel type from projected X
    k_matrix_ = CpuOperations<T>::GenKernelMatrix(
        projected_x_matrix, kernel_type_, constant_);


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
//    if (y_matrix_.rows() == 0) {
//      Print(u_matrix_, "u_matrix");
//      Print(u_matrix_normalized_, "u_norm");
//    }
  }

  void CheckFiniteOptimizeU(void) {

    CheckFinite(k_matrix_, "Kernel");
    CheckFinite(d_matrix_to_the_minus_half_, "d_matrix_to_minus_half");
    CheckFinite(l_matrix_, "L");
    CheckFinite(u_matrix_, "U");
  }

  void CheckFiniteOptimizeW(void) {
    CheckFinite(didj_matrix_, "didj");
    CheckFinite(gamma_matrix_, "Gamma");
    CheckFinite(w_matrix_, "W");
  }

  void OptimizeW(void) {
    // didj matrix contains the element (i, j) that equal to d_i * d_j
    didj_matrix_ = d_i_ * d_i_.transpose();
    // Generate the Gamma matrix in equation 5, which is a constant since
    // we have U fixed. Note that instead of generating one element of
    // gamma_ij on the fly as in the paper, we generate the whole gamma matrix
    // at one time and then access its entry of (i, j)
    // This is an element-wise operation
    // u*ut and didj matrix has the same size
    gamma_matrix_ = ((u_matrix_ * u_matrix_.transpose()).array() /
        didj_matrix_.array()).matrix() - lambda_ * y_matrix_tilde_;
    // After gamma_matrix is generated, we are optimizing gamma * kij as in 5
    // g_of_w is g(w_l) that is multiplied by g(w_(l+1)) in each iteration
    // of changing l.
    // Note that here the g_of_w is a n*n matrix because it contains A_ij
    // g_of_w(i, j) corresponding to exp(-w_T * A_ij * w / 2sigma^2)
    // When l = 0, g_of_w is 1
    // when l = 1, g_of_w is 1 .* g(w_1)
    // when l = 2, g_of_w is 1 .* g(w_1) .* g(w_2)...
    Matrix<T> g_of_w = Matrix<T>::Constant(n_, n_, 1);
    // If w_matrix is still I (d x d), now it is time to change it to d x q
    if (w_matrix_.cols() == d_)
      w_matrix_ = Matrix<T>::Identity(d_, q_);


    // We optimize each column in the W matrix
    for (int l = 0; l < w_matrix_.cols(); l++) {
      Vector<T> w_l;
      // Number of iterations in converging w_l
      int num_iters = 0;
      // Get orthogonal to make w_l orthogonal to vectors from w_0 to w_(l-1)
      // when l is not 0
      if (l == 0) {
        w_l = w_matrix_.col(l);
        w_l = w_l.array() / w_l.norm();
      } else {
        w_l = GenOrthonormal(w_matrix_.leftCols(l), w_matrix_.col(l));
      }
      w_matrix_.col(l) = w_l;
      // Search for the w_l that maximizes formula 5
      // The initial objective is set to the lowest number
      T objective = std::numeric_limits<T>::lowest();
      bool w_l_converged = false;
      while (!w_l_converged) {
        Vector<T> grad_f_vertical;
        T pre_objective = objective;
        // Calculate the w gradient in equation 13, then find the gradient
        // that is vertical to the space spanned by w_0 to w_l
        Vector<T> grad_f = GenWGradient(g_of_w, w_l);
        grad_f_vertical =
            GenOrthonormal(w_matrix_.leftCols(l + 1), grad_f);
        LineSearch(grad_f_vertical, &w_l, &alpha_, &objective);
        w_l = sqrt(1.0 - pow(alpha_, 2)) * w_l +
            alpha_ * grad_f_vertical;
        w_matrix_.col(l) = w_l;
        w_l_converged = CheckConverged(objective, pre_objective, threshold_);
        num_iters ++;
      }
      num_iters_w_matrix_(l) += num_iters;
      UpdateGOfW(g_of_w, w_l);
      // TODO: Need to learn about if using Vector<T> &w_l = w_matrix_.col(l)
      CheckFiniteOptimizeW();
    }
  }

  void LineSearch(const Vector<T> &gradient,
                  Vector<T> *w_l, float *alpha, T *objective) {
    *alpha = 1.0;
    float a1 = 0.1;
    float rho = 0.8;
    float phi_of_alpha = 0;
    float phi_of_zero = 0;
    float phi_of_zero_prime = 0;

    if (kernel_type_ == kGaussianKernel) {
      GenPhi(*alpha, *w_l, gradient, true,
             &phi_of_alpha, &phi_of_zero, &phi_of_zero_prime);
      if (phi_of_zero_prime < 0) {
        *w_l = -(*w_l);
        GenPhi(*alpha, *w_l, gradient, true,
               &phi_of_alpha, &phi_of_zero, &phi_of_zero_prime);
      }
      while (phi_of_alpha < phi_of_zero + *alpha * a1 * phi_of_zero_prime) {
        *alpha = *alpha * rho;
        GenPhi(*alpha, *w_l, gradient, false,
               &phi_of_alpha, &phi_of_zero, &phi_of_zero_prime);
      }
//      std::cout << "obj: " << phi_of_alpha << std::endl;
      *objective = phi_of_alpha;
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
  void GenPhi(const float &alpha,
              const Vector<T> &w_l,
              const Vector<T> &gradient,
              bool w_l_changed,
              float *phi_of_alpha,
              float *phi_of_zero,
              float *phi_of_zero_prime
  ) {

    if (kernel_type_ == kGaussianKernel) {
      float alpha_square = pow(alpha, 2);
      float sqrt_one_minus_alpha = pow((1 - alpha_square), 0.5);
      float denom = -1 / (2 * pow(constant_, 2));
      *phi_of_alpha = 0;

      if (w_l_changed) {
        GenPhiCoeff(w_l, gradient);
        *phi_of_zero = 0;
        *phi_of_zero_prime = 0;
      }

      for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
          T waw = waw_matrix_(i, j);
          T waf = waf_matrix_(i, j);
          T faf = faf_matrix_(i, j);
          T kij = exp(denom * ((faf - waw) * alpha_square +
              2 * waf * sqrt_one_minus_alpha * alpha + waw));
          *phi_of_alpha += gamma_matrix_(i, j) * kij;
          if (w_l_changed) {
            T kij = exp(denom * waw);
            *phi_of_zero += gamma_matrix_(i, j) * kij;
            *phi_of_zero_prime += gamma_matrix_(i, j) * denom * 2 * waf * kij;
          }
//          if (gen_prime) {
//            *phi_of_alpha_prime += gamma_matrix_(i, j) *
//                denom * (2 * waf * (1 - 2 * alpha_square) /
//                sqrt_one_minus_alpha + 2 * (faf - waw) * alpha) * kij;
//          }
        }
      }
    }
  }

  void GenPhiCoeff(const Vector<T> &w_l, const Vector<T> &gradient) {
    // Three terms used to calculate phi of alpha
    // They only change if w_l or gradient change
    for (int i = 0; i < n_; i++) {
      for (int j = 0; j < n_; j++) {
        Matrix<T> &a_matrix_ij = a_matrix_list_[i * n_ + j];
        waw_matrix_(i, j) = w_l.transpose() * a_matrix_ij * w_l;
        waf_matrix_(i, j) = w_l.transpose() * a_matrix_ij * gradient;
        faf_matrix_(i, j) = gradient.transpose() * a_matrix_ij * gradient;
      }
    }
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

  bool CheckConverged(const Matrix<T> &matrix, const Matrix<T> &pre_matrix,
                      const T &threshold) {
    if ( (matrix.rows() != pre_matrix.rows()) ||
        (matrix.cols() != pre_matrix.cols()) )
      return false;
    T change = CpuOperations<T>::FrobeniusNorm(matrix - pre_matrix)
        / CpuOperations<T>::FrobeniusNorm(pre_matrix);
    bool converged = (change < threshold);
    return converged;
  }

  bool CheckConverged(const Vector<T> &vector, const Vector<T> &pre_vector,
                      const T &threshold) {
    if ( vector.rows() != pre_vector.rows() )
      return false;
    T change = (vector - pre_vector).norm() / pre_vector.norm();
    bool converged = (change < threshold);
    return converged;
  }

  bool CheckConverged(const T &scalar, const T &pre_scalar, const T &threshold) {
    T change = (scalar - pre_scalar) / scalar;
    bool converged = (change < threshold);
    return converged;
  }
};
}  // namespace NICE

#endif  // CPP_INCLUDE_KDAC_H