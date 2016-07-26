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
namespace Nice {
template<typename T>
class KDAC {
 public:
  /// This is the default constructor for KDAC
  /// Number of clusters c and reduced dimension q will be both set to 2
  KDAC() {
    SetCQ(2, 2);
    SetKernel(kGaussianKernel, 1.0);
  }
  /// This is the KDAC constructor that takes one parameter c
  /// \param c
  /// Number of clusters c, the reducded dimension q will be set to c
  KDAC(int c) {
    // reduced dimension q by default equals to k
    SetCQ(c, c);
    SetKernel(kGaussianKernel, 1.0);
  }
  /// This is the KDAC constructor that takes both number of clusters c and
  /// reduced dimension q
  /// \param c
  /// Number of clusters c
  /// \param q
  /// Reduced dimension q
  KDAC(int c, int q) {
    SetCQ(c, q);
    SetKernel(kGaussianKernel, 1.0);
  }

  /// This is the KDAC constructor that takes number of clusters c,
  /// reduced dimension q and kernel_type
  /// \param c
  /// Number of clusters c
  /// \param q
  /// Reduced dimension q
  /// \param kernel_type
  /// Can be chosen from kGaussianKernel, kPolynomialKernel and kLinearKernel
  KDAC(int c, int q, KernelType kernel_type) {
    SetCQ(c, q);
    SetKernel(kernel_type, 1.0);
  }

  /// This is the KDAC constructor that takes number of clusters c,
  /// reduced dimension q, kernel_type and the constant number associated
  /// with a specific kernel type
  /// \param c
  /// Number of clusters c
  /// \param q
  /// Reduced dimension q
  /// \param kernel_type
  /// Can be chosen from kGaussianKernel, kPolynomialKernel and kLinearKernel
  /// \param constant
  /// In Gaussian kernel, this is sigma;
  /// In Polynomial kernel, this is constant c
  /// In Linear kernel, this is c as well
  KDAC(int c, int q, KernelType kernel_type, float constant) {
    SetCQ(c, q);
    SetKernel(kernel_type, constant);
  }

  /// This function creates the first clustering result
  /// \param input_matrix
  /// The input matrix of n samples and d features where each row
  /// represents a sample
  /// \return
  /// It only generates the clustering result but does not returns it
  /// Users can use Predict() to get the clustering result returned
  void Fit(const Matrix<T> &input_matrix) {
    // Following the pseudo code in Algorithm 1 in the paper
    Init(input_matrix);
    u_matrix_ = GenU();
  }

  /// This function creates the alternative clustering result
  /// Users can change c and q using other Fit() functions
  /// Must be called after \ref Fit(const Matrix<T> &input_matrix)
  void Fit(void) {

  }

  /// This function creates the alternative clustering result
  /// with user-specified number of clusters c
  /// reduced dimension q would be set to equal to c
  /// Must be called after \ref Fit(const Matrix<T> &input_matrix)
  /// \param c
  /// Number of clusters c
  /// \sa
  /// \ref Fit(const int c, const int q)
  void Fit(const int c) {
    SetCQ(c, c);
    Fit();
  }

  /// This function creates the alternative clustering result
  /// with user-specified number of clusters c and reduced dimension q
  /// Must be called after \ref Fit(const Matrix<T> &input_matrix)
  /// \param c
  /// Number of clusters c
  /// \param q
  /// Reduced dimension q
  /// \sa
  /// \ref Fit(int c)
  void Fit(const int c, const int q) {
    SetCQ(c, q);
    Fit();
  }

  /// Running Predict() after Fit() returns
  /// the current clustering result as a Vector of T
  /// \return
  /// A NICE vector of T that specifies the clustering result
  Vector<T> Predict(void) {

  }


 private:
  int c_;  // cluster number c
  int q_;  // reduced dimension q
  int n_;  // number of samples in input data X
  int d_;  // input data X dimension d
  KernelType kernel_type_;  // The kernel type of the kernel matrix
  float constant_;  // In Gaussian kernel, this is sigma;
                    // In Polynomial kernel, this is constant c
                    // In Linear kernel, this is c as well
  bool u_converge;  // If matrix U reaches convergence, false by default
  bool v_converge;  // If matrix V reaches convergence, false by default
  Matrix<T> x_matrix_;  // Input matrix X (n by d)
  Matrix<T> w_matrix_;  // Transformation matrix W (d by q). Initialized to I
  Matrix<bool> y_matrix_;  // Labeling matrix Y (n by (c0 + c1 + c2 + ..))
  Matrix<T> d_matrix_;  // Diagonal degree matrix D (n by n)
  Matrix<T> k_matrix_;  // Kernel matrix K (n by n)
  Matrix<T> u_matrix_;  // Soft labeling matrix U (n by c)
  Matrix<T> h_matrix_;  // Centering matrix (n by n)
  Vector<T> clustering_result;  // Current clustering result


  // Initialization
  void Init(const Matrix<T> &input_matrix) {
    x_matrix_ = input_matrix;
    n_ = input_matrix.rows();
    d_ = input_matrix.cols();
    w_matrix_ = Matrix<T>::Identity(d_, q_);
    y_matrix_ = Matrix<bool>::Constant(n_, c_, 0);
    d_matrix_ = Matrix<T>::Constant(n_, n_, 0);
    k_matrix_ = Matrix<T>::Constant(n_, n_, 0);
    u_matrix_ = Matrix<T>::Constant(n_, c_, 0);
    h_matrix_ = Matrix<T>::Identity(n_, n_)
        - Matrix<T>::Constant(n_, n_, 1) / float(n_);
    clustering_result = Vector<T>::Zero(n_);
    u_converge = false;
    v_converge = false;
  }

  // Check if q is not bigger than c
  void SetCQ(int c, int q) {
    if (q <= c) {
      c_ = c;
      q_ = q;
    } else {
      std::cerr <<
          "Reduced dimension q cannot exceed cluster number c" << std::endl;
      exit(1);
    }
  }
  void SetKernel(KernelType kernel_type, float constant) {
    kernel_type_ = kernel_type;
    constant_ = constant;
  }

  Matrix<T> GenU(void) {
//    kernel_matrix_ =
//    return Matrix<T>::Zero(n_, c_);
  }

};
}  // namespace NICE

#endif  // CPP_INCLUDE_KDAC_H
