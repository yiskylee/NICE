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

#ifndef CPP_INCLUDE_ALTERNATIVE_SPECTRAL_CLUSTERING_H_
#define CPP_INCLUDE_ALTERNATIVE_SPECTRAL_CLUSTERING_H_

#include <functional>
#include <vector>
#include <cmath>
#include "include/matrix.h"
#include "include/vector.h"
#include "include/svd_solver.h"
#include "include/kmeans.h"
#include "include/spectral_clustering.h"
#include "Eigen/Core"
#include "include/util.h"
#include "include/kernel_types.h"


namespace Nice {

template<typename T>
class AlternativeSpectralClustering {
 public:
  AlternativeSpectralClustering(){
  }
  AlternativeSpectralClustering(const Matrix<T> &data_matrix,
                                int num_clusters) {
    data_matrix_ = data_matrix;
    CenterData(data_matrix_);
    num_samples_ = data_matrix_.rows();
    num_features_ = data_matrix_.cols();
    alternative_dimension_ = num_features_ - 1;
    num_clusters_ = num_clusters;
    kernel_type_ = kGaussianKernel;
    sigma_ = 1;
    lambda_ = 1;
    alpha_ = 0.01;
    polynomial_order_ = 2;
    kernel_matrix_ = Matrix<T>::Zero(num_samples_, num_samples_);
    normalized_u_matrix_ = Matrix<T>::Zero(num_samples_, num_clusters_);
    pre_num_clusters_ = 0;
  }
  void CenterData(Matrix<T> &data_matrix) {
    Matrix<T> centering_matrix =
        (Matrix<T>::Identity(data_matrix.rows(), data_matrix.rows()) -
        (1.0 / float(data_matrix.rows())) *
        Matrix<T>::Constant(data_matrix.rows(), data_matrix.rows(), 1));
    data_matrix = centering_matrix * data_matrix;
  }
  void InitHMatrix(void) {
    h_matrix_ = Matrix<T>::Identity(num_samples_, num_samples_)
        - Matrix<T>::Constant(num_samples_, num_samples_, 1)
            / float(num_samples_);
  }
  void InitWMatrix(void) {
    if (w_matrix_.rows() == 0 or w_matrix_.cols() == 0)
      w_matrix_ = Matrix<T>::Identity(num_features_, num_features_);
    else
      w_matrix_ = w_matrix_.block(0, 0, num_samples_,
                                  alternative_dimension_ - 1);
  }
  void UOptimize(void) {
    l_matrix_ = d_matrix_ * kernel_matrix_ * d_matrix_;
    l_matrix_ = h_matrix_ * l_matrix_ * h_matrix_;
    SvdSolver<T> solver;
    solver.Compute(l_matrix_);
    u_matrix_ = solver.MatrixU().leftCols(num_clusters_);
  }
  void OptimizaGaussianKernel(void) {
    bool w_u_converge = false;
    while (!w_u_converge) {
      InitHMatrix();
      InitWMatrix();
      CalcGaussianKernel();
      UOptimize();
      if (pre_num_clusters_ == 0)
        return;
      WOptimizeGaussian();
    }
  }
  Matrix<T> CreateYTilde(void) {
    if (y_matrix_.rows() != 0 && y_matrix_.cols() != 0) {
      Matrix<T> kernel_y = y_matrix_ * y_matrix_.transpose();
      Matrix<T> inner_p = h_matrix_ * kernel_y * h_matrix_;
      return d_matrix_ * inner_p * d_matrix_;
    }
    else {
      T inner_p = 0;
      return d_matrix_ * inner_p * d_matrix_;
    }
  }
  void WOptimizeGaussian(void) {
    Matrix<T> y_tilde = CreateYTilde();
    Matrix<T> previous_gw = Matrix<T>::Constant(num_samples_, num_samples_,
                                                1);
    bool w_converge = false;
    float last_w = 0;

    for (int m = 0; m < alternative_dimension_; m++) {
      w_matrix_.col(m) = GetOrthogonalVector(m, w_matrix_.col(m));
      while (!w_converge) {
        Matrix<T> w_l = w_matrix_.col(m);
      }
     }
  }

  Vector<T> GetOrthogonalVector(int m, Vector<T> input_vector) {
    int count_down = m;
    while (count_down != 0) {
      count_down --;
      Vector<T> w_prev = w_matrix_.col(count_down);
      Vector<T> projected_direction =
          (w_prev.dot(input_vector) / w_prev.dot(w_prev)) * w_prev;
      input_vector = input_vector - projected_direction;
    }
    input_vector = input_vector / input_vector.norm();
    return input_vector;
  }

  void CalcGaussianKernel(void) {
    // This for loop generates the kernel matrix using gaussian kernel
    for (unsigned long i = 0; i < num_samples_; i++) {
      for (unsigned long j = 0; j < num_samples_; j++) {
        // Calculate vector[i] - vector[j] for all (i,j) pairs
        Vector<T> i_j_diff = data_matrix_.row(i) - data_matrix_.row(j);
        // Variable entry is used to store the v[i] * w * wT * v[j]T
        T entry = i_j_diff.transpose() * w_matrix_ * w_matrix_.transpose()
            * i_j_diff;
        // Then the entry in the kernel matrix is replaced by
        // esp(-entry / (2*sigma^2)), The reason that the entry is
        // not directly calculated is to avoid some weird Eigen errors
        // that do not allow doing dividing operations on a chain of matrix
        // operations
        kernel_matrix_(i, j) = exp(-entry / (2 * sigma_ * sigma_));
      }
    }
    Vector<T> dia = kernel_matrix_.rowwise().sum().array().sqrt().unaryExpr(
        std::ptr_fun(util::reciprocal<T>));
    d_matrix_ = dia.asDiagonal();
  }

  void NormalizeEachURow() {
    normalized_u_matrix_ = u_matrix_.array().colwise() /
        u_matrix_.rowwise().norm().array();
  }

  void RunKMeans() {
    KMeans<T> km;
    allocation_ = km.FitPredict(normalized_u_matrix_, num_clusters_);
  }

  Vector<int> FitPredict(void) {
    if (kernel_type_ == kGaussianKernel)
      OptimizaGaussianKernel();
    NormalizeEachURow();
    RunKMeans();
    Vector<int> v;
    return v;
  }

  int num_features_;  // input data dimensions: d
  int num_samples_;
  int num_clusters_;
  float sigma_;  // Sigma value for the gaussian kernel
  int polynomial_order_;  // Order for the polynomial kernel
  int alternative_dimension_;  // q
  float lambda_;  // Alternative cluster tuning parameter
  float alpha_;
  Matrix<T> data_matrix_;
  Matrix<T> kernel_matrix_;  // num_samples_ * num_samples_
  Matrix<T> h_matrix_;  // centering matrix: num_samples_ * num_samples_
  Matrix<T> y_matrix_;  // all previous allocations size = num_samples *
                           // (num_clusters * pre_num_clusters)
  int pre_num_clusters_;  // the number of previous clusters
  KernelType kernel_type_;  // Gaussian, polynomial or linear

  // output matrices
  Matrix<T> d_matrix_;  // Degree matrix: num_samples_ * num_samples_
  Matrix<T> u_matrix_;  // Columns as eigenvectors,
                        // size: num_samples_ * num_clusters_
  Matrix<T> normalized_u_matrix_;
                        // normalized u matrix by L2 norm
                        // size: num_samples_ * num_clusters_
  Matrix<T> w_matrix_;  // size: num_features * alternative_dimension_
                        // initially: num_features * num_features
  // output
  Vector<int> allocation_;
  Matrix<int> binary_allocation_;

  // temporary
  Matrix<T> l_matrix_; // For testing if l_matrix is calculated correctly
};

}
#endif  // CPP_INCLUDE_ALTERNATIVE_SPECTRAL_CLUSTERING_H_
