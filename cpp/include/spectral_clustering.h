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

#ifndef CPP_INCLUDE_SPECTRAL_CLUSTERING_H_
#define CPP_INCLUDE_SPECTRAL_CLUSTERING_H_

#include "include/matrix.h"
#include "include/vector.h"
#include "include/kmeans.h"
#include "include/svd_solver.h"
#include <vector>
#include <Eigen/Dense> 

namespace Nice {

template<typename T>
class SpectralClustering {
 public: 
  SpectralClustering()
  :
  sigma_(1.0), kmeans_(), svd_() {}

  void Fit(const Matrix<T> &input_data, int k) {
    k_ = k;
    SimilarityGraph(input_data);
    ComputeLaplacian();
    int n_ = similarity_.rows();
    u_.resize(n_, n_);
    y_.resize(n_, k_); 
    svd_.Compute(laplacian_); 
    u_ = svd_.MatrixV();
    y_ = u_.block(0, 0, n_, k_);
    kmeans_.Fit(y_, k_);
    labels_.resize(n_, 1);
    labels_ = kmeans_.GetLabels();
  }
  void SetSigma(T s) {
    sigma_ = s;
  }
  void SimilarityGraph(const Matrix<T> &input_data) {
    int rows = input_data.rows();
    similarity_.resize(rows, rows);
    degrees_.resize(rows, rows);
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < rows; j++) {
        degrees_(i, j) = 0;
      }
    }
    int counter = 0;
    for(int i = 0; i < rows; i++) {
      for(int j = counter; j < rows; j++) {
        Vector<T> x_i = input_data.row(i);
        Vector<T> x_j = input_data.row(j);
        double sim = exp((x_i - x_j).norm())/(2*sigma_);
        similarity_(i, j) = sim;
        similarity_(j, i) = sim;
      }
      counter++;
    }
    for(int i = 0; i < rows; i++) {
      int sum = 0;
      for(int j = 0; j < rows; j++) {
        sum += similarity_(i, j);
      }
      degrees_(i, i) = sum;
    }
  }
  void ComputeLaplacian() {
    laplacian_.resize(similarity_.rows(), similarity_.rows());
    laplacian_ = degrees_ - similarity_;
  }
  Matrix <T> GetLabels() {
    return labels_;
  }
 private:
  int k_;
  T sigma_;
  KMeans<T> kmeans_;
  SvdSolver<T> svd_;
  Matrix<T> similarity_;
  Matrix<T> degrees_;
  Matrix<T> laplacian_;
  Matrix<T> u_;
  Vector<T> y_;
  Matrix<T> labels_;
};
}

#endif  // CPP_INCLUDE_SPECTRAL_CLUSTERING_H_
