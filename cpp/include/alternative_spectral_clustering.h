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

#include "include/matrix.h"
#include "include/vector.h"
#include "include/kernel_types.h"
#include <vector>

namespace Nice {

template<typename T>
class AlternativeSpectralClustering {
 private:
  int num_features_;
  int num_samples_;
  int num_clusters_;
  float sigma_;
  float polynomial_order_;
  int alternative_dimension_;
  float lambda_;
  float alpha_;
  Matrix<T> data_matrix_;
  Matrix<T> kernel_matrix_;
  Matrix<T> h_matrix_;
  Matrix<bool> y_matrix_;
  int pre_num_clusters_;
  KernelType kernel_type_;

  // output matrices
  Matrix<T> d_matrix_;
  Matrix<T> u_matrix_;
  Matrix<T> w_matrix_;

  // output
  Vector<int> assignments_;
  Matrix<bool> binary_allocation_;

 public:
//  AlternativeSpectralClustering();
  AlternativeSpectralClustering(const Matrix<T> &data_matrix,
                                int num_clusters = 2,
                                KernelType kernel_type = kGaussianKernel,
                                float sigma = 1,
                                float lambda = 1,
                                float alpha = 1,
                                float polynomial_order = 2,
                                int alternative_dimension = 1);
  void optimize_gaussian_kernel(void);
  Vector<unsigned long> FitPredict(void);
  Matrix<T> MatrixU();
  Matrix<T> MatrixV();
  Matrix<T> MatrixW();
};

template<typename T>
AlternativeSpectralClustering<T>::AlternativeSpectralClustering(
    const Matrix<T> &data_matrix,
    int num_clusters = 2,
    KernelType kernel_type = kGaussianKernel,
    float sigma = 1,
    float lambda = 1,
    float alpha = 1,
    float polynomial_order = 2,
    int alternative_dimension = 1) {
  data_matrix_ = data_matrix;
  num_samples_ = data_matrix_.rows();
  num_features_ = data_matrix_.cols();
  num_clusters_ = num_clusters;
  kernel_type_ = kernel_type;
  sigma_ = sigma;
  lambda_ = lambda;
  alpha_ = alpha;
  polynomial_order_ = polynomial_order;
  alternative_dimension_ = alternative_dimension;
}

template<typename T>
void AlternativeSpectralClustering<T>::optimize_gaussian_kernel(void) {
  h_matrix_ = Matrix<T>::Identity(num_samples_, num_samples_);


}


template<typename T>
Vector<unsigned long> AlternativeSpectralClustering<T>::FitPredict(void) {
  if(kernel_type_ == kGaussianKernel) {
    optimize_gaussian_kernel();
  }
}

}
#endif  // CPP_INCLUDE_ALTERNATIVE_SPECTRAL_CLUSTERING_H_
