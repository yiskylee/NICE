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
//#include "include/model.h"
#include <vector>
#include "dlib/clustering.h"
#include "include/matrix.h"

namespace Nice {

//template<typename T>
//class SpectralClustering : public Model<T> {
// public:
//  Vector<unsigned long> FitPredict(const Matrix<T> &input_data,
//                       int k);
//};

template<typename T>
class SpectralClustering {
 public:
  std::vector<unsigned long> FitPredict(const Matrix<T> &input_data, int k);
};

template<typename T>
std::vector<unsigned long> SpectralClustering<T>::FitPredict(
    const Matrix<T> &input_data, int k) {
  int num_features = input_data.cols();
  typedef dlib::matrix<T> sample_type;
  typedef dlib::radial_basis_kernel<sample_type> kernel_type;
  std::vector<sample_type> samples;
  sample_type m;
  m.set_size(2,1);
  for (long i = 0; i < 10; i++) {
    for (long j = 0; j < num_features; j++) {
      m(j) = input_data(i, j);
      samples.push_back(m);
    }
  }
  std::vector<unsigned long> assignments = dlib::spectral_cluster(
      kernel_type(0.1), samples, 3);
//  Vector<unsigned long> v(assignments.data());
//  Vector<unsigned long> v;
  return assignments;
}
//template class SpectralClustering<int>;
//template class SpectralClustering<float>;
//template class SpectralClustering<double>;
}

#endif  // CPP_INCLUDE_SPECTRAL_CLUSTERING_H_
