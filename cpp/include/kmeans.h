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

#ifndef CPP_INCLUDE_KMEANS_H
#define CPP_INCLUDE_KMEANS_H

#include "include/matrix.h"
#include "include/vector.h"
#include <vector>
#include "dlib/clustering.h"


namespace Nice {

template<typename T>
class KMeans {
 public:
  Vector<T> FitPredict(const Matrix<T> &input_data, int k) {
    int num_features = input_data.cols();
    int num_samples = input_data.rows();
    typedef dlib::matrix<T> sample_type;
    typedef dlib::radial_basis_kernel<sample_type> kernel_type;
    std::vector<sample_type> samples;
    std::vector<sample_type> initial_centers;
    sample_type m;
    m.set_size(num_features, 1);
    for (long i = 0; i < num_samples; i++) {
      for (long j = 0; j < num_features; j++)
        m(j) = input_data(i, j);
      samples.push_back(m);
    }
    dlib::kcentroid<kernel_type> kc(kernel_type(0.01), 0.0001, 20);
    dlib::kkmeans<kernel_type> km(kc);
    km.set_number_of_centers(k);
    dlib::pick_initial_centers(k, initial_centers, samples, km.get_kernel());
    km.train(samples, initial_centers);
    Vector<T> assignments(num_samples);
    for (long i = 0; i < num_samples; i++) {
//      std::cout << samples[i] << std::endl;
      assignments[i] = km(samples[i]);
    }
    return assignments;
  }
};
}
#endif  // CPP_INCLUDE_KMEANS_H
