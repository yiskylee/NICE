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

#ifndef CPP_INTERFACE_CLUSTERING_SPECTRAL_INTERFACE_H_
#define CPP_INTERFACE_CLUSTERING_SPECTRAL_INTERFACE_H_

#include <boost/python.hpp>

#include <map>
#include <memory>
#include <vector>
#include <iostream>
#include <string>

#include "Eigen/Dense"
#include "Eigen/Core"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/cpu_operations.h"
#include "include/gpu_operations.h"
#include "include/spectral_clustering.h"
#include "include/util.h"

namespace Nice {
// The numpy array is stored in row major
// using
// IMatrixMap = Eigen::Map< Eigen::Matrix<int, Eigen::Dynamic,
//                                          Eigen::Dynamic, Eigen::RowMajor> >;

using DMatrixMap = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
                                            Eigen::Dynamic, Eigen::RowMajor> >;

template<typename T>
using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor> >;
template<typename T>
using VectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic,
                                           1, Eigen::ColMajor>>;

template<typename T>
class SpectralInterface {
 public:
  explicit SpectralInterface() {
    spectral_ = std::make_shared<Nice::SpectralClustering<T>>();
  }

  ~SpectralInterface() {}

  void Fit(PyObject *input_obj, int row_1, int col_1, unsigned int k) {
    Py_buffer input_buf;
    PyObject_GetBuffer(input_obj, &input_buf, PyBUF_SIMPLE);
    MatrixMap<T> input(reinterpret_cast<T *>(input_buf.buf), row_1, col_1);
    spectral_->Fit(input, k);
    PyBuffer_Release(&input_buf);
  }
  void SetSigma(T sigma) {
    spectral_->SetSigma(sigma);
  }
  void GetLabels(PyObject *l_obj, int row, int col) {
    Py_buffer l_buf;
    PyObject_GetBuffer(l_obj, &l_buf, PyBUF_SIMPLE);
    MatrixMap<T> l(reinterpret_cast<T *>(l_buf.buf), row, col);
    l = spectral_->GetLabels();
    PyBuffer_Release(&l_buf);
  }
  void FitPredict(PyObject *input_obj, int row_i, int col_i, unsigned int k,
                  PyObject *l_obj, int row_l, int col_l) {
    Py_buffer input_buf, l_buf;
    PyObject_GetBuffer(input_obj, &input_buf, PyBUF_SIMPLE);
    PyObject_GetBuffer(l_obj, &l_buf, PyBUF_SIMPLE);
    MatrixMap<T> input(reinterpret_cast<T *>(input_buf.buf), row_i, col_i);
    MatrixMap<T> l(reinterpret_cast<T *>(l_buf.buf), row_l, col_l);
    spectral_->Fit(input, k);
    l = spectral_->GetLabels();
    PyBuffer_Release(&input_buf);
    PyBuffer_Release(&l_buf);
  }
 protected:
  std::shared_ptr<Nice::SpectralClustering<T>> spectral_;
};

}  // namespace Nice
#endif  //  CPP_INTERFACE_CLUSTERING_SPECTRAL_INTERFACE_H_
