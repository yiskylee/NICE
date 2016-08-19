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

#ifndef CPP_INTERFACE_PY_INTERFACE_H_
#define CPP_INTERFACE_PY_INTERFACE_H_

#include <boost/python.hpp>

#include <map>
#include <vector>
#include <iostream>

#include "Eigen/Dense"
#include "Eigen/Core"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/cpu_operations.h"
#include "include/gpu_operations.h"
#include "include/util.h"

namespace Nice {

enum DataType {
  INT = 0,
  FLOAT,
  DOUBLE
};

enum ModelType {
  KMEANS = 0,
  INVERSE,
  OTHERS
};


using IMatrixMap = Eigen::Map< Matrix<int> >;
using FMatrixMap = Eigen::Map< Matrix<float> >;
using DMatrixMap = Eigen::Map< Matrix<double> >;
using IMatrix = Matrix<int>;
using FMatrix = Matrix<float>;
using DMatrix = Matrix<double>;

class PyInterface {
 private:
  DataType dtype_;
  std::map<ModelType, boost::python::dict> param_map_;

  IMatrix input_imat_;
  FMatrix input_fmat_;
  DMatrix input_dmat_;
  IMatrix output_imat_;
  FMatrix output_fmat_;
  DMatrix output_dmat_;

  template <typename T>
  void RunKmeans(const boost::python::dict &param,
    const Matrix<T> &in,
    Matrix<T> *out) {
    *out = in + in * 3.;
  }

  // Only for test
  template <typename T>
  void RunInverse(const Matrix<T> &in,
    Matrix<T> *out) {
    // Has to be this way due to a bug of eigen
    Matrix<T> temp = CpuOperations<T>::Inverse(in);
    *out = temp;

    // out = CpuOperations<T>::Inverse(in);
  }

 public:
  PyInterface();
  explicit PyInterface(DataType dtype);
  void Init(const char *path, int row, int col);
  void Init(PyObject *in, int row, int col);
  void SetupParams(const boost::python::dict &params, ModelType model_type);
  void Run(ModelType model_type, PyObject *out, int row, int col);
};

}  // namespace Nice

#endif  // CPP_INTERFACE_PY_INTERFACE_H_
