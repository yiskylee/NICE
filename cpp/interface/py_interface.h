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
#include <memory>
#include <vector>
#include <iostream>

#include "Eigen/Dense"
#include "Eigen/Core"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/cpu_operations.h"
#include "include/gpu_operations.h"
#include "include/kdac.h"
#include "include/util.h"

namespace Nice {

enum DataType {
  FLOAT = 0,
  DOUBLE
};

// The numpy array is stored in row major
using IMatrixMap = Eigen::Map< Eigen::Matrix<int, Eigen::Dynamic,
                               Eigen::Dynamic, Eigen::RowMajor> >;
using FMatrixMap = Eigen::Map< Eigen::Matrix<float, Eigen::Dynamic,
                               Eigen::Dynamic, Eigen::RowMajor> >;
using DMatrixMap = Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic,
                               Eigen::Dynamic, Eigen::RowMajor> >;
template<typename T>
using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic,
                             Eigen::Dynamic, Eigen::RowMajor> >;


class PyInterface {
 protected:
  DataType dtype_;
  boost::python::dict params_;

  std::vector<IMatrixMap> input_imat_;
  std::vector<FMatrixMap> input_fmat_;
  std::vector<DMatrixMap> input_dmat_;
  IMatrixMap output_imat_;
  FMatrixMap output_fmat_;
  DMatrixMap output_dmat_;

 public:
  PyInterface()
  : dtype_(DOUBLE),
    output_imat_(nullptr, 0, 0),
    output_fmat_(nullptr, 0, 0),
    output_dmat_(nullptr, 0, 0) {
    input_imat_.push_back(IMatrixMap(nullptr, 0, 0));
    input_fmat_.push_back(FMatrixMap(nullptr, 0, 0));
    input_dmat_.push_back(DMatrixMap(nullptr, 0, 0));
  }

  explicit PyInterface(DataType dtype)
  : dtype_(dtype),
    output_imat_(nullptr, 0, 0),
    output_fmat_(nullptr, 0, 0),
    output_dmat_(nullptr, 0, 0) {
    input_imat_.push_back(IMatrixMap(nullptr, 0, 0));
    input_fmat_.push_back(FMatrixMap(nullptr, 0, 0));
    input_dmat_.push_back(DMatrixMap(nullptr, 0, 0));
  }

  virtual void SetupParams(const boost::python::dict &params) = 0;
};

}  // namespace Nice

#endif  // CPP_INTERFACE_PY_INTERFACE_H_