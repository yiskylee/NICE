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

#ifndef CPP_INCLUDE_PY_INTERFACE_H_
#define CPP_INCLUDE_PY_INTERFACE_H_

#include <map>
#include <vector>
#include <boost/python.hpp>

#include "Eigen/Dense"
#include "Eigen/Core"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/util.h"

namespace Nice {

enum DataType {
  INT = 0,
  FLOAT,
  DOUBLE
};

enum ModelType {
  KMEANS = 0,
  OTHERS
};

using IMatrixMap = Eigen::Map< Matrix<int> >;
using FMatrixMap = Eigen::Map< Matrix<float> >;
using DMatrixMap = Eigen::Map< Matrix<double> >;

using IVectorMap = Eigen::Map< Vector<int> >;
using FVectorMap = Eigen::Map< Vector<float> >;
using DVectorMap = Eigen::Map< Vector<double> >;

class PyInterface {
 private:
  int row_;
  int col_;
  DataType dtype_;
  std::map<ModelType, boost::python::dict> param_map_;

  std::shared_ptr<IMatrixMap> input_imat_;
  std::shared_ptr<FMatrixMap> input_fmat_;
  std::shared_ptr<DMatrixMap> input_dmat_;
  std::shared_ptr<IMatrixMap> output_imat_;
  std::shared_ptr<FMatrixMap> output_fmat_;
  std::shared_ptr<DMatrixMap> output_dmat_;
  std::shared_ptr<IVectorMap> input_ivec_;
  std::shared_ptr<FVectorMap> input_fvec_;
  std::shared_ptr<DVectorMap> input_dvec_;
  std::shared_ptr<IVectorMap> output_ivec_;
  std::shared_ptr<FVectorMap> output_fvec_;
  std::shared_ptr<DVectorMap> output_dvec_;
  
  template <typename T>
  void RunKmeans(boost::python::dict &param,
    const Eigen::MatrixBase<T>& in,
    Eigen::MatrixBase<T>& out) { out = in; } 
 public:
  PyInterface();
  PyInterface(DataType dtype);
  void Init(const char *path, int row, int col);
  void Init(PyObject *in, int row, int col);
  void SetupParams(boost::python::dict &params, ModelType model_type);
  void Run(ModelType model_type, PyObject *out, int row, int col);
};

}  // namespace Nice

#endif  // CPP_INCLUDE_PY_INTERFACE_H_
