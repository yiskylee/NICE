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

typedef void (*FunctionPointers)(boost::python::dict &);

enum DataType {
  INT = 0,
  FLOAT,
  DOUBLE
};

template <typename T>
using MatrixMap = Eigen::Map< Matrix<T> >;

template <typename T>
using VectorMap = Eigen::Map< Vector<T> >;

class PyInterface {
 private:
  int row_;
  int col_;
  std::map<const char*, boost::python::dict> param_map_;
  std::map<const char *, FunctionPointers> func_map_;
  void RunKmeans(boost::python::dict &param);
 public:
  PyInterface(){}//;
  void Init(const char *, int row, int col, DataType){}//;
  void Init(PyObject *, int row, int col, DataType){}//;
  void SetupParams(boost::python::dict &params, const char *){}//;
  void GetResults(PyObject *, int &row, int &col){}//;
  void Run(const char *){}//;
  template <typename T> static MatrixMap<T> input_mat_;
  template <typename T> static VectorMap<T> input_vec_; 
  template <typename T> static MatrixMap<T> output_mat_;
  template <typename T> static VectorMap<T> output_vec_;
};

}  // namespace Nice

#endif  // CPP_INCLUDE_PY_INTERFACE_H_
