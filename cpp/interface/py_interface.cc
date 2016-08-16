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

#include "include/matrix.h"
#include "include/vector.h"
#include "include/cpu_operations.h"
#include "include/gpu_operations.h"
#include "include/svd_solver.h"
#include "include/gpu_svd_solver.h"
#include "include/util.h"
#include "include/gpu_util.h"

#include "py_interface.h"

#include <boost/python.hpp>

namespace Nice {

PyInterface::PyInterface() 
: dtype_(DOUBLE) {}

PyInterface::PyInterface(DataType dtype)
: dtype_(dtype) {}

void PyInterface::Init(const char *path, int row, int col) {
  // Set dimension
  row_ = row;
  col_ = col;

  // Create corresponding matrix/vector based on the given path
}

void PyInterface::Init(PyObject *in, int row, int col) {
  Py_buffer pybuf;
  PyObject_GetBuffer(in, &pybuf, PyBUF_SIMPLE);

  switch (dtype_) {
    case INT:
      break;
    case FLOAT:
      break;
    case DOUBLE:
      if (col_ == 1) {
        input_dvec_.reset(new DVectorMap((double *)(pybuf.buf), row_, col_));
      } else {
        input_dmat_.reset(new DMatrixMap((double *)(pybuf.buf), row_, col_));
      }
      break;
    default:
      break;
  }
}

void PyInterface::SetupParams(boost::python::dict &params, ModelType model_type) {
}

void PyInterface::Run(ModelType model_type, PyObject *out, int row, int col) {
  Py_buffer pybuf;
  PyObject_GetBuffer(out, &pybuf, PyBUF_SIMPLE);

  boost::python::dict d;
  switch (dtype_) {
    case INT:
      break;
    case FLOAT:
      break;
    case DOUBLE:
      if (col_ == 1) {
        output_dvec_ = std::make_shared<DVectorMap>((double *)(pybuf.buf),
                                                   row, col);
      } else {
        output_dmat_ = std::make_shared<DMatrixMap>((double *)(pybuf.buf),
                                                   row, col);
      }
      break;
    default:
      break;
  }

  switch (model_type) {
    case KMEANS:
      RunKmeans(d, *input_dmat_, *output_dmat_);
      break;
    default:
      break;
  }
 
}

}  // namespace Nice


void (Nice::PyInterface::*Init1)(const char *, int row, int col)
  = &Nice::PyInterface::Init;
void (Nice::PyInterface::*Init2)(PyObject *, int row, int col)
  = &Nice::PyInterface::Init;

BOOST_PYTHON_MODULE(Nice4Py) {
  boost::python::enum_<Nice::DataType>("DataType")
    .value("INT", Nice::DataType::INT)
    .value("FLOAT", Nice::DataType::FLOAT)
    .value("DOUBLE", Nice::DataType::DOUBLE)
    ;
  boost::python::enum_<Nice::ModelType>("ModelType")
    .value("KMEANS", Nice::ModelType::KMEANS)
    .value("OTHERS", Nice::ModelType::OTHERS)
    ;
  boost::python::class_<Nice::PyInterface>("PyInterface", boost::python::init<>())
    .def(boost::python::init<Nice::DataType>())
    .def("Init", Init1)
    .def("Init", Init2)
    .def("SetupParams", &Nice::PyInterface::SetupParams)
    .def("Run", &Nice::PyInterface::Run)
    ;
}
