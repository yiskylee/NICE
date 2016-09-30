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

#include <boost/python.hpp>
#include <typeinfo>
#include <memory>
#include "include/matrix.h"
#include "include/vector.h"
#include "include/cpu_operations.h"
#include "include/gpu_operations.h"
#include "include/svd_solver.h"
#include "include/gpu_svd_solver.h"
#include "include/util.h"
#include "include/gpu_util.h"

#include "interface/clustering/clustering.h"

namespace Nice {

KdacInterface::KdacInterface()
: PyInterface() {
  switch (dtype_) {
    case FLOAT:
      f_kdac_ = std::make_shared<Nice::KDAC<float> >();
      break;
    case DOUBLE:
      d_kdac_ = std::make_shared<Nice::KDAC<double> >();
      break;
    default:
      break;
  }

}

void KdacInterface::SetupParams(const boost::python::dict &params) {
  // Set up parameters according to model type
  switch (dtype_) {
    case FLOAT:
      f_kdac_->SetQ(2);
      f_kdac_->SetC(2);
      break;
    case DOUBLE:
      d_kdac_->SetQ(2);
      d_kdac_->SetC(2);
      break;
    default:
      break;
  }

}

void KdacInterface::Fit(PyObject *in, int row, int col) {
  // Get the python object buffer
  Py_buffer pybuf;
  PyObject_GetBuffer(in, &pybuf, PyBUF_SIMPLE);

  // Initialze the input data
  switch (dtype_) {
    case FLOAT:
      new (&input_fmat_) FMatrixMap(reinterpret_cast<float *>(pybuf.buf),
                                    row, col);
      break;
    case DOUBLE:
      new (&input_dmat_) DMatrixMap(reinterpret_cast<double *>(pybuf.buf),
                                    row, col);
      break;
    default:
      break;
  }

  // Run Fit
  if (dtype_ == FLOAT) {
    TemplateFit<float>(input_fmat_, f_kdac_.get());
  } else if (dtype_ == DOUBLE) {
    TemplateFit<double>(input_dmat_, d_kdac_.get());
  }


}

void KdacInterface::Fit() {
  // Run Fit
  if (dtype_ == FLOAT) {
    TemplateFit<float>(f_kdac_.get());
  } else if (dtype_ == DOUBLE) {
    TemplateFit<double>(d_kdac_.get());
  }

}

void KdacInterface::Predict(PyObject *in, int row, int col) {
  // Get python object buffer
  Py_buffer pybuf;
  PyObject_GetBuffer(in, &pybuf, PyBUF_SIMPLE);

  switch (dtype_) {
    case FLOAT:
      new (&output_fmat_) FMatrixMap(reinterpret_cast<float *>(pybuf.buf),
                                     row, col);
      break;
    case DOUBLE:
      new (&output_dmat_) DMatrixMap(reinterpret_cast<double *>(pybuf.buf),
                                     row, col);
      break;
    default:
      break;
  }

  if (dtype_ == FLOAT) {
    output_fmat_ = TemplatePredict<float>(f_kdac_.get());
  } else if (dtype_ == DOUBLE) {
    output_dmat_ = TemplatePredict<double>(d_kdac_.get());
  }

}

void KdacInterface::GetU(PyObject *in, int row, int col) {
  // Get python object buffer
  Py_buffer pybuf;
  PyObject_GetBuffer(in, &pybuf, PyBUF_SIMPLE);

  switch (dtype_) {
    case FLOAT:
      new (&output_fmat_) FMatrixMap(reinterpret_cast<float *>(pybuf.buf),
                                     row, col);
      break;
    case DOUBLE:
      new (&output_dmat_) DMatrixMap(reinterpret_cast<double *>(pybuf.buf),
                                     row, col);
      break;
    default:
      break;
  }

  if (dtype_ == FLOAT) {
    output_fmat_ = TemplateGetU<float>(f_kdac_.get());
  } else if (dtype_ == DOUBLE) {
    output_dmat_ = TemplateGetU<double>(d_kdac_.get());
  }

}

void KdacInterface::GetW(PyObject *in, int row, int col) {
  // Get python object buffer
  Py_buffer pybuf;
  PyObject_GetBuffer(in, &pybuf, PyBUF_SIMPLE);

  switch (dtype_) {
    case FLOAT:
      new (&output_fmat_) FMatrixMap(reinterpret_cast<float *>(pybuf.buf),
                                     row, col);
      break;
    case DOUBLE:
      new (&output_dmat_) DMatrixMap(reinterpret_cast<double *>(pybuf.buf),
                                     row, col);
      break;
    default:
      break;
  }

  if (dtype_ == FLOAT) {
    output_fmat_ = TemplateGetW<float>(f_kdac_.get());
  } else if (dtype_ == DOUBLE) {
    output_dmat_ = TemplateGetW<double>(d_kdac_.get());
  }

}


}  // namespace Nice


void (Nice::KdacInterface::*Fit1)(PyObject *, int row, int col)
  = &Nice::KdacInterface::Fit;
void (Nice::KdacInterface::*Fit2)()
  = &Nice::KdacInterface::Fit;

BOOST_PYTHON_MODULE(Nice4Py) {
  boost::python::enum_<Nice::DataType>("DataType")
    .value("INT", Nice::DataType::INT)
    .value("FLOAT", Nice::DataType::FLOAT)
    .value("DOUBLE", Nice::DataType::DOUBLE);
  boost::python::class_<Nice::KdacInterface>("KDAC",
                                           boost::python::init<>())
    .def("Fit", Fit1)
    .def("Fit", Fit2)
    .def("SetupParams", &Nice::KdacInterface::SetupParams)
    .def("Predict", &Nice::KdacInterface::Predict)
    .def("GetU", &Nice::KdacInterface::GetU)
    .def("GetW", &Nice::KdacInterface::GetW);
}
