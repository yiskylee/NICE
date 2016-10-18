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
#include <cstring>

#include "include/matrix.h"
#include "include/vector.h"
#include "include/cpu_operations.h"
#include "include/gpu_operations.h"
#include "include/svd_solver.h"
#include "include/gpu_svd_solver.h"
#include "include/util.h"
#include "include/gpu_util.h"
#include "include/kernel_types.h"

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

void KdacInterface::GetTimes(boost::python::dict &times) {
  switch (dtype_) {
    case FLOAT:
      times["init"] = TemplateGetTimeInit<float>(f_kdac_.get());
      times["u"] = TemplateGetTimeU<float>(f_kdac_.get());
      times["w"] = TemplateGetTimeW<float>(f_kdac_.get());
      times["kmeans"] = TemplateGetTimeKMeans<float>(f_kdac_.get());
      times["fit"] = TemplateGetTimeFit<float>(f_kdac_.get());
      times["num_iters_fit"] = TemplateGetNumItersFit<float>(f_kdac_.get());
      times["num_iters_w"] = TemplateGetNumItersWMatrix<float>(f_kdac_.get());
      break;
    case DOUBLE:
      times["init"] = TemplateGetTimeInit<double>(d_kdac_.get());
      times["u"] = TemplateGetTimeU<double>(d_kdac_.get());
      times["w"] = TemplateGetTimeW<double>(d_kdac_.get());
      times["kmeans"] = TemplateGetTimeKMeans<double>(d_kdac_.get());
      times["fit"] = TemplateGetTimeFit<double>(d_kdac_.get());
      times["num_iters_fit"] = TemplateGetNumItersFit<double>(d_kdac_.get());
      times["num_iters_w"] = TemplateGetNumItersWMatrix<double>(d_kdac_.get());
      break;
    default:
      break;
  }
}

void KdacInterface::SetupParams(const boost::python::dict &params) {

  // Obtain parameters from python
  boost::python::list key_list = params.keys();
  int c = 2;
  bool has_c = false;
  int q = 2;
  bool has_q = false;
  KernelType kernel = kGaussianKernel;
  bool has_kernel = false;
  double lambda = 1.0;
  bool has_lambda = false;
  double sigma = 1.0;
  bool has_sigma = false;
  for (int i = 0; i < boost::python::len(key_list); i++) {
    if (strcmp("c", boost::python::extract<char *>(key_list[i])) == 0) {
      c = boost::python::extract<int>(params["c"]);
      has_c = true;
      continue;
    }
    if (strcmp("q", boost::python::extract<char *>(key_list[i])) == 0) {
      q = boost::python::extract<int>(params["q"]);
      has_q = true;
      continue;
    }
    if (strcmp("kernel", boost::python::extract<char *>(key_list[i])) == 0) {
      if (strcmp("Gaussian", 
                 boost::python::extract<char *>(params["kernel"])) == 0) {
        kernel = kGaussianKernel;
      }
      if (strcmp("Linear", 
                 boost::python::extract<char *>(params["kernel"])) == 0) {
        kernel = kLinearKernel;
      }
      if (strcmp("Polynomial", 
                 boost::python::extract<char *>(params["kernel"])) == 0) {
        kernel = kPolynomialKernel;
      }
      has_kernel = true;
      continue;
    }
    if (strcmp("lambda", boost::python::extract<char *>(key_list[i])) == 0) {
      lambda = boost::python::extract<double>(params["lambda"]);
      has_lambda = true;
      continue;
    }
    if (strcmp("sigma", boost::python::extract<char *>(key_list[i])) == 0) {
      sigma = boost::python::extract<double>(params["sigma"]);
      has_sigma = true;
      continue;
    }
  }

  // Set up parameters according to model type
  switch (dtype_) {
    case FLOAT:
      if (has_q)
        f_kdac_->SetQ(q);
      if (has_c)
        f_kdac_->SetC(c);
      if (has_lambda)
        f_kdac_->SetLambda(lambda);
      if (has_kernel && has_sigma)
        f_kdac_->SetKernel(kernel, sigma);
      break;
    case DOUBLE:
      if (has_q)
        d_kdac_->SetQ(q);
      if (has_c)
        d_kdac_->SetC(c);
      if (has_lambda)
        d_kdac_->SetLambda(lambda);
      if (has_kernel && has_sigma)
        d_kdac_->SetKernel(kernel, sigma);
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
      new (&input_fmat_[0]) FMatrixMap(reinterpret_cast<float *>(pybuf.buf),
                                    row, col);
      break;
    case DOUBLE:
      new (&input_dmat_[0]) DMatrixMap(reinterpret_cast<double *>(pybuf.buf),
                                    row, col);
      break;
    default:
      break;
  }

  // Run Fit
  if (dtype_ == FLOAT) {
    TemplateFit<float>(input_fmat_[0], f_kdac_.get());
  } else if (dtype_ == DOUBLE) {
    TemplateFit<double>(input_dmat_[0], d_kdac_.get());
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

void KdacInterface::Fit(PyObject *in_1, int row_1, int col_1,
                        PyObject *in_2, int row_2, int col_2) {
  // Get the python object buffer
  Py_buffer pybuf_1;
  PyObject_GetBuffer(in_1, &pybuf_1, PyBUF_SIMPLE);
  Py_buffer pybuf_2;
  PyObject_GetBuffer(in_2, &pybuf_2, PyBUF_SIMPLE);

  // Initialze the input data
  switch (dtype_) {
    case FLOAT:
      if (input_fmat_.size() < 2) {
        input_fmat_.push_back(FMatrixMap(nullptr, 0, 0));
      }
      new (&input_fmat_[0]) FMatrixMap(reinterpret_cast<float *>(pybuf_1.buf),
                                    row_1, col_1);
      new (&input_fmat_[1]) FMatrixMap(reinterpret_cast<float *>(pybuf_2.buf),
                                    row_2, col_2);
      break;
    case DOUBLE:
      if (input_dmat_.size() < 2) {
        input_dmat_.push_back(DMatrixMap(nullptr, 0, 0));
      }
      new (&input_dmat_[0]) DMatrixMap(reinterpret_cast<double *>(pybuf_1.buf),
                                    row_1, col_1);
      new (&input_dmat_[1]) DMatrixMap(reinterpret_cast<double *>(pybuf_2.buf),
                                    row_2, col_2);
      break;
    default:
      break;
  }

  // Run Fit
  if (dtype_ == FLOAT) {
    TemplateFit<float>(input_fmat_[0], input_fmat_[1], f_kdac_.get());
  } else if (dtype_ == DOUBLE) {
    TemplateFit<double>(input_dmat_[0], input_dmat_[1], d_kdac_.get());
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

int KdacInterface::GetD() {
  if (dtype_ == FLOAT) {
    return TemplateGetD<float>(f_kdac_.get());
  } else if (dtype_ == DOUBLE) {
    return TemplateGetD<double>(d_kdac_.get());
  }
  return -1;
}

int KdacInterface::GetN() {
  if (dtype_ == FLOAT) {
    return TemplateGetN<float>(f_kdac_.get());
  } else if (dtype_ == DOUBLE) {
    return TemplateGetN<double>(d_kdac_.get());
  }
  return -1;
}

int KdacInterface::GetQ() {
  if (dtype_ == FLOAT) {
    return TemplateGetQ<float>(f_kdac_.get());
  } else if (dtype_ == DOUBLE) {
    return TemplateGetQ<double>(d_kdac_.get());
  }
  return -1;
}

}  // namespace Nice


void (Nice::KdacInterface::*Fit1)(PyObject *, int row, int col)
  = &Nice::KdacInterface::Fit;
void (Nice::KdacInterface::*Fit2)()
  = &Nice::KdacInterface::Fit;
void (Nice::KdacInterface::*Fit3)(PyObject *, int row_1, int col_1,
                                  PyObject *, int row_2, int col_2)
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
    .def("Fit", Fit3)
    .def("SetupParams", &Nice::KdacInterface::SetupParams)
    .def("GetTimes", &Nice::KdacInterface::GetTimes)
    .def("Predict", &Nice::KdacInterface::Predict)
    .def("GetU", &Nice::KdacInterface::GetU)
    .def("GetW", &Nice::KdacInterface::GetW)
    .def("GetD", &Nice::KdacInterface::GetD)
    .def("GetN", &Nice::KdacInterface::GetN)
    .def("GetQ", &Nice::KdacInterface::GetQ);
}
