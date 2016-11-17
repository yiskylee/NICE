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
#include <string>

#include "include/matrix.h"
#include "include/vector.h"
#include "include/cpu_operations.h"
#include "include/gpu_operations.h"
#include "include/svd_solver.h"
#include "include/gpu_svd_solver.h"
#include "include/util.h"
#include "include/gpu_util.h"
#include "include/kernel_types.h"
#include "include/timer.h"
#include "interface/clustering/clustering.h"

#define CALL_FUNC_NO_BUF(func)\
  switch (dtype_) {\
    case FLOAT:\
      return f_kdac_ -> func();\
    case DOUBLE:\
      return d_kdac_ -> func();\
    default:\
      std::cerr << "Unknown Data Type" << std::endl;\
      exit(1);\
}\

#define CALL_FUNC_ONE_BUF(func, in, row, col)\
  Py_buffer pybuf;\
  PyObject_GetBuffer(in, &pybuf, PyBUF_SIMPLE);\
  switch (dtype_) {\
    case FLOAT:\
      new (&output_fmat_) FMatrixMap(reinterpret_cast<float *>(pybuf.buf), row, col);\
      output_fmat_ = f_kdac_ -> func();\
      break;\
    case DOUBLE:\
      new (&output_dmat_) DMatrixMap(reinterpret_cast<double *>(pybuf.buf), row, col);\
      output_dmat_ = d_kdac_ -> func();\
      break;\
    default:\
      std::cerr << "Unknown Data Type" << std::endl;\
      exit(1);\
  }\

namespace Nice {

KDACInterface::KDACInterface()
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

void KDACInterface::GetTimePerIter(PyObject *time_per_iter,
                                   int num_iters,
                                   std::string stat_name) {
  KDACProfiler profiler;
  Py_buffer pybuf;
  PyObject_GetBuffer(time_per_iter, &pybuf, PyBUF_SIMPLE);
  DMatrixMap output(reinterpret_cast<double *>(pybuf.buf),
                    num_iters, 1);
  switch (dtype_) {
    case FLOAT:profiler = f_kdac_->GetProfiler();
      break;
    case DOUBLE:profiler = d_kdac_->GetProfiler();
      break;
    default:
      std::cerr << "Unknown Data Type" << std::endl;
      exit(1);
  }
  if (stat_name == "u_time_per_iter")
    output = profiler.u.GetTimePerIter();
  else if (stat_name == "w_time_per_iter")
    output = profiler.w.GetTimePerIter();
  else if (stat_name == "w_part1")
    output = profiler.w_part1.GetTimePerIter();
  else if (stat_name == "w_part2")
    output = profiler.w_part2.GetTimePerIter();
  else if (stat_name == "w_part3")
    output = profiler.w_part3.GetTimePerIter();
  else if (stat_name == "w_part4")
    output = profiler.w_part4.GetTimePerIter();
  else if (stat_name == "w_part5")
    output = profiler.w_part5.GetTimePerIter();
  else if (stat_name == "w_part6")
    output = profiler.w_part6.GetTimePerIter();
  else if (stat_name == "w_part7")
    output = profiler.w_part7.GetTimePerIter();
  else if (stat_name == "w_part8")
    output = profiler.w_part8.GetTimePerIter();
}

void KDACInterface::GetProfiler(boost::python::dict &profile) {
  KDACProfiler profiler;
  switch (dtype_) {
    case FLOAT:
      profiler = f_kdac_ -> GetProfiler();
    break;
    case DOUBLE:
      profiler = d_kdac_ -> GetProfiler();
      break;
    default:
      std::cerr << "Unknow Data Type" << std::endl;
      exit(1);
  }
  profile["init"] = profiler.init.GetTotalTime();
  profile["u"] = profiler.u.GetTotalTime();
  profile["w"] = profiler.w.GetTotalTime();
  profile["kmeans"] = profiler.kmeans.GetTotalTime();
  profile["fit"] = profiler.fit.GetTotalTime();
  profile["num_iters"] = profiler.u.GetNumIters();
}

void KDACInterface::SetupParams(const boost::python::dict &params) {

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
  bool verbose = false;
  bool has_verbose = false;
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
    if (strcmp("verbose", boost::python::extract<char *>(key_list[i])) == 0) {
      verbose = boost::python::extract<double>(params["verbose"]);
      has_verbose = true;
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
      if (has_verbose)
        f_kdac_->SetVerbose(verbose);
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
      if (has_verbose)
        d_kdac_->SetVerbose(verbose);
      break;
    default:
      std::cerr << "Unknown Data Type" << std::endl;
      exit(1);
  }

}

void KDACInterface::Fit() {
  CALL_FUNC_NO_BUF(Fit);
}

void KDACInterface::Fit(PyObject *in, int row, int col) {
  // Get the python object buffer
  Py_buffer pybuf;
  PyObject_GetBuffer(in, &pybuf, PyBUF_SIMPLE);

  // Initialze the input data
  switch (dtype_) {
    case FLOAT:
      new (&input_fmat_[0]) FMatrixMap(reinterpret_cast<float *>(pybuf.buf),
                                    row, col);
      f_kdac_ -> Fit(input_fmat_[0]);
      break;
    case DOUBLE:
      new (&input_dmat_[0]) DMatrixMap(reinterpret_cast<double *>(pybuf.buf),
                                    row, col);
      d_kdac_ -> Fit(input_dmat_[0]);
      break;
    default:
      std::cerr << "Unknown Data Type" << std::endl;
      exit(1);
  }
}

void KDACInterface::Fit(PyObject *in_1, int row_1, int col_1,
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
      f_kdac_ -> Fit(input_fmat_[0], input_fmat_[1]);
      break;
    case DOUBLE:
      if (input_dmat_.size() < 2) {
        input_dmat_.push_back(DMatrixMap(nullptr, 0, 0));
      }
      new (&input_dmat_[0]) DMatrixMap(reinterpret_cast<double *>(pybuf_1.buf),
                                    row_1, col_1);
      new (&input_dmat_[1]) DMatrixMap(reinterpret_cast<double *>(pybuf_2.buf),
                                    row_2, col_2);
      d_kdac_ -> Fit(input_dmat_[0], input_dmat_[1]);
      break;
    default:
      std::cerr << "Unknown Data Type" << std::endl;
      exit(1);
  }
}

void KDACInterface::Predict(PyObject *in, int row, int col) {
  CALL_FUNC_ONE_BUF(Predict, in, row, col);
}

void KDACInterface::GetU(PyObject *in, int row, int col) {
  CALL_FUNC_ONE_BUF(GetU, in, row, col);
}

void KDACInterface::GetW(PyObject *in, int row, int col) {
  CALL_FUNC_ONE_BUF(GetW, in, row, col);
}

int KDACInterface::GetD() {
  CALL_FUNC_NO_BUF(GetD);
}

int KDACInterface::GetN() {
  CALL_FUNC_NO_BUF(GetN);
}

int KDACInterface::GetQ() {
  CALL_FUNC_NO_BUF(GetQ);
}

}  // namespace Nice

void (Nice::KDACInterface::*Fit1)(PyObject *, int row, int col)
  = &Nice::KDACInterface::Fit;
void (Nice::KDACInterface::*Fit2)()
  = &Nice::KDACInterface::Fit;
void (Nice::KDACInterface::*Fit3)(PyObject *, int row_1, int col_1,
                                  PyObject *, int row_2, int col_2)
  = &Nice::KDACInterface::Fit;

BOOST_PYTHON_MODULE(Nice4Py) {
  boost::python::enum_<Nice::DataType>("DataType")
    .value("FLOAT", Nice::DataType::FLOAT)
    .value("DOUBLE", Nice::DataType::DOUBLE);
  boost::python::class_<Nice::KDACInterface>("KDAC", boost::python::init<>())
    .def("Fit", Fit1)
    .def("Fit", Fit2)
    .def("Fit", Fit3)
    .def("SetupParams", &Nice::KDACInterface::SetupParams)
    .def("Predict", &Nice::KDACInterface::Predict)
    .def("GetProfiler", &Nice::KDACInterface::GetProfiler)
    .def("GetW", &Nice::KDACInterface::GetW)
    .def("GetD", &Nice::KDACInterface::GetD)
    .def("GetN", &Nice::KDACInterface::GetN)
    .def("GetQ", &Nice::KDACInterface::GetQ)
    .def("GetTimePerIter", &Nice::KDACInterface::GetTimePerIter);
}
