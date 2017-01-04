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

#ifndef CPP_INTERFACE_CLUSTERING_CLUSTERING_H_
#define CPP_INTERFACE_CLUSTERING_CLUSTERING_H_

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
#include "include/kdac_cpu.h"
#include "include/kdac_gpu.h"
#include "include/util.h"
#include "include/kdac_profiler.h"
#include "interface/py_interface.h"
#include "../../include/kdac_profiler.h"

namespace Nice {
template<typename T>
class KDACInterface {
 public:
  KDACInterface(std::string device_type) {
    if (device_type == "cpu")
      kdac_ = std::make_shared<Nice::KDACCPU<T>>();
    else if (device_type == "gpu")
      kdac_ = std::make_shared<Nice::KDACGPU<T>>();
  }
  void SetupParams(const boost::python::dict &params) {
    // Obtain parameters from python
    boost::python::list key_list = params.keys();
    bool has_kernel = false;
    bool has_sigma = false;
    KernelType  kernel = kGaussianKernel;
    double sigma = 1.0;
    for (int i = 0; i < boost::python::len(key_list); i++) {
      if (strcmp("c", boost::python::extract<char *>(key_list[i])) == 0) {
        int c = boost::python::extract<int>(params["c"]);
        kdac_ -> SetC(c);
        continue;
      }
      if (strcmp("q", boost::python::extract<char *>(key_list[i])) == 0) {
        int q = boost::python::extract<int>(params["q"]);
        kdac_ -> SetQ(q);
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
        double lambda = boost::python::extract<double>(params["lambda"]);
        kdac_ -> SetLambda(lambda);
        continue;
      }
      if (strcmp("sigma", boost::python::extract<char *>(key_list[i])) == 0) {
        sigma = boost::python::extract<double>(params["sigma"]);
        has_sigma = true;
        continue;
      }
      if (strcmp("verbose", boost::python::extract<char *>(key_list[i])) == 0) {
        bool verbose = boost::python::extract<double>(params["verbose"]);
        kdac_ -> SetVerbose(verbose);
        continue;
      }
    }
    if (has_kernel && has_sigma)
      kdac_ -> SetKernel(kernel, sigma);
  }
  void GetProfiler(boost::python::dict &profile) {
    KDACProfiler profiler = kdac_ -> GetProfiler();
    profile["init"] = profiler.init.GetTotalTime();
    profile["u"] = profiler.u.GetTotalTime();
    profile["w"] = profiler.w.GetTotalTime();
    profile["kmeans"] = profiler.kmeans.GetTotalTime();
    profile["fit"] = profiler.fit.GetTotalTime();
    profile["num_iters"] = profiler.u.GetNumIters();
    profile["gen_a"] = profiler.gen_a.GetTotalTime();
    profile["gen_phi"] = profiler.gen_phi.GetTotalTime();
  }
  void GetTimePerIter(PyObject *time_per_iter,
                      int num_iters, std::string stat_name) {
    KDACProfiler profiler;
    Py_buffer pybuf;
    PyObject_GetBuffer(time_per_iter, &pybuf, PyBUF_SIMPLE);
    DMatrixMap output(reinterpret_cast<double *>(pybuf.buf),
                      num_iters, 1);
    profiler = kdac_ -> GetProfiler();
    if (stat_name == "u_time_per_iter")
      output = profiler.u.GetTimePerIter();
    else if (stat_name == "w_time_per_iter")
      output = profiler.w.GetTimePerIter();
    else if (stat_name == "gen_phi_time_per_iter")
      output = profiler.gen_phi.GetTimePerIter();
  }
//  void Fit(PyObject *in, int row, int col) {
//    // Get the python object buffer
//    Py_buffer pybuf;
//    PyObject_GetBuffer(in, &pybuf, PyBUF_SIMPLE);
//    MatrixMap<T> input(nullptr, 0, 0);
//    new (&input) MatrixMap<T>(reinterpret_cast<T *>(pybuf.buf), row, col);
//    kdac_ -> Fit(input);
//  }
//  void Fit() {
//    kdac_ -> Fit();
//  }
  void Fit(PyObject *in_1, int row_1, int col_1,
           PyObject *in_2, int row_2, int col_2) {
    // Get the python object buffer
    Py_buffer pybuf_1;
    PyObject_GetBuffer(in_1, &pybuf_1, PyBUF_SIMPLE);
    Py_buffer pybuf_2;
    PyObject_GetBuffer(in_2, &pybuf_2, PyBUF_SIMPLE);
    MatrixMap<T> input_1(nullptr, 0, 0);
    MatrixMap<T> input_2(nullptr, 0, 0);
    new (&input_1)
        MatrixMap<T>(reinterpret_cast<T *>(pybuf_1.buf), row_1, col_1);
    new (&input_2)
        MatrixMap<T>(reinterpret_cast<T *>(pybuf_2.buf), row_2, col_2);
    kdac_ -> Fit(input_1, input_2);
  }
  void Predict(PyObject *in, int row, int col) {
    Py_buffer pybuf;
    PyObject_GetBuffer(in, &pybuf, PyBUF_SIMPLE);
    MatrixMap<T> output(nullptr, 0, 0);
    new (&output) MatrixMap<T>(reinterpret_cast<T *>(pybuf.buf), row, col);
    output = kdac_ -> Predict();
  }
  void GetU(PyObject *in, int row, int col) {
    Py_buffer pybuf;
    PyObject_GetBuffer(in, &pybuf, PyBUF_SIMPLE);
    MatrixMap<T> output(nullptr, 0, 0);
    new (&output) MatrixMap<T>(reinterpret_cast<T *>(pybuf.buf), row, col);
    output = kdac_ -> GetU();
  }
  void GetW(PyObject *in, int row, int col) {
    Py_buffer pybuf;
    PyObject_GetBuffer(in, &pybuf, PyBUF_SIMPLE);
    MatrixMap<T> output(nullptr, 0, 0);
    new (&output) MatrixMap<T>(reinterpret_cast<T *>(pybuf.buf), row, col);
    output = kdac_ -> GetW();
  }
  int GetD() {
    return kdac_ -> GetD();
  }
  int GetN() {
    return kdac_ -> GetN();
  }
  int GetQ() {
    return kdac_ -> GetQ();
  }

 protected:
  std::shared_ptr<Nice::KDAC<T>> kdac_;
};

//template <typename T>
//class KDACGPUInterface: public KDACInterface<T> {
// public:
//  KDACGPUInterface() {
//    this->kdac_ = std::make_shared<Nice::KDACGPU<T>>();
//  }
//};
//
//template <typename T>
//class KDACCPUInterface: public KDACInterface<T> {
// public:
//  KDACCPUInterface() {
//    this->kdac_ = std::make_shared<Nice::KDACCPU<T>>();
//  }
//};

}  // namespace Nice

#endif  // CPP_INTERFACE_CLUSTERING_CLUSTERING_H_