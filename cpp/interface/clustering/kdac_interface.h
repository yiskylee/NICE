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

#ifndef CPP_INTERFACE_CLUSTERING_KDAC_INTERFACE_H_
#define CPP_INTERFACE_CLUSTERING_KDAC_INTERFACE_H_

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

namespace Nice {
// The numpy array is stored in row major
//using IMatrixMap = Eigen::Map< Eigen::Matrix<int, Eigen::Dynamic,
//                                             Eigen::Dynamic, Eigen::RowMajor> >;
//using FMatrixMap = Eigen::Map< Eigen::Matrix<float, Eigen::Dynamic,
//                                             Eigen::Dynamic, Eigen::RowMajor> >;
using DMatrixMap = Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor> >;

template<typename T>
using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor> >;
template<typename T>
using VectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic,
                                           1, Eigen::ColMajor>>;

template<typename T>
class KDACInterface {
 public:
  explicit KDACInterface(std::string device_type) {
    if (device_type == "cpu")
      kdac_ = std::make_shared<Nice::KDACCPU<T>>();
#ifdef CUDA_AND_GPU
    else if (device_type == "gpu")
      kdac_ = std::make_shared<Nice::KDACGPU<T>>();
#endif
  }

  ~KDACInterface() {}

  void SetupParams(const boost::python::dict &params) {
    // Obtain parameters from python
    boost::python::list key_list = params.keys();
    bool has_kernel = false;
    bool has_sigma = false;
    KernelType  kernel = kGaussianKernel;
    double sigma = 1.0;
    for (int i = 0; i < boost::python::len(key_list); i++) {
      char *param = boost::python::extract<char *>(key_list[i]);
      if (strcmp("c", param) == 0) {
        int c = boost::python::extract<int>(params["c"]);
        kdac_ -> SetC(c);
      } else if (strcmp("q", param) == 0) {
        int q = boost::python::extract<int>(params["q"]);
        kdac_ -> SetQ(q);
      } else if (strcmp("max_time", param) == 0) {
        int max_time = boost::python::extract<int>(params["max_time"]);
        kdac_ -> SetMaxTime(max_time);
      } else if (strcmp("method", param) == 0) {
        char* method = boost::python::extract<char *>(params["method"]);
        kdac_ -> SetMethod(method);
      } else if (strcmp("lambda", param) == 0) {
        double lambda = boost::python::extract<double>(params["lambda"]);
        kdac_ -> SetLambda(lambda);
      } else if (strcmp("sigma", param) == 0) {
        sigma = boost::python::extract<double>(params["sigma"]);
        has_sigma = true;
      } else if (strcmp("verbose", param) == 0) {
        bool verbose = boost::python::extract<double>(params["verbose"]);
        kdac_ -> SetVerbose(verbose);
      } else if (strcmp("vectorization", param) == 0) {
        bool vectorization =
            boost::python::extract<double>(params["vectorization"]);
        kdac_ -> SetVectorization(vectorization);
      } else if (strcmp("debug", param) == 0) {
        bool debug = boost::python::extract<double>(params["debug"]);
        kdac_ -> SetDebug(debug);
      } else if (strcmp("threshold1", param) == 0) {
        double thresh1 = boost::python::extract<double>(params["threshold1"]);
        kdac_ -> SetThreshold1(thresh1);
      } else if (strcmp("threshold2", param) == 0) {
        double thresh2 = boost::python::extract<double>(params["threshold2"]);
        kdac_ -> SetThreshold1(thresh2);
      } else if (strcmp("kernel", param) == 0) {
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
      } else {
        std::cout << "Parameter: " << param << " not recognized." << std::endl;
        exit(1);
      }
    }
    if (has_kernel && has_sigma)
      kdac_ -> SetKernel(kernel, sigma);
  }
  void GetProfiler(boost::python::dict &profile) {
    KDACProfiler profiler = kdac_ -> GetProfiler();
    profile["init"] = profiler["init"].GetTotalTime();
    profile["u"] = profiler["u"].GetTotalTime();
    profile["w"] = profiler["w"].GetTotalTime();
    profile["u_avg"] = profiler["u"].GetAvgTimePerIter();
    profile["w_avg"] = profiler["w"].GetAvgTimePerIter();
    profile["kmeans"] = profiler["kmeans"].GetTotalTime();
    profile["fit"] = profiler["fit"].GetTotalTime();
    profile["fit_avg"] = profiler["fit"].GetAvgTimePerIter();
    profile["num_iters"] = profiler["u"].GetNumIters();
    profile["gen_phi"] = profiler["gen_phi"].GetTotalTime();
    profile["gen_grad"] = profiler["gen_grad"].GetTotalTime();
    profile["update_g_of_w"] = profiler["update_g_of_w"].GetTotalTime();
  }
  void GetTimePerIter(PyObject *time_per_iter_obj,
                      int num_iters, std::string stat_name) {
    Py_buffer time_per_iter_buf;
    KDACProfiler profiler;
    PyObject_GetBuffer(time_per_iter_obj, &time_per_iter_buf, PyBUF_SIMPLE);
    DMatrixMap time_per_iter(reinterpret_cast<double *>(time_per_iter_buf.buf),
                      num_iters, 1);
    profiler = kdac_ -> GetProfiler();
    try {
      time_per_iter = profiler[stat_name].GetTimePerIter();
    } catch (const char *msg) {
      std::cerr << msg << " for " << stat_name << std::endl;
      exit(-1);
    }
    PyBuffer_Release(&time_per_iter_buf);
  }
  void Fit(PyObject *input_obj, int row, int col) {
    // Get the python object buffer
    Py_buffer input_buf;
    PyObject_GetBuffer(input_obj, &input_buf, PyBUF_SIMPLE);
    MatrixMap<T> input(reinterpret_cast<T *>(input_buf.buf), row, col);
    kdac_ -> Fit(input);
    PyBuffer_Release(&input_buf);
  }
  void Fit() {
    kdac_ -> Fit();
  }
  void Fit(PyObject *input_obj, int row_1, int col_1,
           PyObject *label_obj, int row_2, int col_2) {
    Py_buffer input_buf;
    Py_buffer label_buf;
    // Get the python object buffer
    PyObject_GetBuffer(input_obj, &input_buf, PyBUF_SIMPLE);
    PyObject_GetBuffer(label_obj, &label_buf, PyBUF_SIMPLE);
//    MatrixMap<T> input(nullptr, 0, 0);
//    MatrixMap<T> label(nullptr, 0, 0);
//    new (&input)
//        MatrixMap<T>(reinterpret_cast<T *>(input_buf.buf), row_1, col_1);
//    new (&label)
//        MatrixMap<T>(reinterpret_cast<T *>(label_buf.buf), row_2, col_2);
    MatrixMap<T> input(reinterpret_cast<T *>(input_buf.buf), row_1, col_1);
    MatrixMap<T> label(reinterpret_cast<T *>(label_buf.buf), row_2, col_2);
    kdac_ -> Fit(input, label);
    PyBuffer_Release(&input_buf);
    PyBuffer_Release(&label_buf);
  }

  void Predict(PyObject *clustering_results_obj, int row, int col) {
    Py_buffer clustering_results_buf;
    PyObject_GetBuffer(clustering_results_obj, &clustering_results_buf, PyBUF_SIMPLE);
    VectorMap<T> clustering_results(
        reinterpret_cast<T *>(clustering_results_buf.buf), row);
    clustering_results = kdac_ -> Predict();
    PyBuffer_Release(&clustering_results_buf);
  }
  void GetU(PyObject *u_obj, int row, int col) {
    Py_buffer u_buf;
    PyObject_GetBuffer(u_obj, &u_buf, PyBUF_SIMPLE);
    MatrixMap<T> u(reinterpret_cast<T *>(u_buf.buf), row, col);
    u = kdac_ -> GetU();
    PyBuffer_Release(&u_buf);
  }
  void GetW(PyObject *w_obj, int row, int col) {
    Py_buffer w_buf;
    PyObject_GetBuffer(w_obj, &w_buf, PyBUF_SIMPLE);
    MatrixMap<T> w(reinterpret_cast<T *>(w_buf.buf), row, col);
    w = kdac_ -> GetW();
    PyBuffer_Release(&w_buf);
  }
  void GetK(PyObject *k_obj, int row) {
    Py_buffer k_buf;
    PyObject_GetBuffer(k_obj, &k_buf, PyBUF_SIMPLE);
    MatrixMap<T> k(reinterpret_cast<T *>(k_buf.buf), row, row);
    k = kdac_ -> GetK();
    PyBuffer_Release(&k_buf);
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
  void DiscardLastRun() {
    kdac_ -> DiscardLastRun();
  }
  void SetW(PyObject *input_obj, int row, int col) {
    Py_buffer input_buf;
    PyObject_GetBuffer(input_obj, &input_buf, PyBUF_SIMPLE);
    MatrixMap<T> w_matrix(reinterpret_cast<T *>(input_buf.buf), row, col);
    kdac_ -> SetW(w_matrix);
    PyBuffer_Release(&input_buf);
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

#endif  // CPP_INTERFACE_CLUSTERING_KDAC_INTERFACE_H_