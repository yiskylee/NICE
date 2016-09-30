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
#include "include/kdac.h"
#include "include/util.h"

#include "interface/py_interface.h"

namespace Nice {

class KdacInterface : public PyInterface{
 private:

  std::shared_ptr<Nice::KDAC<float> > f_kdac_;
  std::shared_ptr<Nice::KDAC<double> > d_kdac_;

  template <typename T>
  void TemplateFit(const Matrix<T> &in,
           Nice::KDAC<T> *kdac) {
    kdac->Fit(in);
  }

  template <typename T>
  void TemplateFit(Nice::KDAC<T> *kdac) {
    kdac->Fit();
  }

  template <typename T>
  Matrix<T> TemplatePredict(Nice::KDAC<T> *kdac) {
    return kdac->Predict();
  }

  template <typename T>
  Matrix<T> TemplateGetU(Nice::KDAC<T> *kdac) {
    return kdac->GetU();
  }

  template <typename T>
  Matrix<T> TemplateGetW(Nice::KDAC<T> *kdac) {
    return kdac->GetW();
  }

 public:
  KdacInterface();
  void SetupParams(const boost::python::dict &params);
  void Fit(PyObject *in, int row, int col);
  void Fit();
  void Predict(PyObject *in, int row, int col);
  void GetU(PyObject *in, int row, int col);
  void GetW(PyObject *in, int row, int col);

};

}  // namespace Nice

#endif  // CPP_INTERFACE_CLUSTERING_CLUSTERING_H_
