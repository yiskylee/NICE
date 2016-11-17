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
#include "include/kdac_profiler.h"
#include "interface/py_interface.h"

namespace Nice {

class KDACInterface : public PyInterface{
 private:

  std::shared_ptr<Nice::KDAC<float> > f_kdac_;
  std::shared_ptr<Nice::KDAC<double> > d_kdac_;

 public:
  KDACInterface();
  void SetupParams(const boost::python::dict &params);
  void GetProfiler(boost::python::dict &profiler);
  void GetTimePerIter(PyObject *time_per_iter,
                      int max_num_iters, std::string stat_name);
  void Fit(PyObject *in, int row, int col);
  void Fit();
  void Fit(PyObject *in_1, int row_1, int col_1,
           PyObject *in_2, int row_2, int col_2);
  void Predict(PyObject *in, int row, int col);
  void GetU(PyObject *in, int row, int col);
  void GetW(PyObject *in, int row, int col);
  int GetD();
  int GetN();
  int GetQ();

};

}  // namespace Nice

#endif  // CPP_INTERFACE_CLUSTERING_CLUSTERING_H_