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

void (Nice::KDACInterface<float>::*Fit1Float)(PyObject *, int row, int col)
  = &Nice::KDACInterface<float>::Fit;
void (Nice::KDACInterface<float>::*Fit2Float)()
  = &Nice::KDACInterface<float>::Fit;
void (Nice::KDACInterface<float>::*Fit3Float)(PyObject *, int row_1, int col_1,
                                  PyObject *, int row_2, int col_2)
  = &Nice::KDACInterface<float>::Fit;

void (Nice::KDACInterface<double>::*Fit1Double)(PyObject *, int row, int col)
= &Nice::KDACInterface<double>::Fit;
void (Nice::KDACInterface<double>::*Fit2Double)()
= &Nice::KDACInterface<double>::Fit;
void (Nice::KDACInterface<double>::*Fit3Double)(PyObject *, int row_1, int col_1,
                                         PyObject *, int row_2, int col_2)
= &Nice::KDACInterface<double>::Fit;

BOOST_PYTHON_MODULE(Nice4Py) {
  boost::python::class_<Nice::KDACInterface<float>>
      ("KDACFLOAT", boost::python::init<>())
    .def("Fit", Fit1Float)
    .def("Fit", Fit2Float)
    .def("Fit", Fit3Float)
    .def("SetupParams", &Nice::KDACInterface<float>::SetupParams)
    .def("Predict", &Nice::KDACInterface<float>::Predict)
    .def("GetProfiler", &Nice::KDACInterface<float>::GetProfiler)
    .def("GetW", &Nice::KDACInterface<float>::GetW)
    .def("GetD", &Nice::KDACInterface<float>::GetD)
    .def("GetN", &Nice::KDACInterface<float>::GetN)
    .def("GetQ", &Nice::KDACInterface<float>::GetQ)
    .def("GetTimePerIter", &Nice::KDACInterface<float>::GetTimePerIter);

  //Use double by default
  boost::python::class_<Nice::KDACInterface<double>>
      ("KDAC", boost::python::init<>())
      .def("Fit", Fit1Double)
      .def("Fit", Fit2Double)
      .def("Fit", Fit3Double)
      .def("SetupParams", &Nice::KDACInterface<double>::SetupParams)
      .def("Predict", &Nice::KDACInterface<double>::Predict)
      .def("GetProfiler", &Nice::KDACInterface<double>::GetProfiler)
      .def("GetW", &Nice::KDACInterface<double>::GetW)
      .def("GetD", &Nice::KDACInterface<double>::GetD)
      .def("GetN", &Nice::KDACInterface<double>::GetN)
      .def("GetQ", &Nice::KDACInterface<double>::GetQ)
      .def("GetTimePerIter", &Nice::KDACInterface<double>::GetTimePerIter);
}

// Explicit Instantiation
template
class Nice::KDACInterface<float>;
template
class Nice::KDACInterface<double>;