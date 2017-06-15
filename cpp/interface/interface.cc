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
#include "include/timer.h"
#include "clustering/kdac_interface.h"
#include "clustering/kmeans_interface.h"
#include "interface/cpu_operations_interface.h"

// void (Nice::KDACCPUInterface<float>::*Fit1Float)
// (PyObject *, int row, int col)
//  = &Nice::KDACCPUInterface<float>::Fit;
// void (Nice::KDACCPUInterface<float>::*Fit2Float)()
//  = &Nice::KDACCPUInterface<float>::Fit;
// void (Nice::KDACCPUInterface<float>::*Fit3Float)
// (PyObject *, int row_1, int col_1,
//                                  PyObject *, int row_2, int col_2)
//  = &Nice::KDACCPUInterface<float>::Fit;
//
// void (Nice::KDACGPUInterface<float>::*GPUFit1Float)
// (PyObject *, int row, int col)
// = &Nice::KDACGPUInterface<float>::Fit;
// void (Nice::KDACGPUInterface<float>::*GPUFit2Float)()
// = &Nice::KDACGPUInterface<float>::Fit;
// void (Nice::KDACGPUInterface<float>::*GPUFit3Float)
// (PyObject *, int row_1, int col_1,
//
// PyObject *, int row_2, int col_2)
// = &Nice::KDACGPUInterface<float>::Fit;
//
//
// void (Nice::KDACCPUInterface<double>::*Fit1Double)
// (PyObject *, int row, int col)
// = &Nice::KDACCPUInterface<double>::Fit;
// void (Nice::KDACCPUInterface<double>::*Fit2Double)()
// = &Nice::KDACCPUInterface<double>::Fit;
// void (Nice::KDACCPUInterface<double>::*Fit3Double)
// (PyObject *, int row_1, int col_1,
//                                         PyObject *, int row_2, int col_2)
// = &Nice::KDACCPUInterface<double>::Fit;

void (Nice::KDACInterface<float>::*Fit0Arg)()
= &Nice::KDACInterface<float>::Fit;
void (Nice::KDACInterface<float>::*Fit1Arg)(PyObject *, int, int)
= &Nice::KDACInterface<float>::Fit;
void (Nice::KDACInterface<float>::*Fit2Arg)(PyObject *, int, int,
                                            PyObject *, int, int)
    = &Nice::KDACInterface<float>::Fit;


BOOST_PYTHON_MODULE(Nice4Py) {
    boost::python::class_<Nice::KDACInterface<float>>
        ("KDAC", boost::python::init<std::string>())
        .def("Fit", Fit0Arg)
        .def("Fit", Fit1Arg)
        .def("Fit", Fit2Arg)
        .def("SetupParams", &Nice::KDACInterface<float>::SetupParams)
        .def("Predict", &Nice::KDACInterface<float>::Predict)
        .def("GetProfiler", &Nice::KDACInterface<float>::GetProfiler)
        .def("GetTimePerIter", &Nice::KDACInterface<float>::GetTimePerIter)
        .def("GetQ", &Nice::KDACInterface<float>::GetQ)
        .def("GetD", &Nice::KDACInterface<float>::GetD)
        .def("GetN", &Nice::KDACInterface<float>::GetN)
        .def("GetK", &Nice::KDACInterface<float>::GetK)
        .def("GetW", &Nice::KDACInterface<float>::GetW)
        .def("GetU", &Nice::KDACInterface<float>::GetU)
        .def("SetW", &Nice::KDACInterface<float>::SetW)
        .def("DiscardLastRun", &Nice::KDACInterface<float>::DiscardLastRun);
    boost::python::class_<Nice::CPUOperationsInterface<float>>("CPUOp")
        .def("GenKernelMatrix",
             &Nice::CPUOperationsInterface<float>::GenKernelMatrix)
        .staticmethod("GenKernelMatrix");

    boost::python::class_<Nice::KmeansInterface<float>>
    ("KMeans", boost::python::init<std::string>())
    .def("Fit", &Nice::KmeansInterface<float>::Fit)
    .def("GetLabels", &Nice::KmeansInterface<float>::GetLabels);

//  boost::python::class_<Nice::CpuOperationsInterface<float>::GenKernelMatrix
}
//  //Use float by default
//  boost::python::class_<Nice::KDACCPUInterface<float>>
//      ("KDACCPU", boost::python::init<>())
//    .def("Fit", Fit1Float)
//    .def("Fit", Fit2Float)
//    .def("Fit", Fit3Float)
//    .def("SetupParams", &Nice::KDACCPUInterface<float>::SetupParams)
//    .def("Predict", &Nice::KDACCPUInterface<float>::Predict)
//    .def("GetProfiler", &Nice::KDACCPUInterface<float>::GetProfiler)
//    .def("GetW", &Nice::KDACCPUInterface<float>::GetW)
//    .def("GetD", &Nice::KDACCPUInterface<float>::GetD)
//    .def("GetN", &Nice::KDACCPUInterface<float>::GetN)
//    .def("GetQ", &Nice::KDACCPUInterface<float>::GetQ)
//    .def("GetTimePerIter", &Nice::KDACCPUInterface<float>::GetTimePerIter);
//
//    //Use float by default
//    boost::python::class_<Nice::KDACGPUInterface<float>>
//    ("KDACGPU", boost::python::init<>())
//    .def("Fit", GPUFit1Float)
//    .def("Fit", GPUFit2Float)
//    .def("Fit", GPUFit3Float)
//    .def("SetupParams", &Nice::KDACGPUInterface<float>::SetupParams)
//    .def("Predict", &Nice::KDACGPUInterface<float>::Predict)
//    .def("GetProfiler", &Nice::KDACGPUInterface<float>::GetProfiler)
//    .def("GetW", &Nice::KDACGPUInterface<float>::GetW)
//    .def("GetD", &Nice::KDACGPUInterface<float>::GetD)
//    .def("GetN", &Nice::KDACGPUInterface<float>::GetN)
//    .def("GetQ", &Nice::KDACGPUInterface<float>::GetQ)
//    .def("GetTimePerIter", &Nice::KDACGPUInterface<float>::GetTimePerIter);
//
//
//  boost::python::class_<Nice::KDACCPUInterface<double>>
//      ("KDACDOUBLE", boost::python::init<>())
//      .def("Fit", Fit1Double)
//      .def("Fit", Fit2Double)
//      .def("Fit", Fit3Double)
//      .def("SetupParams", &Nice::KDACCPUInterface<double>::SetupParams)
//      .def("Predict", &Nice::KDACCPUInterface<double>::Predict)
//      .def("GetProfiler", &Nice::KDACCPUInterface<double>::GetProfiler)
//      .def("GetW", &Nice::KDACCPUInterface<double>::GetW)
//      .def("GetD", &Nice::KDACCPUInterface<double>::GetD)
//      .def("GetN", &Nice::KDACCPUInterface<double>::GetN)
//      .def("GetQ", &Nice::KDACCPUInterface<double>::GetQ)
//      .def("GetTimePerIter", &Nice::KDACCPUInterface<double>::GetTimePerIter);
//}

// Explicit Instantiation
// template
// class Nice::KDACInterface<float>;
// template
// class Nice::KDACInterface<double>;
// template
// class Nice::KDACCPUInterface<float>;
// template
// class Nice::KDACCPUInterface<double>;
// template
// class Nice::KDACGPUInterface<float>;
// template
// class Nice::KDACGPUInterface<double>;
