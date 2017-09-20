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
#include "cpu_operations_interface.h"


void (Nice::KDACInterface<float>::*Fit0Arg)()
= &Nice::KDACInterface<float>::Fit;
void (Nice::KDACInterface<float>::*Fit1Arg)(PyObject *, int, int)
= &Nice::KDACInterface<float>::Fit;
void (Nice::KDACInterface<float>::*Fit2Arg)(PyObject *, int, int,
                                            PyObject *, int, int)
    = &Nice::KDACInterface<float>::Fit;

void (Nice::KDACInterface<double>::*Fit0ArgDouble)()
= &Nice::KDACInterface<double>::Fit;
void (Nice::KDACInterface<double>::*Fit1ArgDouble)(PyObject *, int, int)
= &Nice::KDACInterface<double>::Fit;
void (Nice::KDACInterface<double>::*Fit2ArgDouble)(PyObject *, int, int,
                                            PyObject *, int, int)
= &Nice::KDACInterface<double>::Fit;


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


    boost::python::class_<Nice::KDACInterface<double>>
        ("KDACDouble", boost::python::init<std::string>())
        .def("Fit", Fit0ArgDouble)
        .def("Fit", Fit1ArgDouble)
        .def("Fit", Fit2ArgDouble)
        .def("SetupParams", &Nice::KDACInterface<double>::SetupParams)
        .def("Predict", &Nice::KDACInterface<double>::Predict)
        .def("GetProfiler", &Nice::KDACInterface<double>::GetProfiler)
        .def("GetTimePerIter", &Nice::KDACInterface<double>::GetTimePerIter)
        .def("GetQ", &Nice::KDACInterface<double>::GetQ)
        .def("GetD", &Nice::KDACInterface<double>::GetD)
        .def("GetN", &Nice::KDACInterface<double>::GetN)
        .def("GetK", &Nice::KDACInterface<double>::GetK)
        .def("GetW", &Nice::KDACInterface<double>::GetW)
        .def("GetU", &Nice::KDACInterface<double>::GetU)
        .def("SetW", &Nice::KDACInterface<double>::SetW)
        .def("DiscardLastRun", &Nice::KDACInterface<double>::DiscardLastRun);
        boost::python::class_<Nice::CPUOperationsInterface<double>>("CPUOp")
        .def("GenKernelMatrix",
        &Nice::CPUOperationsInterface<double>::GenKernelMatrix)
        .staticmethod("GenKernelMatrix");
}