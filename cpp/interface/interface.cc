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
#include "clustering/acl_interface.h"
#include "cpu_operations_interface.h"


void (Nice::ACLInterface<float>::*Fit0Arg)()
= &Nice::ACLInterface<float>::Fit;
void (Nice::ACLInterface<float>::*Fit1Arg)(PyObject *, int, int)
= &Nice::ACLInterface<float>::Fit;
void (Nice::ACLInterface<float>::*Fit2Arg)(PyObject *, int, int,
                                            PyObject *, int, int)
    = &Nice::ACLInterface<float>::Fit;

void (Nice::ACLInterface<double>::*Fit0ArgDouble)()
= &Nice::ACLInterface<double>::Fit;
void (Nice::ACLInterface<double>::*Fit1ArgDouble)(PyObject *, int, int)
= &Nice::ACLInterface<double>::Fit;
void (Nice::ACLInterface<double>::*Fit2ArgDouble)(PyObject *, int, int,
                                            PyObject *, int, int)
= &Nice::ACLInterface<double>::Fit;


BOOST_PYTHON_MODULE(Nice4Py) {
    boost::python::class_<Nice::ACLInterface<float>>
        ("ACL", boost::python::init<std::string, std::string>())
        .def("Fit", Fit0Arg)
        .def("Fit", Fit1Arg)
        .def("Fit", Fit2Arg)
        .def("SetupParams", &Nice::ACLInterface<float>::SetupParams)
        .def("Predict", &Nice::ACLInterface<float>::Predict)
        .def("GetProfiler", &Nice::ACLInterface<float>::GetProfiler)
        .def("GetTimePerIter", &Nice::ACLInterface<float>::GetTimePerIter)
        .def("GetQ", &Nice::ACLInterface<float>::GetQ)
        .def("GetD", &Nice::ACLInterface<float>::GetD)
        .def("GetN", &Nice::ACLInterface<float>::GetN)
        .def("GetK", &Nice::ACLInterface<float>::GetK)
        .def("GetW", &Nice::ACLInterface<float>::GetW)
        .def("GetU", &Nice::ACLInterface<float>::GetU)
        .def("SetW", &Nice::ACLInterface<float>::SetW)
        .def("DiscardLastRun", &Nice::ACLInterface<float>::DiscardLastRun);

    boost::python::class_<Nice::CPUOperationsInterface<float>>("CPUOp")
        .def("GenKernelMatrix",
             &Nice::CPUOperationsInterface<float>::GenKernelMatrix)
        .staticmethod("GenKernelMatrix");


    boost::python::class_<Nice::ACLInterface<double>>
        ("ACLDouble", boost::python::init<std::string, std::string>())
        .def("Fit", Fit0ArgDouble)
        .def("Fit", Fit1ArgDouble)
        .def("Fit", Fit2ArgDouble)
        .def("SetupParams", &Nice::ACLInterface<double>::SetupParams)
        .def("Predict", &Nice::ACLInterface<double>::Predict)
        .def("GetProfiler", &Nice::ACLInterface<double>::GetProfiler)
        .def("GetTimePerIter", &Nice::ACLInterface<double>::GetTimePerIter)
        .def("GetQ", &Nice::ACLInterface<double>::GetQ)
        .def("GetD", &Nice::ACLInterface<double>::GetD)
        .def("GetN", &Nice::ACLInterface<double>::GetN)
        .def("GetK", &Nice::ACLInterface<double>::GetK)
        .def("GetW", &Nice::ACLInterface<double>::GetW)
        .def("GetU", &Nice::ACLInterface<double>::GetU)
        .def("SetW", &Nice::ACLInterface<double>::SetW)
        .def("DiscardLastRun", &Nice::ACLInterface<double>::DiscardLastRun);

    boost::python::class_<Nice::CPUOperationsInterface<double>>("CPUOp")
        .def("GenKernelMatrix",
        &Nice::CPUOperationsInterface<double>::GenKernelMatrix)
        .staticmethod("GenKernelMatrix");
}