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

#ifndef CPP_INTERFACE_CPU_OPERATIONS_INTERFACE_H_
#define CPP_INTERFACE_CPU_OPERATIONS_INTERFACE_H_

#include <boost/python.hpp>
#include <iostream>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "include/cpu_operations.h"


namespace Nice {

template<typename T>
using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor> >;
template<typename T>
class CPUOperationInterface {
 public:
  static void GenKernelMatrix(PyObject *input_obj, int row, int col,
                              std::string kernel_type,
                              float constant,
                              PyObject *kernel_matrix_obj) {
    Py_buffer input_buf, kernel_matrix_buf;
    PyObject_GetBuffer(input_obj, &input_buf, PyBUF_SIMPLE);
    PyObject_GetBuffer(kernel_matrix_obj, &kernel_matrix_buf, PyBUF_SIMPLE);
    MatrixMap <T> input(reinterpret_cast<T *>(input_buf.buf), row, col);
    MatrixMap <T> kernel_matrix(reinterpret_cast<T *>
                                (kernel_matrix_buf.buf), row, row);
    if (kernel_type == "gaussian") {
      kernel_matrix =
          CpuOperations::GenKernelMatrix(input,
                                         Nice::kGaussianKernel,
                                         constant);
      PyBuffer_Release(&input_buf);
      PyBuffer_Release(&kernel_matrix_buf);
    } else {
      std::cout << kernel_type << " not yet supported\n";
      exit(1);
    }
  }
};
} // namespace Nice
#endif  // CPP_INTERFACE_CPU_OPERATIONS_INTERFACE_H_