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
#include <string>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "include/cpu_operations.h"
#include "include/kernel_types.h"

namespace Nice {

template<typename T>
using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor> >;
template<typename T>
class CPUOperationsInterface {
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
    if (kernel_type == "Gaussian") {
      kernel_matrix =
          CpuOperations<T>::GenKernelMatrix(input,
                                         kGaussianKernel,
                                         constant);
      PyBuffer_Release(&input_buf);
      PyBuffer_Release(&kernel_matrix_buf);
    } else {
      std::cout << kernel_type << " not yet supported\n";
      exit(1);
    }
  }
  
  void MultiplyMatrix(PyObject *a_obj, int row_a, int col_a,
                      	     PyObject *b_obj, int row_b, int col_b,
		             PyObject *c_obj) { 
    Py_buffer a_buf, b_buf, c_buf;
    PyObject_GetBuffer(a_obj, &a_buf, PyBUF_SIMPLE); 
    PyObject_GetBuffer(b_obj, &b_buf, PyBUF_SIMPLE);
    PyObject_GetBuffer(c_obj, &c_buf, PyBUF_SIMPLE);
    MatrixMap <T> a(reinterpret_cast<T *>(a_buf.buf), row_a, col_a);
    MatrixMap <T> b(reinterpret_cast<T *>(b_buf.buf), row_b, col_b);  
    MatrixMap <T> c(reinterpret_cast<T *>(c_buf.buf), row_a, col_b);
    c = CpuOperations<T>::Multiply(a, b); 
    PyBuffer_Release(&a_buf);
    PyBuffer_Release(&b_buf); 
    PyBuffer_Release(&c_buf);
  }
  void MultiplyMatrix(PyObject *a_obj, int row_a, int col_a,
                             PyObject *b_obj, 
			     float scalar) { 
    Py_buffer a_buf, b_buf;
    PyObject_GetBuffer(a_obj, &a_buf, PyBUF_SIMPLE);
    PyObject_GetBuffer(b_obj, &b_buf, PyBUF_SIMPLE);
    MatrixMap <T> a(reinterpret_cast<T *>(a_buf.buf), row_a, col_a);
    MatrixMap <T> b(reinterpret_cast<T *>(b_buf.buf), row_a, col_a); 
    b = CpuOperations<T>::Multiply(a, scalar); 
    PyBuffer_Release(&a_buf);
    PyBuffer_Release(&b_buf); 
  }
  void InverseMatrix(PyObject *a_obj, int row_a, int col_a,
			    PyObject *b_obj){
    Py_buffer a_buf, b_buf;
    PyObject_GetBuffer(a_obj, &a_buf, PyBUF_SIMPLE);
    PyObject_GetBuffer(b_obj, &b_buf, PyBUF_SIMPLE);
    MatrixMap <T> a(reinterpret_cast<T *>(a_buf.buf), row_a, col_a);
    MatrixMap <T> b(reinterpret_cast<T *>(b_buf.buf), row_a, col_a);
    b = CpuOperations<T>::Inverse(a);
    PyBuffer_Release(&a_buf);
    PyBuffer_Release(&b_buf);
  }
  void NormMatrix(PyObject *m_obj, int row_m, int col_m,
		  int p, int axis,
	          PyObject *v_obj){
    Py_buffer m_buf, v_buf;
    PyObject_GetBuffer(m_obj, &m_buf, PyBUF_SIMPLE);
    PyObject_GetBuffer(v_obj, &v_buf, PyBUF_SIMPLE);
    MatrixMap <T> m(reinterpret_cast<T *>(m_buf.buf), row_m, col_m);
    int row_v = 0;
    if(axis == 0){
	row_v = col_m;
    }
    else if(axis == 1){
	row_v = row_m;
    }
    MatrixMap <T> v(reinterpret_cast<T *>(v_buf.buf), row_v, 1);
    v = CpuOperations<T>::Norm(m, p, axis);
    PyBuffer_Release(&m_buf);
    PyBuffer_Release(&v_buf);
  }
  void CenterMatrix(PyObject *m_obj, int row_m, int col_m,
		    int axis,
		    PyObject *c_obj){
    Py_buffer m_buf, c_buf;
    PyObject_GetBuffer(m_obj, &m_buf, PyBUF_SIMPLE);
    PyObject_GetBuffer(c_obj, &c_buf, PyBUF_SIMPLE);
    MatrixMap <T> m(reinterpret_cast<T *>(m_buf.buf), row_m, col_m);
    MatrixMap <T> c(reinterpret_cast<T *>(c_buf.buf), row_m, col_m); 
    c = CpuOperations<T>::Center(m, axis);
    PyBuffer_Release(&m_buf);
    PyBuffer_Release(&c_buf);
  }
  void NormalizeMatrix(PyObject *m_obj, int row_m, int col_m,
 		       int p, int axis,
		       PyObject *n_obj){
    Py_buffer m_buf, n_buf;
    PyObject_GetBuffer(m_obj, &m_buf, PyBUF_SIMPLE);
    PyObject_GetBuffer(n_obj, &n_buf, PyBUF_SIMPLE);
    MatrixMap <T> m(reinterpret_cast<T *>(m_buf.buf), row_m, col_m);
    MatrixMap <T> n(reinterpret_cast<T *>(n_buf.buf), row_m, col_m);
    n = CpuOperations<T>::Normalize(m, p, axis);
    PyBuffer_Release(&m_buf);
    PyBuffer_Release(&n_buf);
  }
  void StandardDeviationMatrix(PyObject *m_obj, int row_m, int col_m,
                         int axis,
                         PyObject *s_obj){
    Py_buffer m_buf, s_buf;
    PyObject_GetBuffer(m_obj, &m_buf, PyBUF_SIMPLE);
    PyObject_GetBuffer(s_obj, &s_buf, PyBUF_SIMPLE); //fix axis problemo
    MatrixMap <T> m(reinterpret_cast<T *>(m_buf.buf), row_m, col_m);
    int row_s = 0;
    if(axis == 0){
        row_s = col_m;
    }
    else if(axis == 1){
        row_s = row_m;
    }
    MatrixMap <T> s(reinterpret_cast<T *>(s_buf.buf), row_s, 1);
    s = CpuOperations<T>::StandardDeviation(m, axis);
    PyBuffer_Release(&m_buf);
    PyBuffer_Release(&s_buf);
  }
};
}  // namespace Nice
#endif  // CPP_INTERFACE_CPU_OPERATIONS_INTERFACE_H_
