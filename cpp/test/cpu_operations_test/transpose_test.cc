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

// This file tests the CpuOperations::Transpose() function by checking to
// See if a matrix passed is transposed in the test IsTransposed
// A transposed Nice matrix is compared to a transposed Eigen Matrix in
// Transpose Eigen
// Behavior with oddly shaped matrices is also tested with test DifferentShapes
// And TransposeZeroRows
// All tests are made using a templated test fixture which attempts
// Integer, float, and double data types

#include <stdio.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

// This is a template test fixture class containing Eigen and NICE matrices
template<class T>  // Template
class MyTest : public ::testing::Test {  // Inherits from testing::Test
 public:  // Members must be public to be accessed by tests
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _matrix_eigen;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _transpose_eigen;
  Nice::Matrix<T> _matrix_nice;
  Nice::Matrix<T> _transpose_nice;

  // Transposes the Eigen and Nice matrices
  void Transposer() {
    _transpose_eigen = _matrix_eigen.transpose();  // Transpose Eigen
    // Transpose matrixNice
    _transpose_nice = Nice::CpuOperations<T>::Transpose(_matrix_nice);
  }
};

// Establishes a test case with the given types, Char and short types will
// Throw compiler errors
typedef ::testing::Types<int, float, double> MyTypes;
TYPED_TEST_CASE(MyTest, MyTypes);

// Checks to see if each element is transposed using the Transpose() function
TYPED_TEST(MyTest, IsTransposed) {
  // this-> refers to the test fixture object
  this->_matrix_nice.setRandom(3, 3);  // Assign random values
  this->Transposer();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // Check equality for each element
      EXPECT_EQ(this->_matrix_nice(i, j), this->_transpose_nice(j, i));
    }
  }
}

// Transposes a matrix instantiated with random values and compares
// Each element of the Eigen and Nice matrices
TYPED_TEST(MyTest, TransposeEigen) {
  this->_matrix_nice.setRandom(3, 3);  // Random values
  this->_matrix_eigen = this->_matrix_nice;  // Set _matrix_eigen=_matrix_Nice
  this->Transposer();  // Transpose matrices
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // Check equality for each element
      EXPECT_EQ(this->_transpose_nice(i, j), this->_transpose_eigen(i, j));
    }
  }
}

// This function uses a non-square matrix size and compares the number of rows
// And Columns after the transposition
TYPED_TEST(MyTest, DifferentShapes) {
  this->_matrix_nice.setRandom(1, 4);
  this->_matrix_eigen = this->_matrix_nice;
  this->Transposer();
  // .rows() and .cols() return the number of rows and columns respectively
  // Check equality
  EXPECT_EQ(this->_transpose_nice.rows(), this->_transpose_eigen.rows());
  EXPECT_EQ(this->_transpose_nice.cols(), this->_transpose_eigen.cols());
}

// This function uses a matrix with size (0,2) and compares the number of rows
// And Columns after the transposition
TYPED_TEST(MyTest, TransposeZeroRows) {
  this->_matrix_nice.setRandom(0, 2);
  this->_matrix_eigen = this->_matrix_nice;
  this->Transposer();
  EXPECT_EQ(this->_transpose_nice.rows(), this->_transpose_eigen.rows());
  EXPECT_EQ(this->_transpose_nice.cols(), this->_transpose_eigen.cols());
  // It should be noted that the creation of matrix with no rows or columns is
  // Legal, and could potentially disrupt future operations
}

