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

// This file tests the CpuOperations::Add() function by checking to
// See if two matrixes passed are added in the test IsAdded
// A summed Nice matrix is compared to a summed Eigen Matrix in
// Add Eigen
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
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_a_;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_b_;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> added_eigen_;
  Nice::Matrix<T> added_nice_;

  // Add the Eigen and Nice matrices
  void Adder() {
    added_eigen_ = matrix_a_ + matrix_b_;  // Add Eigen
    // Add  matrixNice
    added_nice_ = Nice::CpuOperations<T>::Add(matrix_a_, matrix_b_);
  }
};

// Establishes a test case with the given types, Char and short types will
// Throw compiler errors
typedef ::testing::Types<int, float, double> MyTypes;
TYPED_TEST_CASE(MyTest, MyTypes);

// Checks to see if each element is transposed using the Transpose() function
TYPED_TEST(MyTest, IsAdded) {
  // this-> refers to the test fixture object
  this->matrix_a_.setRandom(3, 3);  // Assign random values
  this->matrix_b_.setRandom(3, 3);  // Assign random values
  this->Adder();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // Check equality for each element
      EXPECT_EQ(this->added_eigen_(i, j), this->added_nice_(i, j));
    }
  }
}
