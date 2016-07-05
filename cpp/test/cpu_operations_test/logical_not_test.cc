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

// This file tests CpuOperations::LogicalNot() to see if it will correctly
// handle basic boolean Matrices and Vectors and to see if it will throw
// an error if an uninitialized variable gets passed into its parameters

#include <stdio.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

class LogicalNotTest :public ::testing::Test {
 public:
  Nice::Matrix<bool> ma;  // Matrix for method input
  Nice::Matrix<bool> mb;  // Expected Matrix output
  Nice::Vector<bool> va;  // Vector for method input
  Nice::Vector<bool> vb;  // Vector for method output
};
// A general test to see if LogicalNot works on Matrices
TEST_F(LogicalNotTest, LogicalNotMatrix) {
  ma.resize(4, 4);
  mb.resize(4, 4);
  ma << 1, 1, 1, 1,
        1, 1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0;
  mb << 0, 0, 0, 0,
        0, 0, 0, 0,
        1, 1, 1, 1,
        1, 1, 1, 1;
  ASSERT_TRUE(mb.isApprox(Nice::CpuOperations<bool>::LogicalNot(ma)));
}

// A general test to see if LogicalNot works on Vectors
TEST_F(LogicalNotTest, LogicalNotVector) {
  va.resize(4);
  vb.resize(4);
  va << 1, 0, 1, 0;
  vb << 0, 1, 0, 1;
  ASSERT_TRUE(vb.isApprox(Nice::CpuOperations<bool>::LogicalNot(va)));
}
// Test to see if LogicalNot for Matrices will throw an exception
TEST_F(LogicalNotTest, MatrixNoValue) {
  ASSERT_DEATH(Nice::CpuOperations<bool>::LogicalNot(ma), ".*");
}
// Test to see if LogicalNot for Vectors will throw an exception
TEST_F(LogicalNotTest, VectorNoValue) {
  ASSERT_DEATH(Nice::CpuOperations<bool>::LogicalNot(va), ".*");
}
