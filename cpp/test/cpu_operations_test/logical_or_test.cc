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

#include <stdio.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

class LogicalOrTest : public ::testing::Test {
 public:
  Nice::Matrix<bool> m1;  // First boolean matrix
  Nice::Matrix<bool> m2;  // Second boolean matrix
  Nice::Matrix<bool> m3;  // Stores LogicalOr return value for matrices
  Nice::Vector<bool> v1;  // First boolean vector
  Nice::Vector<bool> v2;  // Second boolean vector
  Nice::Vector<bool> v3;  // Stores LogicalOr return value for vectors
};
// Tests the basic functionality of LogicalOr for Matrices
TEST_F(LogicalOrTest, LogicalOrMatrixFunctionality) {
  m1.setRandom(4, 4);
  m2.setRandom(4, 4);
  m3 = Nice::CpuOperations<bool>::LogicalOr(m1, m2);
  ASSERT_TRUE(m3.isApprox((m1.array() || m2.array()).matrix() ) );
}

// Tests to see if LogicalOr for matrices throws an error when matrices are of
// different sizes
TEST_F(LogicalOrTest, LogicalOrMatrixDiffSize) {
  m1.setRandom(4, 4);
  m2.setRandom(4, 3);
  ASSERT_DEATH(Nice::CpuOperations<bool>::LogicalOr(m1, m2), ".*");
}

// Logical Or should quit if the parameters aren't initialized
TEST_F(LogicalOrTest, MatrixNoValue) {
  ASSERT_DEATH(Nice::CpuOperations<bool>::LogicalOr(m1, m2), ".*");
}

// Tests the basic functionality of LogicalOr for Vectors
TEST_F(LogicalOrTest, LogicalOrVectorFunctionality) {
  v1.setRandom(4);
  v2.setRandom(4);
  v3 = Nice::CpuOperations<bool>::LogicalOr(v1, v2);
  ASSERT_TRUE(v3.isApprox((v1.array() || v2.array())) );
}

// Tests to see if LogicalOr for vectors throws an error when vectors are of
// different sizes
TEST_F(LogicalOrTest, LogicalOrVectorDiffSize) {
  v1.setRandom(4);
  v2.setRandom(3);
  ASSERT_DEATH(Nice::CpuOperations<bool>::LogicalOr(v1, v2), ".*");
}

// Logical Or should quit if the parameters aren't initialized
TEST_F(LogicalOrTest, VectorNoValue) {
  ASSERT_DEATH(Nice::CpuOperations<bool>::LogicalOr(v1, v2), ".*");
}
