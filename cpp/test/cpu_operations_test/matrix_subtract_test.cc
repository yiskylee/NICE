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

template<class T>
class MatrixSubtractTest : public ::testing::Test {
 public:
  Nice::T scalar;
  Nice::Matrix<T> m1;
  Nice::Matrix<T> m2;
  Nice::Matrix<T> result;
  Nice::Matrix<T> testMatrix;

  void MatrixSubtract() {
    result = Nice::CpuOperations<T>::Subtract(m1, m2);
  }
  void MatrixScalarSubtract() {
    result = Nice::CpuOperations<T>::Subtract(m1, scalar); 
}
};

typedef ::testing::Types<int, double, float> MyTypes;
TYPED_TEST_CASE(MatrixSubtractTest, MyTypes);

TYPED_TEST(MatrixSubtractTest, MatrixSubtractFunctionality) {
  this->m1.resize(2, 2);
  this->m2.resize(2, 2);
  this->testMatrix.resize(2, 2);
  this->m1 << 2, 3,
              4, 5;
  this->m2 << 1, 2,
              3, 2;
  this->MatrixSubtract();
  this->testMatrix << 1, 1,
                      1, 3;
  ASSERT_TRUE(this->result.isApprox(this->testMatrix));
}

TYPED_TEST(MatrixScalarSubtractTest, MatrixScalarSubtractFunctionality) {
  this->m1.resize(2, 2);
  this->scalar == 2;
  this->testMatrix.resize(2, 2);
  this->m1 << 4, 5,
              3, 6;
  this->MatrixScalarSubtract();
  this->testMatrix << 2, 3,
                      1, 4;
  ASSERT_TRUE(this->result.isApprox(this->testMatrix));
}

TYPED_TEST(MatrixSubtractTest, DifferentSizeMatrix) {
  this->m1.resize(2, 2);
  this->m2.resize(3, 2);
  this->m1.setZero();
  this->m2.setZero();
  ASSERT_DEATH(this->MatrixSubtract(), ".*");
}

TYPED_TEST(MatrixSubtractTest, EmptyMatrix) {
  ASSERT_DEATH(this->MatrixSubtract(), ".*");
}

TYPED_TEST(MatrixScalarSubtractTest, EmptyMatrix) {
  ASSERT_DEATH(this->MatrixScalarSubtract(), ".");
}
