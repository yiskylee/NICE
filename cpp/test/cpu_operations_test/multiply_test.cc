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
#include <stdlib.h>
#include <iostream>
#include "include/matrix.h"
//#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"


template<typename T>
class MatrixMultiplyTest : public ::testing::Test {
 public:
  T scalar;
  Nice::Matrix<T> a;
  Nice::Matrix<T> b;
  Nice::Matrix<T> correct_ans;
  Nice::Matrix<T> answer;

  void MatrixMatrixMultiplier() {
    answer = Nice::CpuOperations<T>::Multiply(a, b);
  }

  void MatrixScalarMultiplier() {
    answer = Nice::CpuOperations<T>::Multiply(a, scalar);
  }
};

typedef ::testing::Types<int, float, double> MyTypes;
TYPED_TEST_CASE(MatrixMultiplyTest, MyTypes);

TYPED_TEST(MatrixMultiplyTest, MatrixScalarMultiply) {
  this->scalar = 3;
  this->a.resize(3, 3);
  this->a << 0, 1, 2,
             3, 2, 1,
             1, 3, 0;
  this->correct_ans.resize(3, 3);
  this->correct_ans << 0, 3, 6,
                       9, 6, 3,
                       3, 9, 0;
  this->MatrixScalarMultiplier();
  ASSERT_TRUE(this->correct_ans.isApprox(this->answer));
}

TYPED_TEST(MatrixMultiplyTest, MatrixMatrixMultiply) {
  this->a.resize(3, 3);
  this->a << 0, 1, 2,
             3, 2, 1,
             1, 3, 0;
  this->b.resize(3, 3);
  this->b << 1, 0, 2,
             2, 1, 0,
             0, 2, 1;
  this->correct_ans.resize(3, 3);
  this->correct_ans << 2, 5, 2,
                       7, 4, 7,
                       7, 3, 2;
  this->MatrixMatrixMultiplier();
  ASSERT_TRUE(this->correct_ans.isApprox(this->answer));
}

TYPED_TEST(MatrixMultiplyTest, MatrixMatrixTransposeMultiply) {
  this->a.resize(2, 2);
  this->a << 1, 2,
             3, 4;
  this->a = this->a * this->a.transpose();
  this->correct_ans.resize(2, 2);
  this->correct_ans << 5, 11,
                       11, 25;
  EXPECT_TRUE(this->a.isApprox(this->correct_ans));
}

TYPED_TEST(MatrixMultiplyTest, BlasTest) {
  Nice::Matrix<TypeParam> m1(1000, 1000);
  Nice::Matrix<TypeParam> m2(1000, 1000);
  Nice::Matrix<TypeParam> m3 = m1 * m2;
}

