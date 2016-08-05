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
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

template<typename T>
class MatrixCenterTest : public ::testing::Test {
 public:
  Nice::Matrix<T> a;
  Nice::Matrix<T> correct_ans;
  Nice::Matrix<T> answer;
  float precision = .0001;

  void MatrixCenter(int row) {
    answer = Nice::CpuOperations<T>::Center(a, row);
  }
};

typedef ::testing::Types<float, double> FloatTypes;
TYPED_TEST_CASE(MatrixCenterTest, FloatTypes);

TYPED_TEST(MatrixCenterTest, MatrixCenterCol) {
  this->a.resize(3, 3);
  this->a << 1, 4, 7,
             2, 5, 8,
             3, 6, 9;
  this->correct_ans.resize(3, 3);
  this->correct_ans << -1, -1, -1,
                       0, 0, 0,
                       1, 1, 1;
  this->MatrixCenter(0);
  ASSERT_TRUE(this->correct_ans.isApprox(this->answer, this->precision));
}

TYPED_TEST(MatrixCenterTest, MatrixCenterRow) {
  this->a.resize(3, 3);
  this->a << 1, 2, 3,
             4, 5, 6,
             7, 8, 9;
  this->correct_ans.resize(3, 3);
  this->correct_ans << -1, 0, 1,
                       -1, 0, 1,
                       -1, 0, 1;
  this->MatrixCenter(1);
  ASSERT_TRUE(this->correct_ans.isApprox(this->answer, this->precision));
}

TYPED_TEST(MatrixCenterTest, BadAxis) {
  this->a.resize(3, 3);
  this->a << 1, 2, 3,
             4, 5, 6,
             7, 8, 9;
  ASSERT_DEATH(this->MatrixCenter(2), ".*");
}
