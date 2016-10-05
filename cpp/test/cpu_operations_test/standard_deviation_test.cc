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
class MatrixStandardDeviationTest : public ::testing::Test {
 public:
  Nice::Matrix<T> a;
  Nice::Vector<T> correct_ans;
  Nice::Vector<T> answer;
  float precision = .0001;

  void MatrixStandardDeviation(int row) {
    answer = Nice::CpuOperations<T>::StandardDeviation(a, row);
  }
};

typedef ::testing::Types<float, double> FloatTypes;
TYPED_TEST_CASE(MatrixStandardDeviationTest, FloatTypes);

TYPED_TEST(MatrixStandardDeviationTest, MatrixTrue) {
  ASSERT_TRUE(true);
}

TYPED_TEST(MatrixStandardDeviationTest, MatrixStandardDeviationCol) {
  this->a.resize(3, 2);
  this->a << 1, 4,
             2, 5,
             3, 6;
  this->correct_ans.resize(2);
  this->correct_ans << .8165, .8165;
  this->MatrixStandardDeviation(0);
  ASSERT_TRUE(this->correct_ans.isApprox(this->answer, this->precision));
}

TYPED_TEST(MatrixStandardDeviationTest, MatrixStandardDeviationRow) {
  this->a.resize(2, 3);
  this->a << 1, 2, 3,
             4, 5, 6;
  this->correct_ans.resize(2);
  this->correct_ans << .8165, .8165;
  this->MatrixStandardDeviation(1);
  ASSERT_TRUE(this->correct_ans.isApprox(this->answer, this->precision));
}

TYPED_TEST(MatrixStandardDeviationTest, BadAxis) {
  this->a.resize(2, 3);
  this->a << 1, 2, 3,
             4, 5, 6;
  ASSERT_DEATH(this->MatrixStandardDeviation(2), ".*");
}
