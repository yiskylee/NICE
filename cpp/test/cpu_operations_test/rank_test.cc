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


template<class T>
class RankTest : public ::testing::Test {
 public:
  Nice::Matrix<T> mat_;
  int calculated_ans_;
};

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(RankTest, MyTypes);

TYPED_TEST(RankTest, RankMatrix) {
  this->mat_.resize(4, 4);
  this->mat_ <<  1.0, 3.0, 5.0, 2.0,
                 0.0, 1.0, 0.0, 3.0,
                 0.0, 0.0, 0.0, 1.0,
                 0.0, 0.0, 0.0, 0.0;

  int correct_ans = 3;

  this->calculated_ans_ = Nice::CpuOperations<TypeParam>::Rank(this->mat_);
  EXPECT_EQ(correct_ans, this->calculated_ans_);
}

