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
#include <cmath>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

template<class T>
class NormalizeTest : public ::testing::Test {
 public:
  Nice::Matrix<T> matrix_;
  Nice::Matrix<T> normalized_matrix_;
  Nice::Matrix<T> correct_matrix_;
};

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(NormalizeTest, MyTypes);

TYPED_TEST(NormalizeTest, WhenAxisis0) {
  int axis = 0;
  int p = 2;
  this->matrix_.resize(2, 3);
  this->matrix_ << 1.0, 2.0, 3.0,
                   4.0, 5.0, 6.0;
  this->correct_matrix_.resize(2, 3);
  this->correct_matrix_<< 1.0/sqrt(17), 2.0/sqrt(29), 3.0/sqrt(45),
                             4.0/sqrt(17), 5.0/sqrt(29), 6.0/sqrt(45);
  this->normalized_matrix_ = Nice::CpuOperations<TypeParam>::Normalize(
                                                          this->matrix_,
                                                          p,
                                                          axis);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
     ASSERT_NEAR(this->normalized_matrix_(i, j), this->correct_matrix_(i, j),
                                               0.001);
    }
  }
}

TYPED_TEST(NormalizeTest, WhenAxisis1) {
  int axis = 1;
  int p = 2;
  this->matrix_.resize(2, 3);
  this->matrix_ << 1.0, 2.0, 3.0,
                   4.0, 5.0, 6.0;
  this->correct_matrix_.resize(2, 3);
  this->correct_matrix_<< 1.0/sqrt(14), 2.0/sqrt(14), 3.0/sqrt(14),
                          4.0/sqrt(77), 5.0/sqrt(77), 6.0/sqrt(77);
  this->normalized_matrix_ = Nice::CpuOperations<TypeParam>::Normalize(
                                                          this->matrix_,
                                                          p,
                                                          axis);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
     ASSERT_NEAR(this->normalized_matrix_(i, j), this->correct_matrix_(i, j),
                                               0.001);
    }
  }
}


