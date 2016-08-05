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
class NormTest : public ::testing::Test {
 public:
  Nice::Matrix<T> norm_matrix_;
  Nice::Vector<T> calculated_norm_;
};

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(NormTest, MyTypes);

TYPED_TEST(NormTest, SquareMatrix) {
  int p = 2;
  int axis = 0;
  this->norm_matrix_.resize(3, 3);
  this->norm_matrix_ <<1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0,
                       7.0, 8.0, 9.0;
  float correct_norm[3] = {sqrt(66), sqrt(93), sqrt(126)};
  this->calculated_norm_ = Nice::CpuOperations<TypeParam>::Norm(
                                                          this->norm_matrix_,
                                                          p,
                                                          axis);
  for (int i = 0; i < 3; i++)
    ASSERT_NEAR(correct_norm[i], this->calculated_norm_(i), 0.001);
}

TYPED_TEST(NormTest, NonsingularMatrix) {
  int p = 2;
  int axis = 0;
  this->norm_matrix_.resize(3, 4);
  this->norm_matrix_ <<1.0, 2.0, 3.0, 4.0,
                       5.0, 6.0, 7.0, 8.0,
                       9.0, 10.0, 11.0, 12.0;
  float correct_norm[4] = {sqrt(107), sqrt(140), sqrt(179), sqrt(224)};
  this-> calculated_norm_ = Nice::CpuOperations<TypeParam>::Norm(
                                                           this->norm_matrix_,
                                                           p,
                                                           axis);
  for (int i = 0; i < 4; i++) {
  ASSERT_NEAR(correct_norm[i], this->calculated_norm_(i), 0.001);
}
}

TYPED_TEST(NormTest, WhenAxisIsntZero) {
  int p = 2;
  int axis = 1;
  this->norm_matrix_.resize(3, 4);
  this->norm_matrix_ <<1.0, 2.0, 3.0, 4.0,
                       5.0, 6.0, 7.0, 8.0,
                       9.0, 10.0, 11.0, 12.0;
  float correct_norm[3] = {sqrt(30), sqrt(174), sqrt(446)};
  this-> calculated_norm_ = Nice::CpuOperations<TypeParam>::Norm(
                                                           this->norm_matrix_,
                                                           p,
                                                           axis);
  for (int i = 0; i < 3; i++) {
  ASSERT_NEAR(correct_norm[i], this->calculated_norm_(i), 0.001);
}
}

TYPED_TEST(NormTest, WhenPIsntTwo) {
  int p = 3;
  int axis = 1;
  this->norm_matrix_.resize(3, 4);
  this->norm_matrix_ <<1.0, 2.0, 3.0, 4.0,
                       5.0, 6.0, 7.0, 8.0,
                       9.0, 10.0, 11.0, 12.0;
  float correct_norm[3] = {pow(100, (1.0/3)), pow(1196, (1.0/3))
                            , pow(4788, (1.0/3))};
  this-> calculated_norm_ = Nice::CpuOperations<TypeParam>::Norm(
                                                           this->norm_matrix_,
                                                           p,
                                                           axis);
  for (int i = 0; i < 3; i++) {
  ASSERT_NEAR(correct_norm[i], this->calculated_norm_(i), 0.001);
}
}

