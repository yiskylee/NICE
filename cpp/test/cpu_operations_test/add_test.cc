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
// see if two matrices passed are added, and to see if a matrix and
// scalar is added correctly. This also checks that the test fails if there is
// an empty matrix or if the matricies are not the same size.

#include <stdio.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

template<class T>
class AddTest : public ::testing::Test {
 public:
  Nice::Matrix<T> matrix_a;
  Nice::Matrix<T> matrix_b;
  T scalar;
  Nice::Matrix<T> result_matrix;
  Nice::Matrix<T> result_scalar;
  Nice::Matrix<T> correct;

  void Add_Matrix() {
    result_matrix = Nice::CpuOperations<T>::Add(matrix_a, matrix_b);
  }
  void Add_Scalar() {
    result_scalar = Nice::CpuOperations<T>::Add(matrix_a, scalar);
  }
};

typedef ::testing::Types<int, float, double> MyTypes;
TYPED_TEST_CASE(AddTest, MyTypes);

TYPED_TEST(AddTest, AddFunctionality) {
  // Matrix + Matrix
  this->matrix_a.setRandom(3, 3);
  this->matrix_b.setRandom(3, 3);
  this->Add_Matrix();

  this->correct.setZero(3, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      this->correct(i, j) = (this->matrix_a(i, j) + this->matrix_b(i, j));
    }
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(this->result_matrix(i, j), this->correct(i, j), 0.0001);
    }
  }

  // Matrix + Scalar
  this->matrix_a.setRandom(3, 3);
  this->scalar = 12;
  this->Add_Scalar();

  this->correct.setZero(3, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      this->correct(i, j) = (this->matrix_a(i, j) + this->scalar);
    }
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(this->result_scalar(i, j), this->correct(i, j), 0.0001);
    }
  }
}

TYPED_TEST(AddTest, DifferentSizes) {
  this->matrix_a.setRandom(2, 4);
  this->matrix_b.setRandom(3, 1);
  ASSERT_DEATH(this->Add_Matrix(), ".*");
}

TYPED_TEST(AddTest, EmptyMatrix) {
  ASSERT_DEATH(this->Add_Matrix(), ".*");
  ASSERT_DEATH(this->Add_Scalar(), ".*");
}
