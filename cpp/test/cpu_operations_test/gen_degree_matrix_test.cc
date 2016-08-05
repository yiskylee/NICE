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
#include "include/kernel_types.h"

template<class T>
class GenDegreeMatrixTest : public ::testing::Test {
 public:
  Nice::Matrix<T> kernel_matrix_;
};

typedef ::testing::Types<float, double> FloatTypes;
TYPED_TEST_CASE(GenDegreeMatrixTest, FloatTypes);

#define EXPECT_MATRIX_EQ(a, ref)\
    EXPECT_EQ(a.rows(), ref.rows());\
    EXPECT_EQ(a.cols(), ref.cols());\
    for (int i = 0; i < a.rows(); i++)\
      for (int j = 0; j < a.cols(); j++)\
        EXPECT_NEAR(double(a(i, j)), double(ref(i, j)), 0.0001);\

TYPED_TEST(GenDegreeMatrixTest, SimpleTest) {
  this->kernel_matrix_.resize(2, 2);
  this->kernel_matrix_ << 2.0, 2.0,
                          2.0, 2.0;

  Nice::Matrix<TypeParam> degree_matrix;
  Nice::Matrix<TypeParam> degree_matrix_to_the_minus_half;

  Nice::CpuOperations<TypeParam>::GenDegreeMatrix(
      this->kernel_matrix_,
      degree_matrix,
      degree_matrix_to_the_minus_half);

  Nice::Matrix<TypeParam> degree_matrix_ref(2,2);
  degree_matrix_ref << 4.0, 0.0,
                       0.0, 4.0;
  Nice::Matrix<TypeParam> degree_matrix_to_the_minus_half_ref(2,2);
  degree_matrix_to_the_minus_half_ref << 0.5, 0.0,
                                         0.0, 0.5;
  EXPECT_MATRIX_EQ(degree_matrix, degree_matrix_ref);
  EXPECT_MATRIX_EQ(degree_matrix_to_the_minus_half,
                   degree_matrix_to_the_minus_half_ref);
}
