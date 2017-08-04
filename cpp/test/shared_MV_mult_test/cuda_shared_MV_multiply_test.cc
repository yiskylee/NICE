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

// This file tests the CpuOperations::Transpose() function by checking to
// See if a matrix passed is transposed in the test IsTransposed
// A transposed Nice matrix is compared to a transposed Eigen Matrix in
// Transpose Eigen
// Behavior with oddly shaped matrices is also tested with test DifferentShapes
// And TransposeZeroRows
// All tests are made using a templated test fixture which attempts
// Integer, float, and double data types

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <memory>

#include "include/gpu_operations.h"
#include "include/cpu_operations.h"

#include "include/kernel_types.h"
#include "Eigen/SVD"
#include "include/svd_solver.h"
#include "include/util.h"

#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"
#include "include/cuda_shared_MV_multiply.h"

// This is a template test fixture class containing test matrices
template<typename T>  // Template
class CudaSharedMVMultiplyTest : public ::testing::Test {
 public:  // Members must be public to be accessed by tests
  Nice::Matrix<T> a_;
  Nice::Vector<T> b_;
  Nice::Vector<T> c_;

  int row_;
  int col_;

  // Constructor
  void CreateTestData(int m, int n) {
    // Check matrix
    if (a_.rows() != 0 && a_.cols() != 0)
      return;

    // Set up dimension
    row_ = m;
    col_ = n;

    // Create matrix
    a_ = Nice::Matrix<T>::Random(row_, col_);
    b_ = Nice::Vector<T>::Random(col_);
    //std::cout << a_ << std::endl;
    //std::cout << b_ << std::endl;

    Nice::CpuOperations<T> cpu_op;
    // Solve in CPU
    c_ = cpu_op.Multiply(a_, b_);
  }
};
// Establishes a test case with the given types, Char and short types will
// Throw compiler errors
typedef ::testing::Types<float> dataTypes;
TYPED_TEST_CASE(CudaSharedMVMultiplyTest, dataTypes);

TYPED_TEST(CudaSharedMVMultiplyTest, FunctionalityTest) {
  // Create test data
  int m = 284324;
  int n = 423;
  srand(time(NULL));
  this->CreateTestData(m, n);
  Nice::Vector<TypeParam> gpu_c(m);
  // Test gpu matrix matrix multiply in Nice
  Nice::CudaSharedMVMultiply<TypeParam> gpu_op(1);
  gpu_c = gpu_op.Multiply(this->a_, this->b_);
  // Verify the result
  for (int i = 0; i < m; i++) {
    EXPECT_NEAR(this->c_(i), gpu_c(i), 0.001);
  }
}
/**
TYPED_TEST(CudaMatrixVectorMultiplyTest, SizeTest) {
  // Create test data
  int m = 5;
  int n = 10;
  srand(time(NULL));
  this->CreateTestData(m, n);
  Nice::Vector<TypeParam> gpu_c(m);
  // Test gpu matrix matrix multiply in Nice
  Nice::CudaMatrixVectorMultiply<TypeParam> gpu_op;
  ASSERT_DEATH(gpu_op.Multiply(this->a_, this->b_), ".*");
}

TYPED_TEST(CudaMatrixVectorMultiplyTest, MatrixAndVectorEmptyTest) {
  // Create test data
  int m = 0;
  int n = 0;
  srand(time(NULL));
  this->CreateTestData(m, n);
  Nice::Vector<TypeParam> gpu_c(m);
  // Test gpu matrix matrix multiply in Nice
  Nice::CudaMatrixVectorMultiply<TypeParam> gpu_op;
  ASSERT_DEATH(gpu_op.Multiply(this->a_, this->b_), ".*");
}

TYPED_TEST(CudaMatrixVectorMultiplyTest, MatrixEmptyTest) {
  // Create test data
  int m = 10;
  int n = 5;
  srand(time(NULL));
  this->CreateTestData(m, n);
  this->a_ = Nice::Matrix<TypeParam>::Zero(m, n);
  Nice::Vector<TypeParam> gpu_c(m);
  // Test gpu matrix matrix multiply in Nice
  Nice::CudaMatrixVectorMultiply<TypeParam> gpu_op;
  ASSERT_DEATH(gpu_op.Multiply(this->a_, this->b_), ".*");
}

TYPED_TEST(CudaMatrixVectorMultiplyTest, VectorEmptyTest) {
  // Create test data
  int m = 0;
  int n = 0;
  srand(time(NULL));
  this->CreateTestData(m, n);
  this->b_ = Nice::Vector<TypeParam>::Zero(n);
  Nice::Vector<TypeParam> gpu_c(m);
  // Test gpu matrix matrix multiply in Nice
  Nice::CudaMatrixVectorMultiply<TypeParam> gpu_op;
  ASSERT_DEATH(gpu_op.Multiply(this->a_, this->b_), ".*");
}**/
