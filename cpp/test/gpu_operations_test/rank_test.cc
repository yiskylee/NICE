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

#include <iostream>
#include <cmath>

#include "include/gpu_operations.h"
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"
#include "include/cpu_operations.h"

// This is a template test fixture class containing test matrices
template<class T>  // Template
class GpuRankTest : public ::testing::Test {  // Inherits testing::Test
 public:  // Members must be public to be accessed by tests
  Nice::Matrix<T> matrix_;
  int row_;
  int col_;
  int cpu_rank_ = 0;
  int gpu_rank_ = 0;

  // Constructor
  void CreateTestData() {
    // Check matrix
    if (matrix_.rows() != 0 && matrix_.cols() != 0)
      return;

    // Set up dimension
    row_ = 5;
    col_ = row_;

    // Create matrix
    matrix_ = Nice::Matrix<T>::Random(row_, col_);

    // Do CPU trace computation
    // cpu_rank_ = Nice::CpuOperations<T>::Rank(matrix_);
    cpu_rank_ = 5;
  }
};
// Establishes a test case with the given types, Char and short types will
// Throw compiler errors
typedef ::testing::Types<float, double> dataTypes;
TYPED_TEST_CASE(GpuRankTest, dataTypes);

TYPED_TEST(GpuRankTest, FuncionalityTest) {
  // Create test data
  srand(time(NULL));
  this->CreateTestData();

  // Test trace in Nice
  this->gpu_rank_ = Nice::GpuOperations<TypeParam>::Rank(this->matrix_);

  // Verify
  EXPECT_EQ(this->gpu_rank_, this->cpu_rank_);
}

