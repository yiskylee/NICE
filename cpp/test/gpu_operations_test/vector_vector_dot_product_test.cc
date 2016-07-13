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
#include "include/cpu_operations.h"

#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"

// This is a template test fixture class containing test matrices
template<class T>  // Template
class GpuVectorVectorDotTest : public ::testing::Test {
 public:  // Members must be public to be accessed by tests
  Nice::Vector<T> a_;
  Nice::Vector<T> b_;
  T c_;

  int row_;

  // Constructor
  void CreateTestData(int m) {
    // Check matrix
    if (a_.rows() != 0)
      return;

    // Set up dimension
    row_ = m;

    // Create matrix
    a_ = Nice::Vector<T>::Random(row_);
    b_ = Nice::Vector<T>::Random(row_);
    // CPU ops
    Nice::CpuOperations<T> cpu_op;
    // Solve in CPU
    c_ = cpu_op.DotProduct(a_, b_);
  }
};
// Establishes a test case with the given types, Char and short types will
// Throw compiler errors
typedef ::testing::Types<float, double> dataTypes;
TYPED_TEST_CASE(GpuVectorVectorDotTest, dataTypes);

TYPED_TEST(GpuVectorVectorDotTest, FuncionalityTest) {
  // Create test data
  int m = 10;
  srand(time(NULL));
  this->CreateTestData(m);
  TypeParam gpu_c;
  // Test gpu matrix matrix multiply in Nice
  Nice::GpuOperations<TypeParam> gpu_op;
  gpu_c = gpu_op.DotProduct(this->a_, this->b_);

  // Verify the result
  EXPECT_NEAR(this->c_, gpu_c, 0.001);
}

