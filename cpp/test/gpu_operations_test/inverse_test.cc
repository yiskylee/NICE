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
class GpuInverseTest : public ::testing::Test {  // Inherits testing::Test
 public:  // Members must be public to be accessed by tests
  Nice::Matrix<T> matrix_;
  int row_;
  int col_;
  Nice::Matrix<T> ref_result_;
  Nice::Matrix<T> gpu_result_;

  // Constructor
  void CreateTestData() {
    // Check matrix
    if (matrix_.rows() != 0 && matrix_.cols() != 0)
      return;

    // Set up dimension
    row_ = 5;
    col_ = row_;

    // Create matrix
    matrix_.resize(row_, col_);
    matrix_ << 0.940058,   0.010484,   0.684214,   0.068308,   0.775155,
               0.362836,   0.536293,   0.570479,   0.291084,   0.890316,
               0.038190,   0.858041,   0.881434,   0.435262,   0.762851,
               0.498995,   0.500425,   0.762412,   0.644035,   0.332466,
               0.734139,   0.897120,   0.856919,   0.729722,   0.363101;

    // Create reference results
    ref_result_.resize(row_, col_);
    ref_result_ << 0.516486,   0.197357,  -0.946819,  -1.058853,   1.372200,
                   -0.251945,  -0.071648,   0.502355,  -2.855800,   2.272983,
                   1.387686,  -3.107333,   2.349968,   0.351579,  -0.602387,
                   -1.633532,   2.397464,  -2.050379,   3.827388,  -1.588003,
                   -0.413823,   2.293140,  -0.752141,   0.675126,  -1.023207;
  }
};
// Establishes a test case with the given types, Char and short types will
// Throw compiler errors
typedef ::testing::Types<float, double> dataTypes;
TYPED_TEST_CASE(GpuInverseTest, dataTypes);

TYPED_TEST(GpuInverseTest, FuncionalityTest) {
  // Create test data
  this->CreateTestData();

  // Test inverse in Nice
  this->gpu_result_ = Nice::GpuOperations<TypeParam>::Inverse(this->matrix_);

  // Verify
  for (int i = 0; i < this->row_; i++)
    for (int j = 0; j < this->col_; j++)
      EXPECT_NEAR(this->ref_result_(i, j), this->gpu_result_(i, j), 0.001);
}
