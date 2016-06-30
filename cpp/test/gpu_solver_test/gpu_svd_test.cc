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

#include "include/gpu_svd_solver.h"

#include <iostream>
#include <cmath>

#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/matrix.h"
#include "include/vector.h"

// This is a template test fixture class containing test matrices
template<class T>  // Template
class GpuSvdSolverTest : public ::testing::Test {  // Inherits testing::Test
 public:  // Members must be public to be accessed by tests
  Nice::Matrix<T> matrix_;
  Nice::Matrix<T> u_;
  Nice::Matrix<T> v_;
  Nice::Vector<T> s_;
  int row_;
  int col_;

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
    // CPU SVD
    Eigen::JacobiSVD< Nice::Matrix<T> > cpu_svd;

    // Solve in CPU
    cpu_svd.compute(matrix_, Eigen::ComputeFullU|Eigen::ComputeFullV);

    // Get GPU SVD results
    s_ = cpu_svd.singularValues();
    u_ = cpu_svd.matrixU();
    v_ = cpu_svd.matrixV();
  }
};
// Establishes a test case with the given types, Char and short types will
// Throw compiler errors
typedef ::testing::Types<float, double> dataTypes;
TYPED_TEST_CASE(GpuSvdSolverTest, dataTypes);

TYPED_TEST(GpuSvdSolverTest, FuncionalityTest) {
  // Create test data
  this->CreateTestData();

  // Test svd solver in Nice
  Nice::GpuSvdSolver<TypeParam> gpu_svd;
  gpu_svd.Compute(this->matrix_);
  Nice::Vector<TypeParam> gpu_s = gpu_svd.SingularValues();
  Nice::Matrix<TypeParam> gpu_u = gpu_svd.MatrixU();
  Nice::Matrix<TypeParam> gpu_v = gpu_svd.MatrixV();

  // Verify the result U
  for (int i = 0; i < this->row_; i++)
    for (int i = 0; i < this->col_; i++)
      EXPECT_NEAR(abs(this->u_(i)), abs(gpu_u(i)), 0.001);

  // Verify the result V
  // for (int i = 0; i < this->row_; i++)
  //  for (int i = 0; i < this->col_; i++)
  //    EXPECT_NEAR(abs(this->v_(i)), abs(gpu_v(i)), 0.001);

  // Verify the result S
  for (int i = 0; i < this->row_; i++)
    EXPECT_NEAR(this->s_(i), gpu_s(i), 0.1);
}

