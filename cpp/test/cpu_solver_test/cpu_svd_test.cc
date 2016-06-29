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

#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/svd_solver.h"
#include "include/matrix.h"
#include "include/vector.h"

// This is a template test fixture class containing test matrices
template<class T>  // Template
class CpuSvdSolverTest : public ::testing::Test {
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

    row_ = 5;
    col_ = row_;
    matrix_.resize(row_, col_);
    matrix_ << -129.3026,  1031.7118,  2548.9163,  -511.0120,  2.7719,
                583.9671,  220.9613,  536.9512,  55.0691,  954.5454,
                694.9577,  -626.7673,  -972.7809,  800.4132,  298.9344,
                659.9324,  1235.7984,  -688.4173,  -55.1088,  -1194.2583,
                1552.5551,  513.1012,  -2574.5784,  -1489.1966,  -55.4250;
    u_.resize(row_, col_);
    u_ << -0.555596,  -0.617009,  -0.244002,  -0.092006,   0.492557,
          -0.088014,  -0.051527,  -0.654019,  -0.478284,  -0.577150,
          0.240369,   0.395314,  -0.108216,  -0.651940,   0.590943,
          0.228426,  -0.529382,   0.585784,  -0.527902,  -0.213902,
          0.757372,  -0.424414,  -0.397328,   0.243049,   0.171226;
    v_.resize(row_, col_);
    v_ << 0.363743,  -0.277844,  -0.404770,  -0.608509,   0.506332,
          -0.017398,  -0.718194,   0.118253,  -0.281295,  -0.625128,
          -0.913964,  -0.214438,  -0.153148,  -0.105130,   0.290136,
          -0.160719,   0.524143,   0.346509,  -0.733898,  -0.201915,
          -0.078912,  0.293753,  -0.823805,  -0.030541,  -0.477383;
    s_.resize(row_);
    s_ << 4162.54, 2461.32, 1620.37, 1136.40, 265.90;
  }
};

// Establishes a test case with the given types, Char and short types will
// Throw compiler errors
typedef ::testing::Types<float, double> dataTypes;
TYPED_TEST_CASE(CpuSvdSolverTest, dataTypes);

TYPED_TEST(CpuSvdSolverTest, FuncionalityTest) {
  // Create test data
  this->CreateTestData();

  // Test svd solver in Nice
  Nice::SvdSolver<TypeParam> svd_solver;
  svd_solver.Compute(this->matrix_);
  Nice::Matrix<TypeParam> result_u = svd_solver.MatrixU();
  Nice::Matrix<TypeParam> result_v = svd_solver.MatrixV();
  Nice::Vector<TypeParam> result_s = svd_solver.SingularValues();

  // Verify the result U
  for (int i = 0; i < this->row_; i++)
    for (int i = 0; i < this->col_; i++)
      EXPECT_NEAR(abs(this->u_(i)), abs(result_u(i)), 0.001);

  // Verify the result V
  for (int i = 0; i < this->row_; i++)
    for (int i = 0; i < this->col_; i++)
      EXPECT_NEAR(abs(this->v_(i)), abs(result_v(i)), 0.001);

  // Verify the result S
  for (int i = 0; i < this->row_; i++)
    EXPECT_NEAR(this->s_(i), result_s(i), 0.1);
}

TYPED_TEST(CpuSvdSolverTest, RankTest) {
  // Create test data
  this->CreateTestData();

  // Test svd solver in Nice
  Nice::SvdSolver<TypeParam> svd_solver;
  int rank = svd_solver.Rank(this->matrix_);

  // Verify the rank
  EXPECT_EQ(5, rank);
}

