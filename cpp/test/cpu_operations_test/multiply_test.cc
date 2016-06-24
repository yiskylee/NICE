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
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

TEST(Mutilply, ScalarMatrixMult) {
  int scalar;
  scalar = 3;
  Eigen::MatrixXi a(3, 3);
  a << 0, 1, 2,
       3, 2, 1,
       1, 3, 0;
  Eigen::MatrixXi correct_ans(3, 3);
  correct_ans << 0, 3, 6,
                 9, 6, 3,
                 3, 9, 0;
  Nice::Matrix<int> calc_ans = Nice::CpuOperations<int>::Multiply(a, scalar);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        EXPECT_EQ(correct_ans(i, j), calc_ans(i, j));
}

TEST(Mutilply, MatrixMatrixMult) {
  Eigen::MatrixXi a(3, 3);
  a << 0, 1, 2,
       3, 2, 1,
       1, 3, 0;
  Eigen::MatrixXi b(3, 3);
  b << 1, 0, 2,
       2, 1, 0,
       0, 2, 1;
  Eigen::MatrixXi correct_ans(3, 3);
  correct_ans << 2, 5, 2,
                 7, 4, 7,
                 7, 3, 2;
  Nice::Matrix<int> calc_ans = Nice::CpuOperations<int>::Multiply(a, b);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        EXPECT_EQ(correct_ans(i, j), calc_ans(i, j));
}
