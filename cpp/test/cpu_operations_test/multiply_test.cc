// Copyright 2016 <Morgan Rockett>
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
