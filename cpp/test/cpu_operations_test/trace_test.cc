// Copyright 2016 <Morgan Rockett>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"
TEST(Trace, TraceMatrix) {
Eigen::MatrixXi a(4, 4);
a << 8, 5, 3, 4,
     2, 4, 8, 9,
     7, 6, 1, 0,
     9, 2, 5, 7;
int correct_ans = 20;
int calc_ans = Nice::CpuOperations<int>::Trace(a);
EXPECT_EQ(correct_ans, calc_ans);
}
