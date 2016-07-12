#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

TEST(Rank, RankMatrix){

Eigen::MatrixXi a(4,4);
a << 1, 3, 5, 2
     0, 1, 0, 3,
     0, 0, 0, 1,
     0, 0, 0, 0;

int correct_ans = 3;

int calculated_ans = Nice::CpuOperations<int>::Rank(a);
EXPECT_EQ(correct_ans, calculated_ans);

}
