// This file tests the cpu_operations.h Transpose function by creating a Matrix
// And transposing it via the Eigen transpose function and the cpu_operations
// Function, comparing the results

#include <iostream>
#include <stdio.h>
#include "Eigen/Eigen/Dense"
#include <gtest/gtest.h> 
#include "core/cpu_operations.h"


// This function takes a matrix as a parameter and returns the transpose
Eigen::Matrix3i transpose(Eigen::Matrix3i m) {
  std::cout << "The original matrix is:" << std::endl << Eigen::cout <<
  Eigen::m << std::endl;
  std::cout << "The transposed matrix is:" << std::endl << Eigen::cout <<
  Eigen::m.transpose() << std::endl;
  return Eigen::m.transpose();
}

// The test checks to make sure that the matrix was transposed
TEST(Transpose, IsTransposed) {
  nice::Matrix<int> m1;
  nice::m1::Random(3,3);
  Eigen::Matrix3i m2=nice::m1;
  Eigen::Matrix3i m3 = transpose(m2);
  nice::Transpose(m1);
  EXPECT_EQ (Eigen::m3(1, 2), nice::m1(1, 2));
}

// Start and run the tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
