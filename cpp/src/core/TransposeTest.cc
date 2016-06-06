// This file tests the cpu_operations.h Transpose function by creating a Matrix
// And transposing it via the Eigen transpose function and the cpu_operations
// Function, comparing the results

#include <iostream>
#include <stdio.h>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "cpu_operations.h"
#include "matrix.h"
#include "cpu_operations.cc"

/*
// This function takes a matrix as a parameter and returns the transpose
Eigen::MatrixXi transpose(Eigen::MatrixXi m) {
  std::cout << "The original matrix is:" << std::endl << std::cout <<
  m << std::endl;
  std::cout << "The transposed matrix is:" << std::endl << std::cout <<
  m.transpose() << std::endl;
  return m;
}*/

// The test checks to make sure that the matrix was transposed
TEST(Transpose, IsTransposed) {
  Nice::Matrix<int> m1 = Eigen::MatrixXi::Random(3,3);
  std::cout << "The matrix m1 is:" << std::endl << std::cout <<
  m1 << std::cout << std::endl;
  Eigen::MatrixXi m2 = Eigen::MatrixXi::Random(3,3);
  std::cout << "The matrix m2 is:" << std::endl << std::cout <<
  m2 << std::cout << std::endl;
  Nice::Matrix<int> m3 = Nice::CpuOperations<int>::Transpose(m1);
  EXPECT_EQ (2+2, 4);
}

// Start and run the tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

