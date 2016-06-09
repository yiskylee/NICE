// This file tests the cpu_operations.cc Transpose() function by comparing a
// Transposed Eigen::Matrix and a Nice::Matrix
// Tests are made for integers and floats by using a template type test
// Behavior with oddly shaped matrices (oddshape tests) is also tested with
// Matrices of size (1,4) and (0,2)

#include <iostream>
#include <stdio.h>
#include "Eigen/Dense"
#include <gtest/gtest.h>
#include "cpu_operations.h"
#include "cpu_operations.cc"  // Included to reference the templated class
#include "matrix.h"  // This is included for testing purposes

// This is a test fixture class containing NICE matrices
class MyTest : public ::testing::Test {  // Inherits from testing::Test
 public:

  Nice::Matrix<bool> _matrix_nice1;
  Nice::Matrix<bool> _matrix_nice2;
  Nice::Matrix<bool> _logical_and;

  // Prints out the original and logical and matrices for reference
  void LogicalAnd() {
    std::cout << std::endl << "------------------------------" << std::endl;
    std::cout << std::endl << "The original matrices are:" << std::endl
        << _matrix_nice1 << std::endl;  // Display original
    std::cout << "------------------------------" << std::endl;
    std::cout << _matrix_nice2 << std::endl;  // Display original
    std::cout << "------------------------------" << std::endl;
    // Apply logic to matrixNice
    _logical_and = Nice::CpuOperations<bool>::LogicalAnd(_matrix_nice1, _matrix_nice2);
    std::cout << "The LogicalAnd matrix is:" << std::endl << _logical_and
        << std::endl;  // Display new matrix
    std::cout << "------------------------------" << std::endl;
  }
};

TEST_F(MyTest, LogicalAndTest1) {
  this->_matrix_nice1.setRandom(3,3);  // Random bool values
  this->_matrix_nice2.setRandom(3,3);
  this->_logical_and.setZero(3,3);
  this->LogicalAnd();
  for(int i=0; i < 3; ++i) {
    for(int j=0; j < 3; ++j) {
      // Check equality for each element
      EXPECT_EQ(((this->_matrix_nice1(i, j)) && (this->_matrix_nice2(i, j))), (this->_logical_and(i, j)));
    }
  }
}

TEST_F(MyTest, LogicalAndTest2) {
  this->_matrix_nice1.setRandom(3,4);  // Random bool values
  this->_matrix_nice2.setRandom(2,3);
  this->_logical_and.setZero(3,3);
  this->LogicalAnd();  // Expect failure, display message
}

TEST(Tests, IntTest) {
  Nice::Matrix<int> matrix_int_1;
  matrix_int_1.setRandom(3,3);
  Nice::Matrix<int> matrix_int_2;
  matrix_int_2.setRandom(3,3);
//  Nice::Matrix<int> matrix_logical_and = Nice::CpuOperations<int>::LogicalAnd(matrix_int_1, matrix_int_2);
}

// Start and run the tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

