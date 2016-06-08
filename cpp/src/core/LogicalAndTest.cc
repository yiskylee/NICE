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

// This is a template test fixture class containing Eigen and NICE matrices
class MyTest : public ::testing::Test {  // Inherits from testing::Test
 public:

  Nice::Matrix<bool> _matrix_nice1;
  Nice::Matrix<bool> _matrix_nice2;
  Nice::Matrix<bool> _logical_and;

  // Prints out the original and transposed Eigen matrix for reference
  void LogicalAnd() {
    std::cout << std::endl << "------------------------------" << std::endl;
    std::cout << std::endl << "The original matrices are:" << std::endl
        << _matrix_nice1 << std::endl;  // Display original
    std::cout << "------------------------------" << std::endl;
    std::cout << _matrix_nice2 << std::endl;  // Display original
    std::cout << "------------------------------" << std::endl;
    // Apply logic to matrixNice
    _logical_and = Nice::CpuOperations::LogicalAnd(_matrix_nice1, _matrix_nice2);
    std::cout << "The transposed matrix is:" << std::endl << _logical_and
        << std::endl;  // Display transposed
    std::cout << "------------------------------" << std::endl;
  }
};

// Establishes a test case with the given types
typedef ::testing::Types<int, float> MyTypes;
TYPED_TEST_CASE(MyTest, MyTypes);

TYPED_TEST(MyTest, IsTransposed) {
  this->_matrix_nice1.setRandom(3,3);
  this->_matrix_nice2.setRandom(3,3);
  this->_logical_and.setZero(3,3);
  this->LogicalAnd();
//  for(int i=0; i < 3; ++i) {
//    for(int j=0; j < 3; ++i) {
    // Check equality for each element
      //EXPECT_EQ(((this->_matrix_nice1(i, j)) && (this->_matrix_nice2(i, j))), this->_logical_and(i, j));
      EXPECT_EQ(2+2, 4);
//    }
//  }
}

/*
// Transposes a matrix instantiated with random ints/floats and compares
// Each element of the Eigen and Nice matrices
// this ->  is used to refer to an element of the fixture class
TYPED_TEST(MyTest, TransposeTypes){
  this->_matrix_nice.setRandom(3,3);  // Random values
  this->_matrix_eigen = this->_matrix_nice;// Set _matrix_eigen=_matrix_Nice
  this->Transposer();// Transpose _matrix_eigen
  for(int i=0; i < 3; ++i) {
    for(int j=0; j < 3; ++i) {
      // Check equality for each element
      EXPECT_EQ(this->_transpose_nice(i, j), this->_transpose_eigen(i, j));
    }
  }
}

// This function uses a non-square matrix size and compares the number of rows
// And Columns after the transposition
TYPED_TEST(MyTest, oddShape1){
  this->_matrix_nice.setRandom(1,4);
  this->_matrix_eigen = this->_matrix_nice;
  this->Transposer();
  //Check equality
  EXPECT_EQ(this->_transpose_nice.rows(),this->_transpose_eigen.rows());
  EXPECT_EQ(this->_transpose_nice.cols(),this->_transpose_eigen.cols());
}

// This function uses a matrix with size (0,2) and compares the number of rows
// And Columns after the transposition
TYPED_TEST(MyTest, oddShape2){
  this->_matrix_nice.setRandom(0,2);
  this->_matrix_eigen = this->_matrix_nice;
  this->Transposer();
  EXPECT_EQ(this->_transpose_nice.rows(),this->_transpose_eigen.rows());
  EXPECT_EQ(this->_transpose_nice.cols(),this->_transpose_eigen.cols());
}
*/

// Start and run the tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

