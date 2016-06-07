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
template <class T>  // Template
class MyTest : public ::testing::Test {  // Inherits from testing::Test
 public:

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _matrix_eigen;  // Public members
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _transpose_eigen;
  Nice::Matrix<T> _matrix_nice;
  Nice::Matrix<T> _transpose_nice;

  // Prints out the original and transposed Eigen matrix for reference
  void Transposer() {
    std::cout << std::endl << "------------------------------" << std::endl;
    std::cout << std::endl << "The original matrix matrixNice is:" << std::endl <<
    _matrix_eigen << std::endl;  // Display original
	std::cout << "------------------------------" << std::endl;
    _transpose_eigen = _matrix_eigen.transpose();  // Transpose
    // Transpose matrixNice
    _transpose_nice = Nice::CpuOperations<T>::Transpose(_matrix_nice);
    std::cout << "The transposed matrix is:" << std::endl <<
    _transpose_eigen << std::endl;  // Display transposed
	std::cout << "------------------------------" << std::endl;
  }
};

// Establishes a test case with the given types
typedef ::testing::Types<int, float> MyTypes;
TYPED_TEST_CASE(MyTest, MyTypes);


// Transposes a matrix instantiated with random ints/floats and compares
// Each element of the Eigen and Nice matrices
// this ->  is used to refer to an element of the fixture class
TYPED_TEST(MyTest, TransposeTypes) {
  this->_matrix_nice.setRandom(3,3);  // Random values
  this->_matrix_eigen = this->_matrix_nice;  // Set _matrix_eigen=_matrix_Nice
  this->Transposer();  // Transpose _matrix_eigen
  for(int i; i < 3; ++i) {
    for(int j; j < 3; ++i) {
      // Check equality for each element
      EXPECT_EQ(this->_transpose_nice(i, j), this->_transpose_eigen(i, j));
    }
  }
}

// This function uses a non-square matrix size and compares the number of rows
// And Columns after the transposition
TYPED_TEST(MyTest, oddShape1) {
  this->_matrix_nice.setRandom(1,4);
  this->_matrix_eigen = this->_matrix_nice;
  this->Transposer();
  //Check equality
  EXPECT_EQ(this->_transpose_nice.rows(),this->_transpose_eigen.rows());
  EXPECT_EQ(this->_transpose_nice.cols(),this->_transpose_eigen.cols());
}

// This function uses a matrix with size (0,2) and compares the number of rows
// And Columns after the transposition
TYPED_TEST(MyTest, oddShape2) {
  this->_matrix_nice.setRandom(0,2);
  this->_matrix_eigen = this->_matrix_nice;
  this->Transposer();
  EXPECT_EQ(this->_transpose_nice.rows(),this->_transpose_eigen.rows());
  EXPECT_EQ(this->_transpose_nice.cols(),this->_transpose_eigen.cols());
}

// Start and run the tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


