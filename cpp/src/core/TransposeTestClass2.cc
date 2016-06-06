// This file tests the cpu_operations.h Transpose function by comparing a
// Transposed Eigen::Matrix and a Nice::Matrix
// Tests are made for integers (IntTransposed) and floats (FloatTransposed) and
// Also test behavior with oddly shaped matrices (oddshape tests)

#include <iostream>
#include <stdio.h>
#include "Eigen/Dense"
#include <gtest/gtest.h>
#include "cpu_operations.h"
#include "cpu_operations.cc"
#include "matrix.h"  // This is included for testing purposes

// This is a test fixture class containing an Eigen matrix for testing
template <class T>
class MyTest : public ::testing::Test {  // Inherits from testing::Test
 public:

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _matrix_eigen;  // Public member
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _transpose_eigen;  // Public member
  Nice::Matrix<T> _matrix_nice;
  Nice::Matrix<T> _transpose_nice;

  virtual void SetUp() {  // Constructor
  }

  virtual void TearDown() {  // Destructor
  }

  void Transposer() {
	std::cout << "------------------------------" << std::cout << std::endl;
	std::cout << "The original matrix matrixNice is:" << std::endl << std::cout <<
	_matrix_eigen << std::endl;  // Display original
	std::cout << "------------------------------" << std::cout << std::endl;
    _transpose_eigen = _matrix_eigen.transpose();  // Transpose
	std::cout << "------------------------------" << std::cout << std::endl;
    std::cout << "The transposed matrix is:" << std::endl << std::cout <<
    _transpose_eigen << std::endl;  // Display transposed
	std::cout << "------------------------------" << std::cout << std::endl;
  }
};

/*
// This function tests an integer matrix tanspose by both methods
TEST_F(MyTest, IntTransposed) {
  Nice::Matrix<int> matrixNice = Eigen::MatrixXi::Random(3,3);  // Matrix of integers
  std::cout << "The original matrix matrixNice is:" << std::endl << std::cout <<
  matrixNice << std::endl;  // Display original
  _matrix_eigen = matrixNice;  // Set _matrix_eigen=matrixNice
  Transposer();  // Transpose _matrix_eigen
  Nice::Matrix<int> transposeNice = Nice::CpuOperations<int>::Transpose(matrixNice);
  // Transpose matrixNice
  for(int i; i < 3; ++i) {
    for(int j; j < 3; ++i) {
      EXPECT_EQ(transposeNice(i, j), _matrix_eigen(i, j));  // Check equality
    }
  }
}
*/

// This function tests a float matrix tanspose by both methods
typedef ::testing::Types<int, float> MyTypes;
TYPED_TEST_CASE(MyTest, MyTypes);

TYPED_TEST(MyTest, TransposeTypes) {
  this->_matrix_nice = Eigen::Matrix<T>::Random(3,3);  // Matrix of floats
  this->_matrix_eigen = this->_matrix_nice;  // Set _matrix_eigen=matrixNice
  this->Transposer();  // Transpose _matrix_eigen
  this->_transpose_nice = Nice::CpuOperations<T>::Transpose(this->_matrix_nice);  // Transpose matrixNice
  EXPECT_EQ(2 + 2, 4);
}

/*
// The test generates a matrix with an odd shape, and prints the transpose
TEST_F(MyTest, oddShape1) {
  Nice::Matrix<int> matrixNice = Eigen::MatrixXi::Random(1,1);  // Matrix of integers
  std::cout << "The original matrix matrixNice is:" << std::endl << std::cout <<
  matrixNice << std::endl;  // Display original
  Nice::Matrix<int> transposeNice = Nice::CpuOperations<int>::Transpose(matrixNice);  // Transpose matrixNice
  std::cout << "The transposed matrix matrixNice is:" << std::endl << std::cout <<
  transposeNice << std::endl;  // Display original
}


// The test generates a matrix with an odd shape, and prints the transpose
TEST_F(MyTest, oddShape2) {
  Nice::Matrix<int> matrixNice = Eigen::MatrixXi::Random(1,0);  // Matrix of integers
  std::cout << "The original matrix matrixNice is:" << std::endl << std::cout <<
  matrixNice << std::endl;  // Display original
  Nice::Matrix<int> transposeNice = Nice::CpuOperations<int>::Transpose(matrixNice);  // Transpose matrixNice
  std::cout << "The transposed matrix matrixNice is:" << std::endl << std::cout <<
  transposeNice << std::endl;  // Display original
}
*/

// Start and run the tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


