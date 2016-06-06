// This file tests the cpu_operations.h Transpose function by comparing a
// Transposed Eigen::Matrix and a Nice::Matrix
// Test is made for integers only

#include <iostream>
#include <stdio.h>
#include "Eigen/Dense"
#include <gtest/gtest.h>
#include "cpu_operations.h"
#include "matrix.h"
#include "cpu_operations.cc"  // Included to reference the templated class

class MyTest : public ::testing::Test {
 public:

  Eigen::MatrixXi matrixEigen;  // Local integer matrix

  // The test fixture will create a class object and run the test and then
  // Delete it. If there is additional testing framework that needs to be
  // Set up, the SetUp and TearDown functions act as an additional form of
  // Constructor or Destructor to do so

  virtual void SetUp() {  // Constructor
    matrixEigen = Eigen::MatrixXi::Zero(3,3);  // Initializes matrix to zeroes
  }

  virtual void TearDown() {  // Destructor
  }

  void Transposer() {
    std::cout << "------------------------------" << std::endl;
    std::cout << "The original matrix m2 is:" << std::endl <<
    matrixEigen << std::endl;  // Display original
    matrixEigen.transposeInPlace();  // Transpose
    std::cout << "------------------------------" << std::endl;
    std::cout << "The transposed matrix is:" << std::endl <<
    matrixEigen << std::endl;  // Display transposed
    std::cout << "------------------------------" << std::endl;
  }

};

// This function takes a matrix as a parameter and returns the transpose
TEST_F(MyTest, TransposeTest) {
  Nice::Matrix<int> matrixNice = Eigen::MatrixXi::Random(3,3);  // Matrix of integers
  matrixEigen = matrixNice;  // Set m2 = m1
  Transposer();  // Transpose m2
  Nice::Matrix<int> transposeNice = Nice::CpuOperations<int>::Transpose(matrixNice);  // Transpose m1
  for(int i; i < 3; ++i) {
    for(int j; j < 3; ++i) {
      EXPECT_EQ(transposeNice(i, j), matrixEigen(i, j));  // Check equality for each element
    }
  }
}

/*
// The test checks to make sure that the matrix was transposed
TEST_F(MyStackTest, FloatTransposed) {
  Nice::Matrix<float> m1;  // Matrix of floats
  Nice::m1::Random(3,3);
  m2=Nice::m1;
  Transposer();
  Nice::Transpose(m1);
  EXPECT_EQ (Nice::m1(1,2), Eigen::m2(1,2));
}
*/

// Start and run the tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

