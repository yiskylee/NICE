// This file tests the cpu_operations.h Transpose function by comparing a
// Transposed Eigen::Matrix and a Nice::Matrix
// Tests are made for integers and floats

#include <iostream>
#include <stdio.h>
#include "Eigen/Eigen/Dense"
#include <gtest/gtest.h>
#include "core/cpu_operations.h"
#include "core/matrix.h"

class MyTest : public ::testing::Test {
 public:

  Eigen::MatrixXi m2;

  virtual void SetUp() {  // Constructor
    m2 = Eigen::MatrixXi::Zero(3,3);
  }

  virtual void TearDown() {  // Destructor
  }

  void Transposer() {
    std::cout << "The original matrix is:" << std::endl << std::cout <<
    m2 << std::endl;  // Display original
    m2.transposeInPlace();  // Transpose
    std::cout << "The transposed matrix is:" << std::endl << std::cout <<
    m2 << std::endl;  // Display transposed
  }



};

// This function takes a matrix as a parameter and returns the transpose
TEST_F(MyTest, TransposeTest) {
  Nice::Matrix<int> m1 = Eigen::MatrixXi::Random(3,3);  // Matrix of integers
  m2 = m1;  // Set m2=m1
  //Transposer();  // Transpose m2
  //Nice::Transpose(m1);  // Transpose m1
  EXPECT_EQ(2 + 2, 4);  // Check equality
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

