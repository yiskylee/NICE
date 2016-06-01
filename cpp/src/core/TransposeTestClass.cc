// This file tests the cpu_operations.h Transpose function by comparing a
// Transposed Eigen::Matrix and a nunice::Matrix
// Tests are made for integers and floats

#include <iostream>
#include <stdio.h>
#include "Eigen/Eigen/Dense"
#include <gtest/gtest.h>
#include "core/cpu_operations.h"

/*
class MyStackTest : public::testing::Test {
 public:
  Eigen::Matrix3 m2;


  virtual void SetUp() {  // Constructor
  }

  virtual void TearDown() {  // Destructor
  }

  void Transposer() {
    std::cout << "The original matrix is:" << std::endl << Eigen::cout <<
    Eigen::m2 << std::endl;  // Display original
    Eigen::m2.transposeInPlace();  // Transpose
    std::cout << "The transposed matrix is:" << std::endl << Eigen::cout <<
    Eigen::m2 << std::endl;  // Display transposed
  }
};

// This function takes a matrix as a parameter and returns the transpose
TEST_F(MyStackTest, TransposeTest)
  nunice::Matrix<int> m1;  // Matrix of integers
  nunice::m1::Random(3,3);  // Initialized
  Eigen::m2=nunice::m1;  // Set m2=m1
  Transposer();  // Transpose m2
  nunice::Transpose(m1);  // Transpose m1
  EXPECT_EQ(nunice::m1(1,2), Eigen::m2(1,2));  // Check equality
}

// The test checks to make sure that the matrix was transposed
TEST_F(MyStackTest, FloatTransposed) {
  nunice::Matrix<float> m1;  // Matrix of floats
  nunice::m1::Random(3,3);
  Eigen::m2=nunice::m1;
  Transposer();
  nunice::Transpose(m1);
  EXPECT_EQ (nunice::m1(1,2), Eigen::m2(1,2));
}

// Start and run the tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
*/
