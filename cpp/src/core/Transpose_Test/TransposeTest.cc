// This file contains a simple tranpose function and a test to make sure that
// the transpose worked

#include "../../../../Eigen/Eigen/Dense"
#include <gtest/gtest.h> 
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace Eigen;

// This function takes a matrix as a parameter and returns the transpose
Matrix2i transpose(Matrix2i m) {
  cout << "Here is the matrix m:" << endl << m << endl;
  cout << "Here is the transpose of m:" << endl << m.transpose() << endl;
  cout << "Here is the coefficient (0,1) in the matrix m:" << endl << m(0,1)
       << endl;
  cout << "Here is the coefficient (1,0) in the transpose of m:" << endl
       << m.transpose()(1,0) << endl;
  return m.transpose();
}

// The test checks to make sure that the matrix was transposed
TEST(Transpose, IsTransposed) {
  Matrix2i m1 = Matrix2i::Random();
  Matrix2i m2 = transpose(m1);
  EXPECT_EQ (m1(0,1), m2(1,0));
}

// Start and run the tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
