#include <stdio.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

template<class T>
class MatrixSubtractTest : public ::testing::Test {
 public:
  Nice::Matrix<T> m1;
  Nice::Matrix<T> m2;
  Nice::Matrix<T> result;
  Nice::Matrix<T> testMatrix;

  void MatrixSubtract() {
    result = Nice::CpuOperations<T>::Subtract(m1, m2);
  }
};

typedef ::testing::Types<int, double, float> MyTypes;
TYPED_TEST_CASE(MatrixSubtractTest, MyTypes);

TYPED_TEST(MatrixSubtractTest, MatrixSubtractFunctionality) {
  this->m1.resize(2,2);
  this->m2.resize(2,2);
 // this->result.resize(2,2);
  this->testMatrix.resize(2,2);
  this->m1 << 2, 3,
              4, 5;
  this->m2 << 1, 2,
              3, 2;
  this->MatrixSubtract();
  this->testMatrix << 1, 1,
                      1, 3;
  ASSERT_TRUE(this->result.isApprox(this->testMatrix));
}

TYPED_TEST(MatrixSubtractTest, DifferentSizeMatrix) {
  this->m1.resize(2,2);
  this->m2.resize(3,2);
  this->m1.setZero();
  this->m2.setZero();
  ASSERT_DEATH(this->MatrixSubtract(), ".*");
}

TYPED_TEST(MatrixSubtractTest, EmptyMatrix) {
  ASSERT_DEATH(this->MatrixSubtract(), ".*");
}
