// The MIT License (MIT)
//
// Copyright (c) 2016 Northeastern University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


// This file tests the cpu_operations.cc Inverse() function by calling it
// on a square, non-singular matrix (should work), a sqaure, signular matrix
// (should not work), a matrix that is not square (should not work), and a
// matrix that is empty (should not work).

#include <stdio.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"
#include "include/vector.h"

template<class T>
class InverseTest : public ::testing::Test {
 public:
  Nice::Matrix<T> input;
  Nice::Matrix<T> output;
  Nice::Matrix<T> correct;

  void GetInverse() {
    output = Nice::CpuOperations<T>::Inverse(input);
  }
};


typedef ::testing::Types<int, float, double> MyTypes;
TYPED_TEST_CASE(InverseTest, MyTypes);

TYPED_TEST(InverseTest, InverseFunctionality) {
  this->input.setRandom(3, 3);
  this->input << 1, 2, 3,
                 0, 1, 4,
                 5, 6, 0;
  this->correct.setZero(3, 3);
  this->correct << -24,  18,   5,
                    20, -15,  -4,
                    -5,   4,   1;
  this->GetInverse();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_EQ(this->output(i, j), this->correct(i, j));
    }
  }
}

TYPED_TEST(InverseTest, SingularMatrix) {
  this->input.setConstant(2, 2, 10);
  ASSERT_DEATH(this->GetInverse(), ".*");
}

TYPED_TEST(InverseTest, NonSquareMatrix) {
  this->input.setRandom(2, 3);
  ASSERT_DEATH(this->GetInverse(), ".*");
}

TYPED_TEST(InverseTest, EmptyMatrix) {
  ASSERT_DEATH(this->GetInverse(), ".*");
}
