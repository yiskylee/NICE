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

// This file tests CpuOperations::LogicalNot() to see if it will correctly
// handle basic boolean Matrices and Vectors and to see if it will throw
// an error if an uninitialized variable gets passed into its parameters

#include <stdio.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

template<class T>
class FrobeniusNorm :public ::testing::Test {
 public:
  Nice::Matrix<T> m1;

  int FrobeniusNormer() {
    return Nice::CpuOperations<T>::FrobeniusNorm(m1);
  }
}

typedef ::testing::Types<int> MyTypes;
TYPED_TEST_CASE(FrobeniusNorm, MyTypes);


TYPED_TEST(FrobeniusNorm, BasicFuntionality) {
  m1.resize(3,3);
  m1 << 1, 2, 3,
        1, 2, 3,
        1, 2, 3;
  EXPECT_EQ(42, this->FrobeniusNormer());
}
