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

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

template<class T>
class TraceTest : public ::testing::Test {
 public:
  Nice::Matrix<T> m1;
  T correct_ans;
  T Tracer() {
    return Nice::CpuOperations<T>::Trace(m1);
  }
};

typedef ::testing::Types<int, float, double> MyTypes;
TYPED_TEST_CASE(TraceTest, MyTypes);

TYPED_TEST(TraceTest, BasicTest) {
  this->m1.resize(4, 4);
  this->m1 << 8, 5, 3, 4,
              2, 4, 8, 9,
              7, 6, 1, 0,
              9, 2, 5, 7;
  this->correct_ans = 20;
  EXPECT_EQ(this-> correct_ans, this->Tracer());
}
