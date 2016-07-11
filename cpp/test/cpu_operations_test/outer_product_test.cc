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





#include <unistd.h>
#include <iostream>
#include "include/cpu_operations.h"
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/matrix.h"
#include "include/vector.h"

// Typed Tests
template<class T>
class OuterProductTest : public ::testing::Test {
 public :
  Nice::Vector<T> v1;
  Nice::Vector<T> v2;
  Nice::Matrix<T> m1;
  Nice::Matrix<T> m2;

  void OuterProducter() {
    m2 = Nice::CpuOperations<T>::OuterProduct(this->v1, this->v2);
  }
};

typedef ::testing::Types<int, float, double> MyTypes;
TYPED_TEST_CASE(OuterProductTest, MyTypes);

// Tests a regular outer product operation
TYPED_TEST(OuterProductTest, BasicFunctionality) {
  this->v1.resize(2);
  this->v2.resize(3);
  this->m1.resize(2, 3);
  this->v1 << 1, 2;
  this->v2 << 3, 4, 5;
  this->m1 << 3, 4, 5,
              6, 8, 10;
  this->OuterProducter();
  ASSERT_TRUE(this->m1.isApprox(this->m2));
}

// Tests with empty vectors
TYPED_TEST(OuterProductTest, EmptyVectors) {
  ASSERT_DEATH(this->OuterProducter(), ".*");
}
