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


// This file tests the cpu_operations.cc DotProduct() function .............


#include <stdio.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

class DotProductTest : public ::testing::Test {
 public:
  Nice::Vector<int> vec1;
  Nice::Vector<int> vec2;
  int result;
};

TEST_F(DotProductTest, DotProductFunctionality) {
  this->vec1.setRandom(3);
  this->vec2.setRandom(3);
  result = Nice::CpuOperations<int>::DotProduct(vec1, vec2);
  int correct = ((vec1[0]*vec2[0])+(vec1[1]*vec2[1])+(vec1[2]*vec2[2]));
  EXPECT_EQ(result, correct);
}

TEST_F(DotProductTest, DifferentSizeVectors) {
  this->vec1.setRandom(4);
  this->vec2.setRandom(2);
  ASSERT_DEATH(Nice::CpuOperations<int>::DotProduct(vec1, vec2), ".*");
}

TEST_F(DotProductTest, EmptyVectors) {
  ASSERT_DEATH(Nice::CpuOperations<int>::DotProduct(vec1, vec2), ".*");
}
