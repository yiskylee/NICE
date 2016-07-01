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


// This file tests the cpu_operations.cc DotProduct() function. First, it tests
// the functionality to ensure the dot product works properly by manually
// calculating the dot product and comparing it to the result of the function
// which calculated dot product with the Eigen built-in functionality. Then
// the two cases where incorrect function uses will result in fatal error are
// tested. This involves trying to calculate the dot product of two vectors of
// different size or trying to calculate the dot product of empty vectors.

#include <stdio.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/vector.h"

template<class T>
class DotProductTest : public ::testing::Test {
 public:
  Nice::Vector<T> vec1;
  Nice::Vector<T> vec2;
  T result;

  void DotProd() {
    result = Nice::CpuOperations<T>::DotProduct(this->vec1, this->vec2);
  }
};

typedef ::testing::Types<int, float, double> MyTypes;
TYPED_TEST_CASE(DotProductTest, MyTypes);

TYPED_TEST(DotProductTest, DotProductFunctionality) {
  int vec_size = 15;
  this->vec1.setRandom(vec_size);
  this->vec2.setRandom(vec_size);

  TypeParam correct = 0;
  for (int i = 0; i < vec_size; ++i)
    correct += (this->vec1[i]*this->vec2[i]);

  this->DotProd();

  EXPECT_EQ(this->result, correct);
}

TYPED_TEST(DotProductTest, DifferentSizeVectors) {
  this->vec1.setRandom(4);
  this->vec2.setRandom(2);
  ASSERT_DEATH(this->DotProd(), ".*");
}

TYPED_TEST(DotProductTest, EmptyVectors) {
  ASSERT_DEATH(this->DotProd(), ".*");
}
