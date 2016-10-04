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
#include <iostream>
#include "Eigen/Dense"
#include "include/util.h"
#include "include/matrix.h"
#include "gtest/gtest.h"

template<class T>
class ToFileTest : public ::testing::Test {
 public:
  Nice::Matrix<T> m;
  Nice::Matrix<T> m_expected;
  Nice::Vector<T> v;
  Nice::Vector<T> v_expected;
};

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(ToFileTest, MyTypes);

TYPED_TEST(ToFileTest, Matrix) {
  int row = 20;
  int col = 30;
  this->m.resize(row, col);
  this->m = Nice::Matrix<TypeParam>::Random(row, col);
  Nice::util::ToFile<TypeParam>(this->m,
                                "../test/data_for_test/test_matrix.txt");
  this->m_expected = Nice::util::FromFile<TypeParam>(
                                "../test/data_for_test/test_matrix.txt",
                                ",");
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      EXPECT_NEAR(this->m_expected(i, j), this->m(i, j), 0.0001);

  std::remove("../test/data_for_test/test_matrix.txt");
}

TYPED_TEST(ToFileTest, Vector) {
  int row = 2;
  this->v.resize(row);
  this->v = Nice::Vector<TypeParam>::Random(row);
  Nice::util::ToFile<TypeParam>(this->v,
                                "../test/data_for_test/test_vector.txt");
  this->v_expected = Nice::util::FromFile<TypeParam>(
                                "../test/data_for_test/test_vector.txt",
                                " ");
  for (int i = 0; i < row; i++)
    EXPECT_NEAR(this->v_expected(i), this->v(i), 0.0001);

  std::remove("../test/data_for_test/test_vector.txt");
}
