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

#include "include/gpu_operations.h"
#include "Eigen/Dense"
#include "gtest/gtest.h"

template<class T>
class GpuMatrixMatrixSubTest : public ::testing::Test {
 public:
  Nice::Matrix<T> a;
  Nice::Matrix<T> b;
  Nice::Matrix<T> result;
  Nice::Matrix<T> correct;

  void Subtract() {
    result = Nice::GpuOperations<T>::Subtract(a, b);
  }
};

typedef ::testing::Types<float, double> dataTypes;
TYPED_TEST_CASE(GpuMatrixMatrixSubTest, dataTypes);

TYPED_TEST(GpuMatrixMatrixSubTest, SubtractFunctionality) {
  this->a.setRandom(3, 3);
  this->b.setRandom(3, 3);

  this->correct.setZero(3, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      this->correct(i, j) = this->a(i, j) - this->b(i, j);
    }
  }

  this->Subtract();

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      ASSERT_NEAR(this->result(i, j), this->correct(i, j), 0.0001);
    }
  }
}

TYPED_TEST(GpuMatrixMatrixSubTest, DifferentSizeMatricies) {
  this->a.setRandom(2, 3);
  this->b.setRandom(3, 4);
  ASSERT_DEATH(this->Subtract(), ".*");
}

TYPED_TEST(GpuMatrixMatrixSubTest, EmptyMatrix) {
  ASSERT_DEATH(this->Subtract(), ".*");
}
