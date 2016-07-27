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
class GPU_MATRIX_SCALAR_ADD : public ::testing::Test {
 public:
  Nice::Matrix<T> a;
  Nice::Matrix<T> correct_ans;
  Nice::Matrix<T> calc_ans;
  T scalar;

  void Add() {
    calc_ans = Nice::GpuOperations<T>::Add(a, scalar);
  }
};

typedef ::testing::Types<float, double> dataTypes;
TYPED_TEST_CASE(GPU_MATRIX_SCALAR_ADD, dataTypes);

TYPED_TEST(GPU_MATRIX_SCALAR_ADD, Basic_Test) {
  this->a.resize(3, 4);
  this->a << 1, 2, 3, 4,
             1, 2, 3, 4,
             1, 2, 3, 4;
  this->scalar = 1;
  this->Add();
  this->correct_ans.resize(3, 4);
  this->correct_ans << 2, 3, 4, 5,
                       2, 3, 4, 5,
                       2, 3, 4, 5;
  ASSERT_TRUE(this->correct_ans.isApprox(this->calc_ans));
}
