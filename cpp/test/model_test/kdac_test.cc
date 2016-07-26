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
#include <memory>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/kdac.h"
#include "include/util.h"
#include "include/matrix.h"
#include "include/vector.h"

template<typename T>
class KDACTest : public ::testing::Test {
 protected:
  Nice::Matrix<T> data_matrix_;
  std::shared_ptr<Nice::KDAC<T>> kdac_;
  int c_;
  int n_;
  int d_;

  virtual void SetUp() {
    data_matrix_ = Nice::util::FromFile<T>(
        "../test/data_for_test/kdac/data_matrix_40_2.txt");
    c_ = 2;
    kdac_ = std::make_shared<Nice::KDAC<T>>(c_);
  }

};

typedef ::testing::Types<float, int, long, double> AllTypes;
typedef ::testing::Types<int, long> IntTypes;
typedef ::testing::Types<float, double> FloatTypes;


TYPED_TEST_CASE(KDACTest, FloatTypes);

TYPED_TEST(KDACTest, Fit) {
  this->kdac_->Fit(this->data_matrix_);
}
