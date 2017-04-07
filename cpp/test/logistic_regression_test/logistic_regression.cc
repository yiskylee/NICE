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
#include "gtest/gtest.h"
#include "include/logistic_regression.h"
#include "include/matrix.h"

template<class T>
class LogisticRegressionTest : public ::testing::Test {
 public:
  Nice::LogisticRegression<T> *lr;
  Nice::Matrix<T> x;
  Nice::Vector<T> y;
  Nice::Vector<T> theta_output;
  void make_instance() {
    lr = new Nice::LogisticRegression<T>();
  }
};

typedef ::testing::Types<float> MyTypes;
TYPED_TEST_CASE(LogisticRegressionTest, MyTypes);


TYPED_TEST(LogisticRegressionTest, Test) {
std::cout << "Checkpoint A" << std::endl;
  this->x.resize(10, 2);
    this->x << 2, .5,
               2, 0,
               4, 1,
               5, 2,
               7, 3,
               1, 3,
               2, 2,
               4, 3,
               3, 5,
               6, 3.5;
  std::cout << "Checkpoint B" << std::endl;
  
  // POSSIBLE SOURCE OF ERROR IN INITALIZING MATRIX
  this->y.resize(10);
  this->theta_output.resize(2);
  this->y << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1;
  this->theta_output << 10.0,10.0;
  
  
  std::cout << "Checkpoint C" << std::endl;
  this->make_instance();
  this->theta_output = this->lr->logistic_regression(this->x,this->y,this->theta_output);
  std::cout << this->theta_output << std::endl;
  // this->lr->SetLambda(.0001);
  // this->lr->Fit(this->x, this->y);
  // this->predict_input.resize(2,2);
  // this->predict_input << 8, 3,
                         //2, 11;
  // this->Predict();
  // std::cout << "predicted output:" << this->predicted << std::endl;
  ASSERT_TRUE( 0 == 0);
}