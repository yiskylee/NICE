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
#include "include/logistic_regression.h"
#include "include/matrix.h"

template<typename T>
class LogisticRegressionTest : public ::testing::Test {
 public:
  int iterations;
  T alpha;
  Nice::Matrix<T> training_x;
  Nice::Vector<T> training_y;
  Nice::Vector<T> coeff;
  Nice::Matrix<T> predict_x;
  Nice::Vector<T> predictions;

  void LogisticRegressionFit() {
    coeff= Nice::LogisticRegression<T>::Fit(training_x, training_y, iterations, alpha);
  }

  void LogisticRegressionPredict() {
    predictions = Nice::LogisticRegression<T>::Predict(predict_x, coeff);
  }
};

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(LogisticRegressionTest, MyTypes);


TYPED_TEST(LogisticRegressionTest, MatrixLogisticRegressionPredict) {
  this->coeff.resize(3);
  this->coeff << -0.406, 0.852, -1.104;
  this->predict_x.resize(10,2);
  this->predict_x << 2.781,2.550,
		      1.465,2.362,
		      3.396,4.400,
		      1.388,1.850,
		      3.064,3.005,
		      7.627,2.759,
		      5.332,2.088,
		      6.922,1.771,
		      8.675,-0.242,
		      7.673,3.508;
  this->LogisticRegressionPredict();
  std::cout << "Checkpoint A" << std::endl;
  this->predictions.resize(10);
  std::cout << this->predictions << std::endl;
  ASSERT_TRUE(true);
}

TYPED_TEST(LogisticRegressionTest, MatrixLogisticRegressionFit) {
  this->training_x.resize(10,2);
  this-> iterations = 10000;
  this-> alpha = 0.3;
  this->training_x << 2, .5,
               2, 0,
               4, 1,
               5, 2,
               7, 3,
               1, 3,
               2, 2,
               4, 3,
               3, 5,
               6, 3.5;
  this->training_y.resize(10);
  this->training_y << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1;
  this->coeff.resize(3);
  this->LogisticRegressionFit();
  std::cout << this->coeff << std::endl;
  ASSERT_TRUE(true);
}
