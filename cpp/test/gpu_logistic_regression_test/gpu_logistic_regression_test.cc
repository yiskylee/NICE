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
#include "include/gpu_logistic_regression.h"

template<typename T>
class GpuLogisticRegressionTest: public ::testing::Test {
 public:
  int iterations;
  T alpha;
  Nice::Matrix<T> training_x;
  Nice::Vector<T> training_y;
  Nice::Matrix<T> predict_x;
  Nice::Vector<T> predictions;
  Nice::GpuLogisticRegression<T> testModel1;
  Nice::GpuLogisticRegression<T> testModel2;
};

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(LogisticRegressionTest, MyTypes);

// Runs both the fit and predict function on a single model.
TYPED_TEST(GpuLogisticRegressionTest, MatrixLogisticRegressionOneModel) {
  // Setup for the Fit function
  this->training_x.resize(10, 2);
  this->iterations = 10000;
  this->alpha = 0.3;
  this->training_x << 2.781, 2.550,
          1.465, 2.362,
          3.396, 4.400,
          1.388, 1.850,
          3.064, 3.005,
          7.627, 2.759,
          5.332, 2.088,
          6.922, 1.771,
          8.675, -0.242,
          7.673, 3.508;
  this->training_y.resize(10);
  this->training_y << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1;
  this->testModel1.Fit(this->training_x, this->training_y, this->iterations,
    this->alpha);

  // Setup for the Predict function
  this->predict_x.resize(10, 2);
  this->predict_x << 2.781, 2.550,
          1.465, 2.362,
          3.396, 4.400,
          1.388, 1.850,
          3.064, 3.005,
          7.627, 2.759,
          5.332, 2.088,
          6.922, 1.771,
          8.675, -0.242,
          7.673, 3.508;
  this->predictions = this->testModel1.Predict(this->predict_x);
  this->predictions.resize(10);
  std::cout << this->predictions << std::endl;
  ASSERT_TRUE(true);
}

// Runs both the fit and predict function on two separate models in
// the same test.
TYPED_TEST(GpuLogisticRegressionTest, MatrixLogisticRegressionTwoModels) {
  // Setup for Model 1's Fit function
  this->training_x.resize(10, 2);
  this->iterations = 10000;
  this->alpha = 0.3;
  this->training_x << 2.781, 2.550,
          1.465, 2.362,
          3.396, 4.400,
          1.388, 1.850,
          3.064, 3.005,
          7.627, 2.759,
          5.332, 2.088,
          6.922, 1.771,
          8.675, -0.242,
          7.673, 3.508;
  this->training_y.resize(10);
  this->training_y << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1;
  this->testModel1.Fit(this->training_x, this->training_y, this->iterations,
    this->alpha);

  // Setup for Model 2's Fit function
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
  this->testModel2.Fit(this->training_x, this->training_y,
    this->iterations, this->alpha);

  // Setup for Model 1's Predict function
  this->predict_x.resize(10, 2);
  this->predict_x << 2.781, 2.550,
          1.465, 2.362,
          3.396, 4.400,
          1.388, 1.850,
          3.064, 3.005,
          7.627, 2.759,
          5.332, 2.088,
          6.922, 1.771,
          8.675, -0.242,
          7.673, 3.508;
  this->predictions = this->testModel1.Predict(this->predict_x);

  // Setup for Model 2's Fit function
  this->predictions.resize(10);
  std::cout << this->predictions << std::endl;
  this->predict_x << 2, .5,
               2, 0,
               4, 1,
               5, 2,
               7, 3,
               1, 3,
               2, 2,
               4, 3,
               3, 5,
               6, 3.5;
  this->predictions = this->testModel2.Predict(this->predict_x);
  std::cout << this->predictions << std::endl;
  ASSERT_TRUE(true);
}
