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
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/gpu_logistic_regression.h"
#include "include/logistic_regression.h"
#include "include/matrix.h"

template<typename T>
class Benchmark: public ::testing::Test {
 public:
  int iterations;
  T alpha;
  Nice::Matrix<T> training_x;
  Nice::Vector<T> training_y;
  Nice::Matrix<T> predict_x;
  Nice::Vector<T> predictions;
  Nice::Vector<T> expected_vals;
  Nice::LogisticRegression<T> model;

  Nice::Matrix<T> filler(std::string fileName, std::string d){
    std::string folder = "../test/data_for_test/LogisticRegressionBenchmark/";
    std::string testName = ::testing::UnitTest::GetInstance()->
      current_test_info()->name();
    return Nice::util::FromFile<T>(folder + testName + "/" + fileName, d);
  }
};

typedef ::testing::Types<float> MyTypes;
TYPED_TEST_CASE(Benchmark, MyTypes);

// Runs both the fit and predict function on a single model.
TYPED_TEST(Benchmark, Heart) {
  // Setup for the Fit function
  //this->training_x.resize(10, 2);
  this->iterations = 1000000;
  this->alpha = 0.001;
  this->training_x = this->filler("heart_x.txt", ",");

  //std::cout << this->training_x << "\n";//this->training_y.resize(10);
  this->training_y = this->filler("heart_y.txt", " ");
  this->model.Fit(this->training_x, this->training_y, this->iterations,
    this->alpha);
  // Setup for the Predict function
  this->predict_x = this->filler("heart_predict.txt", ",");
  this->predictions = this->model.Predict(this->predict_x);
  this->expected_vals = this->filler("heart_expected.txt", " ");
  //std::cout << this->predictions << std::endl;
  int total, correct;
  correct = total = this->predictions.size();
  for (int i = 0; i < this->predictions.size(); i++){
    if (std::abs(this->predictions(i) - this->expected_vals(i)) >= 0.5){
      correct--;
    }
  }
  printf("The model predicts %i / %i correctly or with %2.3f accuracy\n",
    correct, total, correct / (float)total);
  ASSERT_TRUE(true);
}
