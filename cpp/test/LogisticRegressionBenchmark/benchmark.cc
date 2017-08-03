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
  Nice::Vector<T> gpuPredictions;
  Nice::Vector<T> expected_vals;
  Nice::LogisticRegression<T> model;
  Nice::GpuLogisticRegression<T> gpuModel;
  Nice::Matrix<T> filler(std::string fileName, std::string d){
    std::string folder = "../test/data_for_test/LogisticRegressionBenchmark/";
    std::string testName = ::testing::UnitTest::GetInstance()->
      current_test_info()->name();
    return Nice::util::FromFile<T>(folder + testName + "/" + fileName, d);
  }
  void resultsCheck(Nice::Vector<T> results, std::string type){
    int correct = 0;
    int total = results.size();
    for (int i = 0; i < results.size(); i++){
      if ((results(i) <= 0.5 && expected_vals(i) <= 0.5) || (results(i) > 0.5 && expected_vals(i) > 0.5)){
        correct++;
      }
    }
    printf("The %s model predicts %i / %i correctly or with %2.3f accuracy\n",
      type.c_str(), correct, total, correct / (float)total);
  }
};

typedef ::testing::Types<float> MyTypes;
TYPED_TEST_CASE(Benchmark, MyTypes);

// Runs both the fit and predict function on a single model.
TYPED_TEST(Benchmark, Heart) {
  // Setup for the Fit function
  //this->training_x.resize(10, 2);
  this->iterations = 10000;
  this->alpha = 0.001;
  this->training_x = this->filler("heart_x.txt", ",");

  //std::cout << this->training_x << "\n";//this->training_y.resize(10);
  this->training_y = this->filler("heart_y.txt", " ");
  this->model.Fit(this->training_x, this->training_y, this->iterations,
    this->alpha);
  //std::cout << this->model.getTheta() << "\n";
  this->gpuModel.GpuFit(this->training_x, this->training_y, this->iterations,
    this->alpha);
  this->gpuModel.setTheta(this->model.getTheta());
  // Setup for the Predict function
  this->predict_x = this->filler("heart_predict.txt", ",");
  this->predictions = this->model.Predict(this->predict_x);
  this->gpuPredictions = this->gpuModel.GpuPredict(this->predict_x);
  this->expected_vals = this->filler("heart_expected.txt", " ");
  //std::cout << this->predictions << std::endl;
  this->resultsCheck(this->gpuPredictions, "GPU");
  this->resultsCheck(this->predictions, "CPU");
  ASSERT_TRUE(true);
}

TYPED_TEST(Benchmark, MNIST) {
  // Setup for the Fit function
  //this->training_x.resize(10, 2);
  this->iterations = 10000;
  this->alpha = 0.001;
  this->training_x = this->filler("mnist_x.txt", ",");
  this->training_y = this->filler("mnist_y.txt", " ");
  std::cout << "Fitting the data" << "\n";
  this->model.Fit(this->training_x, this->training_y, this->iterations,
    this->alpha);
  //std::cout << this->model.getTheta() << "\n";
  //this->gpuModel.GpuFit(this->training_x, this->training_y, this->iterations,
  //  this->alpha);
  this->gpuModel.setTheta(this->model.getTheta());
  // Setup for the Predict function
  this->predict_x = this->filler("mnist_predict.txt", ",");
  std::cout << "Predicting the data" << "\n";

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  this->predictions = this->model.Predict(this->predict_x);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>( t2 - t1 ).count();
  std::cout << "CPU Logistic Regression - Predict: " << (long)duration << std::endl;

  this->gpuPredictions = this->gpuModel.GpuPredict(this->predict_x);
  this->expected_vals = this->filler("mnist_expected.txt", " ");
  //std::cout << this->predictions << std::endl;
  this->resultsCheck(this->gpuPredictions, "GPU");
  this->resultsCheck(this->predictions, "CPU");
  ASSERT_TRUE(true);
}
