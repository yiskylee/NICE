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
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/gpu_logistic_regression.h"
#include "include/logistic_regression.h"
#include "include/matrix.h"
#include <chrono>
using namespace std::chrono;

template<typename T>
class GpuLogisticRegressionTest: public ::testing::Test {
 public:
  int iterations;
  T alpha;
  Nice::Matrix<T> training_x_;
  Nice::Vector<T> training_y_;
  Nice::Matrix<T> test_x_;
  Nice::Vector<T> predictions_;
  Nice::Vector<T> gpu_predictions_;
  Nice::Vector<T> test_y_;
  Nice::LogisticRegression<T> model_;
  Nice::GpuLogisticRegression<T> gpu_model_;

  // Loads data from file
  Nice::Matrix<T> Filler(std::string fileName, std::string d) {
    std::string folder = "../test/data_for_test/logistic_regression_benchmark/";
    std::string testName = ::testing::UnitTest::GetInstance()->
      current_test_info()->name();
    return Nice::util::FromFile<T>(folder + testName + "/" + fileName, d);
  }

  // Function to check input against the expected result
  void ResultsCheck(Nice::Vector<T> results, std::string type) {
    int correct = 0;
    int total = results.size();
    for (int i = 0; i < results.size(); i++) {
      if ((results(i) <= 0.5 && test_y_(i) <= 0.5) || (results(i) > 0.5 &&
        test_y_(i) > 0.5)) {
        correct++;
      }
    }
    printf("The %s model_ predicts %i / %i correctly or with %2.3f accuracy\n",
      type.c_str(), correct, total, correct / static_cast<float>(total));
  }

  void ThetaCompare(Nice::Vector<T> cpu, Nice::Vector<T> gpu) {
    for (int i = 0; i < cpu.size(); i++){
      if (i >50 && i < 70) {
        std::cout << "CPU: " << cpu(i) << " GPU: " << gpu(i) << "\n";
      }
      EXPECT_NEAR(cpu(i), gpu(i), 0.001);
    }
  }
};

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(GpuLogisticRegressionTest, MyTypes);

TYPED_TEST(GpuLogisticRegressionTest, Basic) {
  // Setup for the Fit function
  this->training_x_.resize(10, 2);
  this->iterations = 10000;
  this->alpha = 0.01;
  // Populates matrix with values from txt files
  this->training_x_ << 2.781, 2.550,
        1.465, 2.362,
        3.396, 4.400,
        1.388, 1.850,
        3.064, 3.005,
        7.627, 2.759,
        5.332, 2.088,
        6.922, 1.771,
        8.675, -0.242,
        7.673, 3.508;
  this->training_y_.resize(10);
  this->training_y_ << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1;

  this->test_y_.resize(10);
  this->test_y_ << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1;

  this->test_x_.resize(10, 2);
  this->test_x_ << 2.781, 2.550,
       1.465, 2.362,
       3.396, 4.400,
       1.388, 1.850,
       3.064, 3.005,
       7.627, 2.759,
       5.332, 2.088,
       6.922, 1.771,
       8.675, -0.242,
       7.673, 3.508;
  // CPU Fit with timing functionality around it
  // this->model_.SetIterations(this->iterations);
  // this->model_.SetAlpha(this->alpha);
  this->model_.SetIterations(this->iterations);
  this->model_.SetAlpha(this->alpha);
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  this->model_.Fit(this->training_x_, this->training_y_);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "CPU Logistic Regression - Fit: " << static_cast<float>(duration)
    << std::endl;


  // GPU Fit with timing functionality around it
  this->gpu_model_.SetIterations(this->iterations);
  this->gpu_model_.SetAlpha(this->alpha);
  t1 = high_resolution_clock::now();
  this->gpu_model_.GpuFit(this->training_x_, this->training_y_);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "GPU Logistic Regression - Fit: " << static_cast<float>(duration)
    << std::endl;

  // CPU Predict function with timing
  t1 = high_resolution_clock::now();
  this->predictions_ = this->model_.Predict(this->test_x_);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "CPU Logistic Regression - Predict: " <<
    static_cast<float>(duration) << std::endl;

  // GPU Predict function witgpu_modelh timing
  t1 = high_resolution_clock::now();
  this->gpu_predictions_ = this->gpu_model_.GpuPredict(this->test_x_);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "GPU Logistic Regression - Predict: " <<
    static_cast<float>(duration) << std::endl;

  // Compares CPU and GPU values against ground truth
  this->ResultsCheck(this->gpu_predictions_, "GPU");
  this->ResultsCheck(this->predictions_, "CPU");

  // Compares the CPU and GPU theta value with each other
  this->ThetaCompare(this->model_.GetTheta(), this->gpu_model_.GetTheta());

  // Prints out the first 20 values of predict vectors
  for (int i = 0; i < 10; i++) {
    std::cout << this->gpu_predictions_(i) << " :: " << this->predictions_(i)
      << std::endl;
  }
}

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(Benchmark, MyTypes);


TYPED_TEST(GpuLogisticRegressionTest, Heart) {
  // Setup for the Fit function
  this->iterations = 1000;
  this->alpha = 0.001;

  // Populates matrix with values from txt files
  this->training_x_ = this->Filler("heart_x.txt", ",");
  this->training_y_ = this->Filler("heart_y.txt", " ");
  this->test_x_ = this->Filler("heart_predict.txt", ",");

  // CPU Fit with timing functionality around it
  // this->model_.SetIterations(this->iterations);
  // this->model_.SetAlpha(this->alpha);
  this->model_.SetIterations(this->iterations);
  this->model_.SetAlpha(this->alpha);
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  this->model_.Fit(this->training_x_, this->training_y_);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "CPU Logistic Regression - Fit: " << static_cast<float>(duration)
    << std::endl;


  // GPU Fit with timing functionality around it
  this->gpu_model_.SetIterations(this->iterations);
  this->gpu_model_.SetAlpha(this->alpha);
  t1 = high_resolution_clock::now();
  this->gpu_model_.GpuFit(this->training_x_, this->training_y_);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "GPU Logistic Regression - Fit: " << static_cast<float>(duration)
    << std::endl;

  // CPU Predict function with timing
  t1 = high_resolution_clock::now();
  this->predictions_ = this->model_.Predict(this->test_x_);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "CPU Logistic Regression - Predict: " <<
    static_cast<float>(duration) << std::endl;

  // GPU Predict function witgpu_modelh timing
  t1 = high_resolution_clock::now();
  this->gpu_predictions_ = this->gpu_model_.GpuPredict(this->test_x_);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "GPU Logistic Regression - Predict: " <<
    static_cast<float>(duration) << std::endl;

  // Compares CPU and GPU values against ground truth
  this->test_y_ = this->Filler("heart_expected.txt", " ");
  this->ResultsCheck(this->gpu_predictions_, "GPU");
  this->ResultsCheck(this->predictions_, "CPU");

  // Compares the CPU and GPU theta value with each other
  this->ThetaCompare(this->model_.GetTheta(), this->gpu_model_.GetTheta());

  // Prints out the first 20 values of predict vectors
  for (int i = 0; i < 20; i++) {
    std::cout << this->gpu_predictions_(i) << " :: " << this->predictions_(i)
      << std::endl;
  }
}

TYPED_TEST(GpuLogisticRegressionTest, MNIST) {
  // Setup for the Fit function
  this->iterations = 100;
  this->alpha = 0.001;

  // Populates matrix with values from txt files
  this->training_x_ = this->Filler("mnist_x.txt", ",");
  this->training_y_ = this->Filler("mnist_y.txt", " ");
  this->test_x_ = this->Filler("mnist_predict.txt", ",");

  // CPU Fit with timing functionality around it
  // this->model_.SetIterations(this->iterations);
  // this->model_.SetAlpha(this->alpha);
  this->model_.SetIterations(this->iterations);
  this->model_.SetAlpha(this->alpha);
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  this->model_.Fit(this->training_x_, this->training_y_);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "CPU Logistic Regression - Fit: " << static_cast<float>(duration)
    << std::endl;


  // GPU Fit with timing functionality around it
  this->gpu_model_.SetIterations(this->iterations);
  this->gpu_model_.SetAlpha(this->alpha);
  t1 = high_resolution_clock::now();
  this->gpu_model_.GpuFit(this->training_x_, this->training_y_);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "GPU Logistic Regression - Fit: " << static_cast<float>(duration)
    << std::endl;

  // CPU Predict function with timing
  t1 = high_resolution_clock::now();
  this->predictions_ = this->model_.Predict(this->test_x_);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "CPU Logistic Regression - Predict: " <<
    static_cast<float>(duration) << std::endl;

  // GPU Predict function witgpu_modelh timing
  t1 = high_resolution_clock::now();
  this->gpu_predictions_ = this->gpu_model_.GpuPredict(this->test_x_);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "GPU Logistic Regression - Predict: " <<
    static_cast<float>(duration) << std::endl;

  // Compares CPU and GPU values against ground truth
  this->test_y_ = this->Filler("mnist_expected.txt", " ");
  this->ResultsCheck(this->gpu_predictions_, "GPU");
  this->ResultsCheck(this->predictions_, "CPU");

  // Compares the CPU and GPU theta value with each other
  this->ThetaCompare(this->model_.GetTheta(), this->gpu_model_.GetTheta());

  // Prints out the first 20 values of predict vectors
  for (int i = 0; i < 20; i++) {
    std::cout << this->gpu_predictions_(i) << " :: " << this->predictions_(i)
      << std::endl;
  }
}
