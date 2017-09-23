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

  // Loads data from file
  Nice::Matrix<T> filler(std::string fileName, std::string d){
    std::string folder = "../test/data_for_test/LogisticRegressionBenchmark/";
    std::string testName = ::testing::UnitTest::GetInstance()->
      current_test_info()->name();
    return Nice::util::FromFile<T>(folder + testName + "/" + fileName, d);
  }

  // Function to check input against the expected result
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

  void thetaCompare(Nice::Vector<T> cpu, Nice::Vector<T> gpu){
    for (int i = 0; i < cpu.size(); i++){
      if (i > 30 && i < 50){
        std::cout << "CPU: " << cpu(i) << " GPU: " << gpu(i) << "\n";
      }
      EXPECT_NEAR(cpu(i), gpu(i), 0.001);
    }
  }
};

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(Benchmark, MyTypes);


TYPED_TEST(Benchmark, Heart) {
  // Setup for the Fit function
  this->iterations = 25000;
  this->alpha = 0.001;

  // Populates matrix with values from txt files
  this->training_x = this->filler("heart_x.txt", ",");
  this->training_y = this->filler("heart_y.txt", " ");
  this->predict_x = this->filler("heart_predict.txt", ",");

  // CPU Fit with timing functionality around it
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  Nice::Vector<TypeParam> cpu = this->model.Fit(this->training_x, this->training_y, this->predict_x,
    this->iterations, this->alpha);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>( t2 - t1 ).count();
  std::cout << "CPU Logistic Regression - Fit: " << (long)duration << std::endl;

  // GPU Fit with timing functionality around it
  t1 = high_resolution_clock::now();
  Nice::Vector<TypeParam> gpu = this->gpuModel.GpuFitMV(this->training_x, this->training_y, this->predict_x,
    this->iterations,this->alpha);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count();
  std::cout << "GPU Logistic Regression - Fit: " << (long)duration << std::endl;

  // CPU Predict function with timing
  t1 = high_resolution_clock::now();
  this->predictions = this->model.Predict(this->predict_x);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count();
  std::cout << "CPU Logistic Regression - Predict: " << (long)duration << std::endl;

  // GPU Predict function with timing
  t1 = high_resolution_clock::now();
  this->gpuPredictions = this->gpuModel.GpuPredict(this->predict_x);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count();
  std::cout << "GPU Logistic Regression - Predict: " << (long)duration << std::endl;

  // Compares CPU and GPU values against ground truth
  this->expected_vals = this->filler("heart_expected.txt", " ");
  this->resultsCheck(this->gpuPredictions, "GPU");
  this->resultsCheck(this->predictions, "CPU");

  // Compares the CPU and GPU theta value with each other
  this->thetaCompare(this->model.getTheta(), this->gpuModel.getTheta());
  std::cout << "Number of differences between CPU and GPU Thetas : " << ((cpu - gpu).squaredNorm()) << "\n";

  // Prints out the first 20 values of predict vectors
  for (int i = 0; i < 20; i++){
    std::cout << this->gpuPredictions(i) << " :: " << this->predictions(i) << std::endl;
  }
}

TYPED_TEST(Benchmark, MNIST) {
  // Setup for the Fit function
  this->iterations = 30;
  this->alpha = 0.01;

  // Populates matrix with values from txt files
  this->training_x = this->filler("mnist_x.txt", ",");
  this->training_y = this->filler("mnist_y.txt", " ");
  this->predict_x = this->filler("mnist_predict.txt", ",");

  // CPU Fit with timing functionality around it
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  Nice::Vector<TypeParam> cpu = this->model.Fit(this->training_x, this->training_y, this->predict_x,
    this->iterations, this->alpha);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>( t2 - t1 ).count();
  std::cout << "CPU Logistic Regression - Fit: " << (long)duration << std::endl;

  // CPU Predict with timing functionality around it
  t1 = high_resolution_clock::now();
  this->predictions = this->model.Predict(this->predict_x);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count();
  std::cout << "CPU Logistic Regression - Predict: " << (long)duration << std::endl;

  // GPU Fit with timing functionality around it
  t1 = high_resolution_clock::now();
  Nice::Vector<TypeParam> gpu = this->gpuModel.GpuFitMV(this->training_x, this->training_y, this->predict_x,
    this->iterations, this->alpha);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count();
  std::cout << "GPU Logistic Regression - Fit: " << (long)duration << std::endl;

  // GPU Predict with timing functionality around it
  t1 = high_resolution_clock::now();
  this->gpuPredictions = this->gpuModel.GpuPredict(this->predict_x);
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count();
  std::cout << "GPU Logistic Regression - Predict: " << (long)duration << std::endl;

  // Checks prediction results against ground truth
  this->expected_vals = this->filler("mnist_expected.txt", " ");
  // this->resultsCheck(this->gpuPredictions, "GPU");
  // this->resultsCheck(this->predictions, "CPU");

  // Compares the theta values generated from CPU and GPU
  this->thetaCompare(this->model.getTheta(), this->gpuModel.getTheta());

  // Prints out the number of errors
  std::cout << "Errors: " << ((cpu - gpu).squaredNorm()) << "\n";

  // Prints out the first 20 values of predict vectors
  for (int i = 0; i < 20; i++){
    std::cout << this->gpuPredictions(i) << " :: " << this->predictions(i) << std::endl;
  }
  ASSERT_TRUE(true);
}

// Below are tests for comparing theta function
/**
TYPED_TEST(Benchmark, Heart1) {
  // Setup for the Fit function
  this->iterations = 26;
  this->alpha = 0.001;
  this->training_x = this->filler("heart_x.txt", ",");
  this->training_y = this->filler("heart_y.txt", " ");
  std::cout << "Fitting the data" << "\n";

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  this->model.Fit(this->training_x, this->training_y, this->iterations,
    this->alpha);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>( t2 - t1 ).count();
  std::cout << "CPU Logistic Regression - Fit: " << (long)duration << std::endl;
  this->gpuModel.GpuFit(this->training_x, this->training_y, this->iterations,
    this->alpha);
  Nice::Vector<TypeParam> cpuTheta = this->model.getTheta();
  Nice::Vector<TypeParam> gpuTheta = this->gpuModel.getTheta();

  for (int i = 0; i < cpuTheta.rows(); i++){
    //std::cout << cpuTheta(i) << " :: " << gpuTheta(i) << std::endl;
    EXPECT_NEAR(cpuTheta(i), gpuTheta(i), 0.0001);
  }
}

TYPED_TEST(Benchmark, MNIST1) {
  // Setup for the Fit function
  this->iterations = 7;
  this->alpha = 0.001;
  this->training_x = this->filler("mnist_x.txt", ",");
  this->training_y = this->filler("mnist_y.txt", " ");
  std::cout << "Fitting the data" << "\n";

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  this->model.Fit(this->training_x, this->training_y, this->iterations,
    this->alpha);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>( t2 - t1 ).count();
  std::cout << "CPU Logistic Regression - Fit: " << (long)duration << std::endl;
  this->gpuModel.GpuFit(this->training_x, this->training_y, this->iterations,
    this->alpha);
  Nice::Vector<TypeParam> cpuTheta = this->model.getTheta();
  Nice::Vector<TypeParam> gpuTheta = this->gpuModel.getTheta();

  for (int i = 0; i < cpuTheta.rows(); i++){
    //std::cout << cpuTheta(i) << " :: " << gpuTheta(i) << std::endl;
    EXPECT_NEAR(cpuTheta(i), gpuTheta(i), 0.0001);
  }
}
**/
