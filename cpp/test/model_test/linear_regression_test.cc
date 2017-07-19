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
//
// This file tests the functionality of the Linear_Regression.h program that
// develops Linear Regression modeling by using Maximum Liklihood Estimation.
// This file will also test the function for Gradient descent for the Linear
// Regression modeling.

#include <iostream>
#include <cmath>
#include <vector>
#include "Eigen/Dense"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/linear_regression.h"
#include "gtest/gtest.h"

template<typename T>
class LinearRegressionTest : public ::testing::Test {
 public:
  Nice::Matrix<T> inputX;
  Nice::Matrix<T> inputY;
  Nice::Vector<T> theta;
  Nice::Vector<T> thetaCalc;
  Nice::LinearRegression<T> lr;
  T sr;

  void GetTheta() {
    lr.Fit(inputX, inputY);
    theta = lr.getTheta();
  }
};

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(LinearRegressionTest, MyTypes);

TYPED_TEST(LinearRegressionTest, FunctionalityMLE) {
  this->inputX.resize(5, 5);
  this->inputX << 0.442753, 1.198629, 0.179202, 0.770854, -1.336541,
                  -0.473156, -1.401062, 1.328741, -0.161256, -0.048996,
                  -0.737971, 1.391194, 0.524369, 1.219123, 0.248429,
                  0.058306, 0.078564, -0.001183, 0.221059, 1.193412,
                  0.085405, -0.124505, 0.536142, -0.069715, 0.418620;
  this->inputY.resize(5, 1);
  this->inputY << 32.925552,
                  -35.138399,
                  163.775170,
                  124.143934,
                  52.464189;
  this->theta.resize(5, 1);
  this->lr.setAlgorithm(Nice::MLE);
  this->GetTheta();
  this->thetaCalc.resize(5, 1);
  this->thetaCalc << 52.831916,
                     36.726816,
                     44.855375,
                     88.258825,
                     82.721425;
  Nice::Vector<TypeParam> s;
  s = this->theta - this->thetaCalc;
  Nice::Vector<TypeParam> r;
  r = s * s;
  this->sr = r.sum();
  EXPECT_LT(this->sr, 0.00001);
}

TYPED_TEST(LinearRegressionTest, FuncionalityGD) {
this->inputX.resize(5, 5);
  this->inputX << 0.442753, 1.198629, 0.179202, 0.770854, -1.336541,
                  -0.473156, -1.401062, 1.328741, -0.161256, -0.048996,
                  -0.737971, 1.391194, 0.524369, 1.219123, 0.248429,
                  0.058306, 0.078564, -0.001183, 0.221059, 1.193412,
                  0.085405, -0.124505, 0.536142, -0.069715, 0.418620;
  this->inputY.resize(5, 1);
  this->inputY << 32.925552,
                  -35.138399,
                  163.775170,
                  124.143934,
                  52.464189;
  this->theta.resize(5, 1);
  this->lr.setAlgorithm(Nice::GD);
  this->GetTheta();
  this->thetaCalc.resize(5, 1);
  this->thetaCalc << 52.831916,
                     36.726816,
                     44.855375,
                     88.258825,
                     82.721425;
  Nice::Vector<TypeParam> s;
  s = this->theta - this->thetaCalc;
  Nice::Vector<TypeParam> r;
  r = s * s;
  this->sr = r.sum();
  EXPECT_LT(this->sr, 0.00001);
}
