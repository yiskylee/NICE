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

#ifndef CPP_INCLUDE_LINEAR_REGRESSION_H
#define CPP_INCLUDE_LINEAR_REGRESSION_H

#include <iostream>
#include <vector>
#include <cmath>
#include "include/matrix.h"
#include "include/vector.h"
#include "Eigen/Dense"

namespace Nice {

enum LinearRegressionAlgo {
  MLE = 0,
  GD
}

// LinearRegression lr(MLE);
template<typename T>
class LinearRegression {
 public:
  LinearRegression()  
  :
  algo_(MLE) {}
  LinearRegression(LinearRegressionAlgo algo)
  :
  algo_(algo) {} 
  Vector <T> Fit(Matrix<T> &a, Matrix<T> &y) { 
    alpha = 0.0001;
    iterations = 230;
    settheta(X); 
    if (algo_ == MLE) {
      return MLE(X, Y); 
    } else if (algo_ == GD) {
        return GradientDescent(X, Y);
    } else { 
        std::cout << "Not valid linear regression algorithm type, enter 0 or 1" << std::endl;
    }
  }
  Vector<T> Predict(Matrix<T> &X) {
    Vector<T> newY;
    newY.resize(X.rows());
    newY = theta.transpose() * X.transpose();
    return newY;
  }
  Vector<T> HypothesisFunction(Matrix<T> &X) {
    int i = X.rows();
    Vector<T> hf;
    hf.resize(i);
    hf = theta.transpose() * X.transpose();
    return hf;
}
  float DCostFunction(Matrix<T> &X, Matrix<T> &Y) {
    for (int k = 0; k < iterations; k++) {          
      delta = 2 / ((double) X.rows()) * (X.transpose() * (X * theta - Y));
      theta = (theta - (alpha * delta));
    }
    return theta;
}
  float CostFunction(Matrix<T> &X, Matrix<T> &Y) {
    Vector<T> cost;
    cost.resize(X.rows());
    Vector<T> hf;
    hf.resize(X.rows());
    for(int m = 0; m < X.rows(); m++) {
      hf = HypothesisFunction(X);
      cost(m) = pow((hf(m) - Y(m)), 2);
    }
    float final_error = cost.sum();
    final_error *= (1.0/(2*X.rows()));
    return final_error;
}
  Vector<T> MaximumLikelihoodEstimation(Matrix<T> &X, Matrix<T> &Y) { 
    Matrix<T> xTransposed = X.transpose();
    Matrix<T> thetaPart1 = xTransposed*X;
    Matrix<T> thetaPart1Inv = thetaPart1.inverse();
    Vector<T> thetaPart2 = xTransposed*Y;
    theta = thetaPart1Inv*thetaPart2;
    return theta; 
  }
  Vector<T> GradientDescent(Matrx<T> &X, Matrix<T> &Y) {
    int v = 1;
    float delta = 10;
    Vector<T> cost_history;
    cost_history.resize(iterations);
    cost_history.setZero();
    Vector<T> dcost_history;
    dcost_history.resize(iterations);
    dcost_history.setZero();
    while((delta > pow(10, -10)) & (v < iterations)) {
      cost_history(0) = CostFunction(X, Y);
      dcost_history(0) = DCostFunction(X, Y);
        for(int j = 0; j < X.cols(); j++) {
          for(int i = 0; i < X.rows(); i++) {
            theta(j) -= alpha * DCostFunction(X, Y) * X(i, j);
          }
        }
        dcost_history(v) = DCostFunction(X, Y);
        cost_history(v) = CostFunction(X, Y);
        delta = (cost_history(v-1) - cost_history(v));
        v++;
      }
        std::cout<<"Done!"<<std::endl;
        std::cout<<"This took "<<v<<" iterations."<<std::endl;
        std::cout<<"The values of theta are: "<<"\n"<<theta<<std::endl;
        return theta;
      }
 private:
  LinearRegressionAlgo algo_;
  Vector<T> theta;
  void settheta(Matrix<T> &X) { 
   theta.resize(X.cols());
   theta.setOnes();
  }
  float alpha;
  int iterations;
};
}
#endif
