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
    seta();
    setY();
    //had setA and setY here...instead of setting them by functions should I set the class members to the parameters?
    Norma(); //Feature Normalization //is this needed for gradient descent?
    settheta(); //is this needed for gradient descent?
    setdcost_history(); //doesnt return anything...do i need? uses iterations
    setcost_history(); //doesnt return anything...do I need? Uses iterations
    if (algo_ == MLE) {
      return MLE(); //change to no parameters
    } else if (algo_ == GD) {
        return GradientDescent();
    } else { 
        std::cout << "Not valid linear regression algorithm type, enter 0 or 1" << std::endl;
    }
  }
  Vector<T> Predict() {
    int i = a.rows();
    int j = a.cols();
    X.resize(i, j+1);
    X.col(0).setOnes();
    for(int n = 0; n < xval.cols(); n++) {
      X.col(n+1) = xval.col(n);
    }
    newY = theta.transpose() * X.transpose();
    return newY;
  }
  void Norma() {
    for(int b = 0; b < a.cols(); b++) {
      for(int c = 0; c < a.rows(); c++) {
        a(c, b) = (xval(c, b) - xval.mean())/(xval.maxCoeff() - xval.minCoeff());
      }
    }
  }
  Vector<T> HypothesisFunction() {
    int i = a.rows();
    int j = a.cols();
    X.resize(i, j+1);
    X.col(0).setOnes();
    for(int n = 0; n < a.cols(); n++) {
      X.col(n+1) = a.col(n);
    }
    Vector<T> hf;
    hf.resize(i);
    hf = theta.transpose() * X.transpose();
    return hf;
}
  float DCostFunction() { //Derivative of Cost Function
    Vector<T> cost;
    cost.resize(a.rows());
    Vector<T> hf;
    hf.resize(a.rows());
    for(int m = 0; m < a.rows(); m++) {
      hf = HypothesisFunction(); 
      cost(m) = (hf(m) - Y(m));
    }
    float final_error = cost.sum();
    final_error *= (1.0/a.rows());
    return final_error;
   }

  float CostFunction() {
    Vector<T> cost;
    cost.resize(a.rows());
    Vector<T> hf;
    hf.resize(a.rows());
    for(int m = 0; m < a.rows(); m++) {
      hf = HypothesisFunction();
      cost(m) = pow((hf(m) - Y(m)), 2);
    }
    float final_error = cost.sum();
    final_error *= (1.0/(2*a.rows()));
    return final_error;
}
  Vector<T> MaximumLikelihoodEstimation(){ //need to ask Shi what is A, X, and Y?? 
    Matrix<T> xTransposed = X.transpose();
    Matrix<T> bPart1 = xTransposed*X;
    Matrix<T> bPart1Inv = bPart1.inverse();
    Vector<T> bPart2 = xTransposed*Y;
    Vector<T> b = bPart1Inv*bPart2;
    return b; //b is theta for MLE should I use the theta class member?
  }
  Vector<T> GradientDescent() {
    int v = 1;
    float delta = 10;
    while((delta > pow(10, -10)) & (v < iterations)) {
      cost_history(0) = CostFunction();
      dcost_history(0) = DCostFunction();
        for(int j = 0; j < X.cols(); j++) {
          for(int i = 0; i < X.rows(); i++) {
            theta(j) -= alpha * DCostFunction() * X(i, j);
          }
        }
        dcost_history(v) = DCostFunction();
        cost_history(v) = CostFunction();
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
  Matrix<T> a;
  Matrix<T> xval;
  void seta() {
    a.resize(xval.rows(), xval.cols()); //where does xval and yval come from??
  }
  Matrix<T> X;
  Matrix<T> Y;
  Matrix<T> yval;
  void setY() {
    Y.resize(yval.rows(), yval.cols());
    Y = yval;
  }
  Vector<T> theta;
  void settheta() {
   theta.resize(a.cols() +1);
   theta.setOnes();
  }
  Vector<T> dcost_history;
  void setdcost_history() {
   dcost_history.resize(iterations);
   dcost_history.setZero();
  }
  Vector<T> cost_history;
  void setcost_history() {
    cost_history.resize(iterations);
    cost_history.setZero();
  }
  Vector<T> hf;
  Vector<T> newY;
  Vector<T> final_theta;
  float alpha;
  int iterations;
};
}
#endif
