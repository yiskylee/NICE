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

#ifndef CPP_INCLUDE_LOGISTIC_REGRESSION_H_
#define CPP_INCLUDE_LOGISTIC_REGRESSION_H_

#include <string>
#include <iostream>
#include <cmath>
#include "include/matrix.h"
#include "include/vector.h"
#include "include/kernel_types.h"
#include "Eigen/SVD"
#include "include/svd_solver.h"
#include "include/util.h"

namespace Nice {

// Abstract class of common logistic regression functions
template<typename T>
class LogisticRegression {
 private:
  T Sigmoid(T x){
    return (1 / (1 +
            (-x)));
  }
 public:
  static Vector<T> Predict(const Matrix<T> &inputs, const Vector<T> thetas){
    Vector<T> predictions, yhat;
    Matrix<T> product;

    product.resize(inputs.rows(),inputs.cols());
    product = inputs * thetas.bottomRows(thetas.rows()-1);

    yhat.resize(inputs.rows());
    yhat = product.rowwise().sum();
    yhat = yhat.array() + thetas(0);

    predictions.resize(inputs.rows());

    predictions = ((-1 * yhat).array().exp()) + 1;
    predictions = predictions.array().inverse();
    predictions = predictions.matrix();
    predictions.resize(inputs.rows());
    return predictions;
  }

  /// Calculates the hypothesis of a given input vector
  ///
  ///
  ///
  static Vector<T> h(Vector<T> input){
    input = ((-1 * input).array().exp()) + 1;
    return input.array().inverse();
  }

  /// Calculates the coeffients for a logisitic regression of a given matrix and
  /// returns it as a vector.
  ///
  ///
  static Vector<T> Fit( Matrix<T> &xin, const Vector<T> &y, int iterations, T alpha){
    Vector<T> gradient, theta;
    Matrix<T> x;
    x.conservativeResize(x.rows(), x.cols() + 1);
    std::cout << x << std::endl;
    x.resize(xin.rows(), xin.cols() + 1);
    // TODO Parallelize the shift function
    for(int i = 1; i <= xin.cols(); ++i) {
            x.col(i) = xin.col(i - 1);
    }
    x.col(0).setOnes();
    theta.resize(x.cols());
    theta.setZero();
    gradient.resize(x.rows());
    for (int i = 0; i < iterations; i++){
      gradient = (x.transpose() * (h(x * theta) - y)) / y.size();
      theta = theta - (alpha * gradient);
    }
    return theta;
  }
};
}  // namespace Nice
#endif  // CPP_INCLUDE:_LOGISITC_REGRESSION_H
