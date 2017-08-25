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
#include <chrono>
#include "include/matrix.h"
#include "include/vector.h"
#include "include/kernel_types.h"
#include "Eigen/SVD"
#include "include/svd_solver.h"
#include "include/util.h"


using namespace std::chrono;

namespace Nice {


// Abstract class of common logistic regression functions
template<typename T>
class LogisticRegression {
 private:
  Vector<T> theta;
  /// Calculates the hypothesis of a given input Vector
  ///
  /// \param input
  /// Input Vector
  ///
  /// \return
  /// This function returns a Vector of type T
  Vector<T> h(Vector<T> input) {
    input = ((-1 * input).array().exp()) + 1;
    return input.array().inverse();
  }


 public:
  LogisticRegression() {
  }

  /// Sets the theta for the model from an external Vector
  ///
  /// \param input
  /// A Vector containing the theta to manually set the model
  void setTheta(const Vector<T> &input) {
    theta = input;
  }

  /// Returns the current theta for the specific model
  ///
  /// \return
  /// A Vector containing the current theta values
  Vector<T> getTheta() {
    return theta;
  }

  Vector<T> truncate(Vector<T> input) {
    Vector<T> small = (input * 100000).unaryExpr(std::ptr_fun<T,T>(std::floor));
    return (small / 100000);
  }

  /// Given a set of features and parameters creates a vector of target outputs
  ///
  /// \param inputs
  /// Matrix of input conditions
  ///
  /// \param thetas
  /// Vector of parameters to fit with input conditions
  ///
  /// \return
  /// This function returns a Vector of target outputs of type T
  Vector<T> Predict(const Matrix<T> &inputs) {
    Vector<T> predictions, yhat;
    Matrix<T> product;
    product = inputs * theta.bottomRows(theta.rows()-1);
    yhat = product.rowwise().sum();
    yhat = yhat.array() + theta(0);
    predictions = h(yhat);
    return predictions;
  }

  /// Generates a set of parameters from a given training sets
  ///
  /// \param xin
  /// Matrix of features
  ///
  /// \param y
  /// Vector of target variables for each set of features
  Vector<T> Fit(const Matrix<T> &xin, const Vector<T> &y, const Matrix<T> &inputs,
    int iterations, T alpha){
    Vector<T> gradient, predictions;
    theta.resize(xin.cols() + 1);
    gradient.resize(theta.rows());
    theta.setZero();
    gradient.setZero();
    for (int i = 0; i < iterations; i++) {
      Vector<T> x_theta_mult = (xin * (theta.bottomRows(theta.rows() - 1)));
      x_theta_mult = x_theta_mult.array() + theta(0);
      gradient.bottomRows(gradient.rows() - 1) =
        xin.transpose() * (h(x_theta_mult) - y);
      gradient(0) = theta.sum();
      theta = theta - ((alpha/ y.size()) * gradient);
      predictions = Predict(inputs);
      predictions = predictions.unaryExpr(std::ptr_fun<T,T>(std::round));
      if (((predictions - y).squaredNorm() / predictions.size()) <= .07){
        std::cout << "Ended at i = " << i << "\n";
        std::cout << ((predictions - y).squaredNorm() / predictions.size()) << "\n";
        i = iterations;
      }
    }
    return predictions;
  }
};
}  // namespace Nice
#endif  // CPP_INCLUDE_LOGISTIC_REGRESSION_H_
