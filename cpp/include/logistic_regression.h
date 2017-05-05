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
 public:
  static T sigmoid( T x){
    return 1 / (1 + std::exp(-x));
  }

 
  static Vector<T> Predict(const Matrix<T> &inputs, const Vector<T> thetas){
    Vector<T> predictions;
    Vector<T> mult_thetas;
    mult_thetas.resize(thetas.size()-1);  
    mult_thetas << thetas(1), thetas(2);
    Matrix<T> product;
    Vector<T> yhat; 
    product.resize(inputs.rows(),inputs.cols());
    product = inputs * mult_thetas;
    yhat.resize(inputs.rows());
    yhat = product.rowwise().sum();
    yhat = yhat.array() + thetas(0);
    predictions.resize(inputs.rows());
    // TODO Parallelize exponential function
    for (int i = 0; i < yhat.size(); i++){
      T value = (1 / (1 + (exp(-yhat(i)))));  
      predictions(i) = value;  
    }
    return predictions;
  }

  /// Calculates the coeffients for a logisitic regression of a given matrix and
  /// returns it as a vector. 
  ///
  /// 
  static Vector<T> Fit(const Matrix<T> &x, const Vector<T> &y, int iterations, T alpha){
    Vector<T> thetas;
    thetas.resize(3);
    thetas << 0, 0, 0;
    Vector<T> z, grad;
    for (int i = 0; i < iterations; i++){
      z.resize(x.rows());
      grad.resize(thetas.size());
      z = Predict(x, thetas); 
      thetas -= alpha * (x.transpose() * (z - y)) / y.size();
    }
    return thetas;
  }
};
}  // namespace Nice
#endif  // CPP_INCLUDE_CPU_OPERATIONS_H_
