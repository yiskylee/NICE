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

#ifndef CPP_INCLUDE_GPU_LOGISTIC_REGRESSION_H_
#define CPP_INCLUDE_GPU_LOGISTIC_REGRESSION_H_

#ifdef CUDA_AND_GPU

#include <iostream>
#include "include/gpu_util.h"

namespace Nice {

template <typename T>
class GpuLogisticRegression {
 private:
  Vector<T> theta;
  int BLOCK_SIZE;
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

    Vector<T> truncate(Vector<T> input) {
      Vector<T> small = (input * 10000).unaryExpr(std::ptr_fun<T,T>(std::floor));
      return (small / 10000);
    }


 public:
  GpuLogisticRegression() {BLOCK_SIZE = 32;}
  GpuLogisticRegression(int inBlock) {BLOCK_SIZE = inBlock;}
  /// Sets the theta for the model from an external Vector
  ///
  /// \param input
  /// A Vector containing the theta to manually set the model
  void setTheta(const Vector<T> &input) {theta = input;}

  /// Returns the current theta for the specific model
  ///
  /// \return
  /// A Vector containing the current theta values
  Vector<T> getTheta() {return theta;}

  void GpuFit(const Matrix<T> &xin, const Vector<T> &y,
      int iterations, T alpha);

  Vector<T> GpuFitMV(const Matrix<T> &xin, const Vector<T> &y,
      const Matrix<T> &predict_inputs, int iterations, T alpha);

  Vector<T> GpuPredict(const Matrix<T> &inputs);
};
}  // namespace Nice

#endif  // CUDA_AND_GPU
#endif  // CPP_INCLUDE_GPU_LOGISTIC_REGRESSION_H_
