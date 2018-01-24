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
#include <string>
namespace Nice {

template <typename T>
class GpuLogisticRegression {
 private:
  Vector<T> theta_;
  int block_size_;
  T alpha_;
  int iterations_;
  std::string mem_type_;
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
  GpuLogisticRegression(): block_size_(32), alpha_(0.001), iterations_(1000){}

  GpuLogisticRegression(int in_block, int in_iterations, T in_alpha) {
    block_size_ = in_block;
    iterations_ = in_iterations;
    alpha_ = in_alpha;
  }

  void SetAlpha(T in_alpha) {
    alpha_ = in_alpha;
  }

  void SetIterations(int in_iterations) {
    iterations_ = in_iterations;
  }

  void SetMemType(std::string in_mem_type) {
    mem_type_ = in_mem_type;
  }

  T GetAlpha() {
    return alpha_;
  }

  int GetIterations() {
    return iterations_;
  }

  std::string GetMemType() {
    return mem_type_;
  }

  /// Sets the theta for the model from an external Vector
  ///
  /// \param input
  /// A Vector containing the theta to manually set the model
  void SetTheta(const Vector<T> &input) {theta_ = input;}

  /// Returns the current theta for the specific model
  ///
  /// \return
  /// A Vector containing the current theta values
  Vector<T> GetTheta() {return theta_;}

  void GpuFit(const Matrix<T> &xin, const Vector<T> &y);

  Vector<T> GpuPredict(const Matrix<T> &inputs);
};
}  // namespace Nice

#endif  // CUDA_AND_GPU
#endif  // CPP_INCLUDE_GPU_LOGISTIC_REGRESSION_H_
