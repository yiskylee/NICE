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
#ifdef CUDA_AND_GPU
#include "include/gpu_logistic_regression.h"

#define BLOCK_SIZE 16
namespace Nice {
  /// Calculates the hypothesis of a given input Vector
  ///
  /// \param input
  /// Input Vector
  ///
  /// \return
  /// This function returns a Vector of type T
  template <typename T>
  __device__ T h(T input) {
    input = 1 / (((-1 * input).array().exp()) + 1);
    return input;
  }

  template <typename T>
  __global__ void PredictKernel(T *d_theta, T *d_inputs, T *d_predictions, int input_x, int input_y){
    extern __shared__ float product[];
    extern __shared__ float yhat[];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row >= input_x || col >= input_y) return;
    for (int k = 0; k < input_x; k++) {
      sum += (d_inputs[k * input_y + col] * d_theta[row * input_y + k]);
    }
    __syncthreads();
    product[row * input_y + col] = sum;
    sum = 0.0f;
    __syncthreads();

    for (int k = 0; k < input_y; k++) {
      sum += product[input_y + k * col];
    }
    __syncthreads();
    yhat[row * input_y + col] = sum;
    __syncthreads();
    for (int k = 0; k < input_y; k++){
      yhat += d_theta(0);
    }

    for (int k = 0; k < input_y; k++){
      d_predictions[row * input_y + col] = h(yhat[input_y + k * col]);
    }
    __syncthreads();
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
  template <typename T>
  Vector<T> Predict(const Matrix<T> &inputs) {
    int m = inputs.rows();
    int k = inputs.cols();
    const T * h_theta = &theta.bottomRows(theta.rows()-1)(0);
    const T * h_inputs = &inputs(0);
    Vector<T> h_predictions(m);

    T * d_theta;
    T * d_inputs;
    T * d_predictions;

    // Setup GPU memory
    CUDA_CALL(cudaMalloc(&d_inputs, (m * k) * sizeof(T)));
    CUDA_CALL(cudaMemcpy(d_inputs, h_inputs, (m * k) * sizeof(T),
      cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_theta, k * sizeof(T)));
    CUDA_CALL(cudaMemcpy(d_theta, h_theta, k * sizeof(T),
      cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_predictions, m * sizeof(T)));
    CUDA_CALL(cudaMemset(d_predictions, 0, m * sizeof(T)));
    // Launch kernel here
    dim3 dimBlock(BLOCK_SIZE *BLOCK_SIZE);
    dim3 dimGrid((inputs.rows() / dimBlock.x) * (inputs.cols() / dimBlock.y));

    PredictKernel<<<dimBlock, dimGrid, m>>>(d_theta, d_inputs,
      d_predictions, m * k, k);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(&h_predictions(0), d_predictions, m * sizeof(T),
      cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(d_theta));
    CUDA_CALL(cudaFree(d_inputs));
    CUDA_CALL(cudaFree(d_predictions));
    return h_predictions;
  }

  /// Generates a set of parameters from a given training set
  ///
  /// \param xin
  /// Matrix of features
  ///
  /// \param y
  /// Vector of target variables for each set of features
  template <typename T>
  void Fit(const Matrix<T> &xin, const Vector<T> &y,
    int iterations, T alpha){
    Vector<T> gradient;
    theta.resize(xin.cols() + 1);
    gradient.resize(theta.rows());
    theta.setZero();
    gradient.setZero();
    for (int i = 0; i < iterations; i++) {
      Vector<T> Xtheta = (xin * (theta.bottomRows(theta.rows() - 1)));
      Xtheta = Xtheta.array() + theta(0);
      gradient.bottomRows(gradient.rows() - 1) =
        xin.transpose() * (h(Xtheta) - y);
      gradient(0) = theta.sum();
      theta = theta - ((alpha/ y.size()) * gradient);
    }
    std::cout << theta << '\n';
  }
}; // namespace Nice
#endif  //CUDA_AND_GPU
