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
#include <cmath>

#define BLOCK_DIM 16
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
    return 1 / ((exp(-1 * input) + 1));
  }

  /// CPU Transpose. Used for testing
  template <typename T>
  __device__ T transpose(T * input) {
    return 1 / ((exp(-1 * input) + 1));
  }

  /// CUDA kernel for predict functionality
  template <typename T>
  __global__ void PredictKernel(T *d_theta, T *d_inputs, T *d_predictions, int input_x, int input_y, T theta_0){
    extern __shared__ float yhat[];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row >= input_y || col >= input_x) return;
    for (int k = 0; k < input_y; k++) {
      sum += (d_inputs[k * input_x + col] * d_theta[row * input_x + k]);

    }
    __syncthreads();
    yhat[row * input_x + col] = sum + theta_0;
    d_predictions[row * input_x + col] = h(yhat[row * input_x + col]);
    __syncthreads();
  }

  /// Work in progress CUDA kernel for Fit functionality
  template <typename T>
  __global__ void FitKernel(T *d_xin, T *d_y, T *d_theta, int iterations,
    T alpha, int input_x, int input_y){
    extern __shared__ float theta[];
    // Variables are hard coded for development only
    __shared__ float gradient[3];
    __shared__ float new_theta[3];
    __shared__ float temp[10];

    // Corresponding row/col variables
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= input_y || col >= input_x) return;

    theta[row * input_x + col] = 0.0;
    gradient[row * input_x + col] = 0.0;
    new_theta[row * input_x + col] = 0.0;
    temp[row * input_x + col] = 0.0;

    // iterations loop for Fit kernel. The two is hard-coded for testing
    // purposes, it will be replaced by the iterations variable
    for (int i = 0; i < 2; i++) {

      // Multiplies xin array by current thetas to generate new thetas
      float sum = 0.0f;
      for (int j = 0; j < input_y; j++) {
        sum += (d_xin[j * input_x + col] * theta[row * input_x + (j+ 1)]);
      }
      __syncthreads();
      new_theta[row * input_x + col] = sum;

      // Adds the value of theta(0) to every value of new_theta
      new_theta[row * input_x + col] = theta[row * input_x] +
        new_theta[row * input_x + col];
      __syncthreads();

      // Generates hypothesis from new_theta and subtracts them from y values
      temp[row * input_x + col] = h(new_theta[row * input_x + col]) - d_y[row * input_x + col];

      /// TODO fix transpose functionality
      /// For this function, it is supposed to multiply the transpose of xin by temp.
      /// Currently, it prints out the correct multiplication values but it does not add them together
      /// The current print out shows only the first values of num in the gradient array.
      sum = 0.0f;
      for (int j = 0; j < (input_y); j++) {
          __syncthreads();
          float num = (d_xin[(row+j) * input_x + col] * temp[row * input_x + col]);
          printf("%i: %5.5f * %5.5f = %5.5f\n", j, d_xin[(row+j) * input_x + col], temp[row * input_x + col], d_xin[(row+j) * input_x + col] * temp[row * input_x + col]);
          __syncthreads();
          gradient[j+1] = gradient[j+1] + num;
          __syncthreads();
      }

      /// Sums up theta and sets it to gradient[0]
      for (int j = 1; j < input_x + 1; j++){
        sum += theta[row * input_x];
      }
      __syncthreads();
      gradient[0] = sum;

      /// Sets thetas according to gradient descent equation. 
      __syncthreads();
      d_theta[row * input_x + col] = d_theta[row * input_x + col] -
        ((alpha / input_x) * gradient[row * input_x + col]);
    }
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
  Vector<T> GpuLogisticRegression<T>::GpuPredict(const Matrix<T> &inputs, const Vector<T> &theta) {
    int m = inputs.rows();
    int k = inputs.cols();
    T theta_0 = theta[0];
    Vector<T> new_theta = (theta.bottomRows(theta.rows()-1));
    const T * h_theta = &new_theta(0);
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
    dim3 dimGrid(inputs.rows() * inputs.cols());
    //std::cout <<  (inputs.cols() / dimBlock.x) * (inputs.rows() / dimBlock.y) << "\n";
    PredictKernel<<<dimGrid, dimBlock, (k + 1)>>>(d_theta, d_inputs,
      d_predictions, m, k, theta_0);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(&h_predictions(0), d_predictions, m * sizeof(T),
      cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(d_theta));
    CUDA_CALL(cudaFree(d_inputs));
    CUDA_CALL(cudaFree(d_predictions));
    return h_predictions;
  }

  template <typename T>
  Vector<T> h(Vector<T> input) {
    input = ((-1 * input).array().exp()) + 1;
    return input.array().inverse();
  }
  /// Generates a set of parameters from a given training set
  ///
  /// \param xin
  /// Matrix of featuresAcademic
  ///
  /// \param y
  /// Vector of target variables for each set of features
  template <typename T>
  Vector<T> GpuLogisticRegression<T>::GpuFit(const Matrix<T> &xin, const Vector<T> &y,
    int iterations, T alpha){
      int m = xin.rows();
      int k = xin.cols();

      const T * h_xin = &xin(0);
      const T * h_y = &y(0);
      Vector<T> h_theta(m);

      T * d_xin;
      T * d_y;
      T * d_theta;

      // Setup GPU memory
      CUDA_CALL(cudaMalloc(&d_xin, (m * k) * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_xin, h_xin, (m * k) * sizeof(T),
        cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMalloc(&d_y, m * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_y, h_y, m * sizeof(T),
        cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMalloc(&d_theta, k * sizeof(T)));
      CUDA_CALL(cudaMemset(d_theta, 0, k * sizeof(T)));

      // Launch kernel here
      dim3 dimBlock(BLOCK_SIZE *BLOCK_SIZE);
      dim3 dimGrid(xin.rows() * xin.cols());
      //std::cout <<  (inputs.cols() / dimBlock.x) * (inputs.rows() / dimBlock.y) << "\n";
      FitKernel<<<dimGrid, dimBlock, m>>>(d_xin, d_y,
        d_theta, iterations, alpha, m, k);

      CUDA_CALL(cudaDeviceSynchronize());

      CUDA_CALL(cudaMemcpy(&h_theta(0), d_theta, k * sizeof(T),
        cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaFree(d_theta));
      CUDA_CALL(cudaFree(d_xin));
      CUDA_CALL(cudaFree(d_y));
      return h_theta;
  }

  template
  Vector<float> GpuLogisticRegression<float>::GpuFit(const Matrix<float> &xin, const Vector<float> &y,
    int iterations, float alpha);

  template
  Vector<float> GpuLogisticRegression<float>::GpuPredict(const Matrix<float> &inputs, const Vector<float> &theta);


}; // namespace Nice
#endif  //CUDA_AND_GPU
