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
#include <chrono>

using namespace std::chrono;

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

  /// CUDA kernel for predict functionality
  template <typename T>
  __global__ void PredictKernel(T *d_theta, T *d_inputs, T *d_predictions,
      int input_x, int input_y, T theta_0){
    extern __shared__ T theta_tile[];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int blockRow = blockIdx.x;
    int threadCol = threadIdx.x;

    float sum = 0.0f;
    if (col >= input_x) return;
    for (int p = 0; p < std::ceil((float)input_y / (blockDim.x)); p++){
      for (int i = 0; i < blockDim.x; i++){
        T * d_input_tile = &d_inputs[(p * blockDim.x * input_x) + (blockDim.x * blockRow)];
        theta_tile[threadCol] = d_theta[blockDim.x * p + threadCol];

        int xGIndex = p * blockDim.x + i;
        int yGIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (xGIndex < input_y && yGIndex < input_x){
          sum += d_input_tile[(input_x * i) + threadCol] * theta_tile[i];
        }
      }
      __syncthreads();
    }
    __syncthreads();
    d_predictions[row * input_x + col] = 1 / ((exp(-1 * (sum + theta_0)) + 1));
  }


  /// Work in progress CUDA kernel for Fit functionality
  template <typename T>
  __global__ void FitKernel(T *d_xin, T *d_y, T *d_theta,
    int iterations, T alpha, int input_x, int input_y){
    extern __shared__ float shared[];
    // Variables are hard coded for development only
    T * theta = (T*)shared;
    T * gradient = (T*)&theta[input_y + 1];
    T * temp = (T*)&gradient[input_y + 1];

    // Corresponding row/col variables
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= input_x) return;
    if (col < input_y + 1){
      theta[col] = 0.0;
    }
    // iterations loop for Fit kernel. The two is hard-coded for testing
    // purposes, it will be replaced by the iterations variable
    for (int i = 0; i < iterations; i++) {
      if(col < input_y + 1){
        gradient[col] = 0.0;
        temp[col] = 0.0;
      }
      float sum = 0.0f;
      for (int j = 0; j < input_y; j++) {
        sum += (d_xin[j * input_x + col] * theta[j+ 1]);
      }
      __syncthreads();
      // Adds the value of theta(0) to every value of new_theta
      // Generates hypothesis from new_theta and subtracts them from y values
      temp[col] = h(sum + theta[0]) - d_y[col];
      sum = 0.0f;
      for (int j = 0; j < (input_y); j++) {
        atomicAdd(&gradient[j + 1], (d_xin[j * input_x + col] * temp[col]));
      }
      /// Sums up theta and sets it to gradient[0]
      sum = 0.0f;
      for (int j = 0; j < input_y + 1; j++){
        sum += theta[j];
      }
      __syncthreads();
      gradient[0] = sum;
      /// Sets thetas according to gradient descent equation.
      if (col < input_y + 1){
        theta[col] = theta[col] - ((alpha / input_x) * gradient[col]);
      }
    }
    if (col < input_y + 1){
      d_theta[col] = theta[col];
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
  Vector<T> GpuLogisticRegression<T>::GpuPredict(const Matrix<T> &inputs) {
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
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid(std::ceil((float)m / (BLOCK_SIZE)));

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    PredictKernel<<<dimGrid, dimBlock, (m + 1) * sizeof(T)>>>(d_theta, d_inputs,
      d_predictions, m, k, theta_0);
    CUDA_CALL(cudaDeviceSynchronize());
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "CUDA Logistic Regression - Predict: " << (long)duration << std::endl;

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
  void GpuLogisticRegression<T>::GpuFit(const Matrix<T> &xin, const Vector<T> &y,
    int iterations, T alpha){
      int m = xin.rows();
      int k = xin.cols();

      const T * h_xin = &xin(0);
      const T * h_y = &y(0);
      Vector<T> h_theta(k+1);

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

      CUDA_CALL(cudaMalloc(&d_theta, (k + 1) * sizeof(T)));
      CUDA_CALL(cudaMemset(d_theta, 0, (k + 1) * sizeof(T)));

      // Launch kernel here
      dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
      dim3 dimGrid(std::ceil((float)m / (BLOCK_SIZE * BLOCK_SIZE)));

      high_resolution_clock::time_point t1 = high_resolution_clock::now();
      int shared_size = ((3 * m )  * sizeof(T) + (k+1)  * sizeof(T));
      FitKernel<<<dimGrid, dimBlock, shared_size>>>(d_xin, d_y,
        d_theta, iterations, alpha, m, k);

      CUDA_CALL(cudaDeviceSynchronize());
      high_resolution_clock::time_point t2 = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>( t2 - t1 ).count();
      std::cout << "CUDA Logistic Regression - Fit: " << (long)duration << std::endl;



      CUDA_CALL(cudaMemcpy(&h_theta(0), d_theta, (k + 1) * sizeof(T),
        cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaFree(d_theta));
      CUDA_CALL(cudaFree(d_xin));
      CUDA_CALL(cudaFree(d_y));
      theta = h_theta;
  }

  template
  void GpuLogisticRegression<float>::GpuFit(const Matrix<float> &xin, const Vector<float> &y,
    int iterations, float alpha);

  template
  Vector<float> GpuLogisticRegression<float>::GpuPredict(const Matrix<float> &inputs);


}; // namespace Nice
#endif  //CUDA_AND_GPU
