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
      T * d_input_tile = &d_inputs[(p * blockDim.x * input_x) + (blockDim.x * blockRow)];
      theta_tile[threadCol] = d_theta[blockDim.x * p + threadCol];
      __syncthreads();
      for (int i = 0; i < blockDim.x; i++){
        int xGIndex = p * blockDim.x + i;
        int yGIndex = col;
        if (xGIndex < input_y && yGIndex < input_x){
          sum += d_input_tile[(input_x * i) + threadCol] * theta_tile[i];
        }
      }
    }
    d_predictions[row * input_x + col] = 1 / ((exp(-1 * (sum + theta_0)) + 1));
  }

  template <typename T>
  __global__ void FitKernel(T *d_xin, T *d_y, T *d_theta, T *d_storage,
    int iterations, T alpha, int input_x, int input_y){
    extern __shared__ T theta_tile[];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int blockRow = blockIdx.x;
    int threadCol = threadIdx.x;

    if (col >= input_x) return;
    if (iterations == 0 && col < input_y){
      d_theta[col] = 0.0;
    }
    float sum = 0.0f;

    for (int p = 0; p < std::ceil((float)input_y / (blockDim.x)); p++){
      T * d_input_tile = &d_xin[(p * blockDim.x * input_x) + (blockDim.x * blockRow)];
      theta_tile[threadCol] = d_theta[blockDim.x * p + threadCol + 1];
      __syncthreads();
      for (int i = 0; i < blockDim.x; i++){
        int xGIndex = p * blockDim.x + i;
        int yGIndex = col;
        if (xGIndex < input_y && yGIndex < input_x){
          sum += d_input_tile[(input_x * i) + threadCol] * theta_tile[i];
        }
      }
    }
    /**
    for (int j = 0; j < input_y; j++) {
      sum += (d_xin[j * input_x + col] * d_theta[j+ 1]);
    }**/
    __syncthreads();
    d_storage[col] = h(sum + d_theta[0]) - d_y[col];
    __syncthreads();
  }


  template <typename T>
  __global__ void FitKernelHelper(T *d_xin, T *d_y, T *d_theta, T *d_storage,
    int iterations, T alpha, int input_x, int input_y){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int blockRow = blockIdx.x;
    int threadCol = threadIdx.x;

    if (col > input_y) return;
    extern __shared__ T storage[];
    T * theta_tile = (T*)storage;
    T * gradient = (T*)&theta_tile[blockDim.x];

    float sum = 0.0f;

    gradient[col] = 0.0;

    if (threadCol < input_y){
      for (int j = 0; j < input_x; j++) {
        sum += d_xin[col * input_x + j] * d_storage[j];
        //printf("%5.5f * %5.5f = %5.5f \n",
         //d_xin[col * input_x + j], d_storage[j],
         //d_xin[col * input_x + j] * d_storage[j]);
      }

      /**
      for (int j = 0; j < input_y; j++) {
        sum += (d_xin[j * input_x + col] * d_theta[j+ 1]);
      }**/

      /**for (int p = 0; p < std::ceil((float)input_y / (blockDim.x)); p++){
        T * d_input_tile = &d_xin[(p * blockDim.x * input_x) + (blockDim.x * blockRow)];
        theta_tile[threadCol] = d_storage[blockDim.x * p + threadCol + 1];
        __syncthreads();
        for (int i = 0; i < blockDim.x; i++){
          int xGIndex = p * blockDim.x + i;
          int yGIndex = col;
          if (xGIndex < input_y && yGIndex < input_x){
            sum += d_input_tile[threadCol * input_x + i] * theta_tile[i];
            //printf("%5.5f * %5.5f = %5.5f \n",
              //d_input_tile[(input_x * i) + threadCol], theta_tile[i],
              //d_input_tile[(input_x * i) + threadCol] * theta_tile[i]);
          }
        }
      }**/
    }

    gradient[col + 1] += sum;
    sum = 0.0f;
    gradient[0] += d_theta[col];

    __syncthreads();
    d_theta[col] = d_theta[col] - ((alpha / input_x) * gradient[col]);
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
    dim3 dimGrid(std::ceil((float) m / (BLOCK_SIZE * BLOCK_SIZE)));

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    PredictKernel<<<dimGrid, dimBlock, BLOCK_SIZE * BLOCK_SIZE * sizeof(T)>>>(d_theta, d_inputs,
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
      T * d_storage;

      // Setup GPU memory
      CUDA_CALL(cudaMalloc(&d_xin, (m * k) * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_xin, h_xin, (m * k) * sizeof(T),
        cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMalloc(&d_y, m * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_y, h_y, m * sizeof(T),
        cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMalloc(&d_theta, (k + 1) * sizeof(T)));
      CUDA_CALL(cudaMemset(d_theta, 0, (k + 1) * sizeof(T)));

      CUDA_CALL(cudaMalloc(&d_storage, m * sizeof(T)));
      CUDA_CALL(cudaMemset(d_storage, 0, m * sizeof(T)));

      // Launch kernel here
      dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
      dim3 dimGrid(std::ceil((float)m / (BLOCK_SIZE * BLOCK_SIZE)));

      dim3 dimHelperB(BLOCK_SIZE * BLOCK_SIZE);
      dim3 dimHelperG(std::ceil((float)(k) / (BLOCK_SIZE * BLOCK_SIZE)));

      high_resolution_clock::time_point t1 = high_resolution_clock::now();

      for (int i = 0; i < iterations; i++) {
        FitKernel<<<dimGrid, dimBlock, BLOCK_SIZE * BLOCK_SIZE * sizeof(T)>>>(d_xin, d_y,
          d_theta, d_storage, i, alpha, m, k);
        //CUDA_CALL(cudaDeviceSynchronize());
        FitKernelHelper<<<dimHelperG, dimHelperB, (BLOCK_SIZE * BLOCK_SIZE + m) * sizeof(T)>>>(d_xin, d_y,
          d_theta, d_storage, i, alpha, m, k);
      }
      CUDA_CALL(cudaDeviceSynchronize());

      high_resolution_clock::time_point t2 = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>( t2 - t1 ).count();
      std::cout << "CUDA Logistic Regression - Fit: " << (long)duration << std::endl;

      CUDA_CALL(cudaMemcpy(&h_theta(0), d_theta, (k + 1) * sizeof(T),
        cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaFree(d_theta));
      CUDA_CALL(cudaFree(d_xin));
      CUDA_CALL(cudaFree(d_storage));
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
