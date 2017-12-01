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
#include "src/cuda/cuda_matrix_vector_multiply_shared_memory.cu"
#include "include/util.h"
#include "include/matrix.h"
#include "include/vector.h"
#include <cmath>
#include <chrono>
#include "include/cuda_matrix_vector_multiply.h"
#include "include/cuda_matrix_vector_multiply_shared_memory.h"
#include "include/gpu_operations.h"
#include "include/gpu_util.h"


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
    //extern __shared__ T theta_tile[];
    SharedMemory<T> shared;
    T* theta_tile = shared.getPointer();
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int blockRow = blockIdx.x;
    int threadCol = threadIdx.x;

    T sum = 0.0f;
    if (col >= input_x) return;

    for (int p = 0; p < std::ceil((T)input_y / (blockDim.x)); p++){
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
    d_predictions[row * input_x + col] = h(sum + theta_0);
  }

 /** // First kernel for fit functionality. Fit is split into two kernels
  template <typename T>
  __global__ void FitKernel(T *d_xin, T *d_y, T *d_theta, T *d_storage,
    int iterations, T alpha, int input_x, int input_y){
    extern __shared__ T theta_tile[];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int blockRow = blockIdx.x;
    int threadCol = threadIdx.x;

    if (col >= input_x) return;

    float sum = 0.0f;
    if (iterations == 0 && col < input_y){
      d_theta[col] = 0.0;
    }

    // Tiled multiplication of d_xin matrix and theta vector
    for (int p = 0; p < std::ceil((float)input_y / (blockDim.x)); p++){
      T * d_input_tile = &d_xin[(p * blockDim.x * input_x) + (blockDim.x * blockRow)];
      theta_tile[threadCol] = d_theta[blockDim.x * p + threadCol];
      __syncthreads();
      for (int i = 0; i < blockDim.x; i++){
        int xGIndex = p * blockDim.x + i;
        int yGIndex = col;
        if (xGIndex < input_y && yGIndex < input_x){
          sum += d_input_tile[(input_x * i) + threadCol] * theta_tile[i + 1];
        }
      }
    }


    float theta_0 = d_theta[0];
    // Calls sigmoid function on each value of result + theta_0
    __syncthreads();

    // Stores result - d_y to global memory to be passed to helper function
    d_storage[col] =  h(sum + theta_0) - d_y[col];
    __syncthreads();
  }

  template <typename T>
  __global__ void FitKernelHelper(T *d_xin, T *d_y, T *d_theta, T *d_storage,
    int iterations, T alpha, int input_x, int input_y){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int blockRow = blockIdx.x;
    int threadCol = threadIdx.x;
    if (col > input_y) return;
    extern __shared__ T gradient[];

    float sum = 0.0f;

    gradient[col] = 0.0;

    // Tiled multiplication of the transpose of d_xin and result from previous kernel
    if (col < input_y){
      T * d_input_tile = &d_xin[(blockRow * blockDim.x * input_x) + (threadCol * input_x)];
      for (int i = 0; i < input_x; i++){
        sum += d_input_tile[i] * d_storage[i];
      }
      __syncthreads();
      // Stores result into gradient
      gradient[col + 1] = sum;
    }
    atomicAdd(&gradient[0], d_theta[col]);
    // Calculates new theta value
    d_theta[col] = d_theta[col] - ((alpha / input_x) * gradient[col]);
  }**/

  template <typename T>
  __global__ void preMultiply(T * d_result, T *d_y, T *d_temp, T theta_0){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    T value = d_result[col] + theta_0;
    d_temp[col] = h(value) - d_y[col];
  }

  template <typename T>
  __global__ void calculateTheta(T *d_gradient, T *d_theta, T factor, int theta_size){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ T theta_0;
    theta_0 = 0;
    if (col < theta_size){
      atomicAdd(&theta_0, d_theta[col]);
      if (col == 0){
        d_theta[0] = d_theta[0] - (factor * theta_0);
      }
      else{
        d_theta[col] = d_theta[col] - (factor * d_gradient[col - 1]);
      }
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
    dim3 dimGrid(std::ceil((T) m / (BLOCK_SIZE * BLOCK_SIZE)));

    PredictKernel<<<dimGrid, dimBlock, BLOCK_SIZE * BLOCK_SIZE * sizeof(T)>>>(d_theta, d_inputs,
      d_predictions, m, k, theta_0);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(&h_predictions(0), d_predictions, m * sizeof(T),
      cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(d_theta));
    CUDA_CALL(cudaFree(d_inputs));
    CUDA_CALL(cudaFree(d_predictions));
    return h_predictions;
  }
  /**
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
      dim3 dimHelperG(std::ceil((float)k / (BLOCK_SIZE * BLOCK_SIZE)));


      for (int i = 0; i < iterations; i++) {
        FitKernel<<<dimGrid, dimBlock, BLOCK_SIZE * BLOCK_SIZE * sizeof(T) >>>(d_xin, d_y,
          d_theta, d_storage, i, alpha, m, k);
        FitKernelHelper<<<dimHelperG, dimHelperB, k * sizeof(T) >>>(d_xin, d_y,
          d_theta, d_storage, i, alpha, m, k);
      }
      CUDA_CALL(cudaDeviceSynchronize());

      CUDA_CALL(cudaMemcpy(&h_theta(0), d_theta, (k + 1) * sizeof(T),
        cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaFree(d_theta));
      CUDA_CALL(cudaFree(d_xin));
      CUDA_CALL(cudaFree(d_storage));
      CUDA_CALL(cudaFree(d_y));
      theta = h_theta;
  }
  **/

  /**template <typename T>
  Vector<T> GpuLogisticRegression<T>::GpuPredictMV(const Matrix<T> &inputs) {
    Vector<T> predictions, yhat;
    Matrix<T> product;

    CUDA_CALL(cudaMalloc(&d_xin, xin.size() * sizeof(T)));
    CUDA_CALL(cudaMemcpy(d_xin, h_xin, xin.size() * sizeof(T),
      cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_y, y.size() * sizeof(T)));
    CUDA_CALL(cudaMemcpy(d_y, h_y, y.size() * sizeof(T),
      cudaMemcpyHostToDevice));

    product = inputs * theta.bottomRows(theta.rows()-1);
    yhat = product.rowwise().sum();
    yhat = yhat.array() + theta(0);
    predictions = h(yhat);
    return predictions;
  }**/

  template <typename T>
  Vector<T> GpuLogisticRegression<T>::GpuFitMV(const Matrix<T> &xin, const Vector<T> &y,
    const Matrix<T> &predict_inputs, int iterations, T alpha){
      Vector<T> gradient;
      theta.resize(xin.cols() + 1);
      gradient.resize(theta.rows());
      theta.setZero();
      gradient.setZero();

      Vector<T> bottom_theta = theta.bottomRows(theta.rows() - 1);
      Matrix<T> xin_trans = xin.transpose();
      Vector<T> temp(xin.rows());

      // Fit
      Vector<T> h_result(xin.rows());
      Vector<T> h_gradient(xin.cols() + 1);
      Vector<T> h_theta(xin.cols() + 1);

      const T * h_xin = &xin(0);
      const T * h_y = &y(0);
      const T * h_temp = &h_result(0);
      const T * h_xin_trans = &xin_trans(0);
      const T * h_bottom_theta = &bottom_theta(0);

      // Predict
      Vector<T> h_predictions(predict_inputs.rows());

      const T * h_predict_inputs = &predict_inputs(0);

      // Fit
      T * d_xin;
      T * d_y;
      T * d_xin_trans;
      T * d_bottom_theta;
      T * d_result;
      T * d_gradient;
      T * d_temp;
      T * d_theta;

      // Predict
      T * d_predictions;
      T * d_predict_inputs;

      // Setup GPU memory
      /**CUDA_CALL(cudaMalloc(&d_xin, xin.size() * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_xin, h_xin, xin.size() * sizeof(T),
        cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMalloc(&d_y, y.size() * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_y, h_y, y.size() * sizeof(T),
        cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMalloc(&d_xin_trans, xin.size() * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_xin_trans, h_xin_trans, xin.size() * sizeof(T),
        cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMalloc(&d_predict_inputs, predict_inputs.size() * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_predict_inputs, h_predict_inputs, predict_inputs.size() * sizeof(T),
        cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMalloc(&d_bottom_theta, bottom_theta.rows() * sizeof(T)));

      CUDA_CALL(cudaMalloc(&d_result, xin.rows() * sizeof(T)));
      CUDA_CALL(cudaMalloc(&d_gradient, xin.size() * sizeof(T)));
      CUDA_CALL(cudaMalloc(&d_temp, xin.size() * sizeof(T)));

      CUDA_CALL(cudaMalloc(&d_theta, theta.size() * sizeof(T)));
      CUDA_CALL(cudaMemset(d_theta, 0, theta.size() * sizeof(T)));

      CUDA_CALL(cudaMalloc(&d_predictions, predict_inputs.rows() * sizeof(T)));**/

      for (int i = 0; i < iterations; i++) {
        bottom_theta = theta.bottomRows(theta.rows() - 1);
        /**CUDA_CALL(cudaMemcpy(d_bottom_theta, h_bottom_theta, bottom_theta.rows() * sizeof(T),
          cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemset(d_result, 0, xin.rows() * sizeof(T)));

        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid(std::ceil((T)xin.rows()/ (BLOCK_SIZE)));

        CudaSharedMVKernel<<<dimGrid, dimBlock, BLOCK_SIZE * sizeof(T)>>>
          (d_xin, d_bottom_theta, d_result, xin.rows(), xin.cols());

        CUDA_CALL(cudaMemcpy(&h_result(0), d_result, xin.rows() * sizeof(T),
          cudaMemcpyDeviceToHost));**/
        Nice::CudaSharedMVMultiply<T> global_op(1024);
        h_result = global_op.Multiply(xin, bottom_theta);
        // h_result = xin * bottom_theta;
        h_result = h_result.array() + theta(0);
        gradient.bottomRows(gradient.rows() - 1) =
          global_op.Multiply(xin.transpose(), (h(h_result) - y));
          // xin.transpose() * (h(h_result) - y);
        gradient(0) = theta.sum();
        theta = theta - ((alpha/ y.size()) * gradient);

        /**CUDA_CALL(cudaMemset(d_temp, 0, xin.rows() * sizeof(T)));
        CUDA_CALL(cudaMemcpy(d_temp, h_temp, xin.rows() * sizeof(T),
          cudaMemcpyHostToDevice));

        preMultiply<<<dimGrid, dimBlock>>>(d_result, d_y, d_temp, theta(0));

        CUDA_CALL(cudaMemset(d_gradient, 0, xin.rows() * sizeof(T)));
        dim3 dimGrid2(std::ceil((float)xin_trans.rows()/ (BLOCK_SIZE)));

        CudaSharedMVKernel<<<dimGrid2, dimBlock, BLOCK_SIZE * sizeof(T)>>>
          (d_xin_trans, d_temp, d_gradient, xin.cols(), xin.rows());

        dim3 dimGrid3(std::ceil((float)theta.size()/ (BLOCK_SIZE)));
        calculateTheta<<<dimGrid3, dimBlock, sizeof(T)>>>(d_gradient, d_theta, alpha / y.size(), theta.size());
        CUDA_CALL(cudaMemcpy(&theta(0), d_theta, theta.size() * sizeof(T),
          cudaMemcpyDeviceToHost));

        CUDA_CALL(cudaMalloc(&d_theta, theta.size() * sizeof(T)));
        CUDA_CALL(cudaMemcpy(d_theta, &theta(0), theta.size() * sizeof(T),
          cudaMemcpyHostToDevice));**/

        Vector<T> yhat;
        Matrix<T> product;
        product = predict_inputs * theta.bottomRows(theta.rows()-1);
        yhat = product.rowwise().sum();
        yhat = yhat.array() + theta(0);
        h_predictions = h(yhat);
        h_predictions = h_predictions.unaryExpr(std::ptr_fun<T,T>(std::round));
      }

      CUDA_CALL(cudaDeviceSynchronize());

      CUDA_CALL(cudaFree(d_xin));
      CUDA_CALL(cudaFree(d_y));
      CUDA_CALL(cudaFree(d_xin_trans));
      CUDA_CALL(cudaFree(d_bottom_theta));
      CUDA_CALL(cudaFree(d_result));
      CUDA_CALL(cudaFree(d_gradient));
      CUDA_CALL(cudaFree(d_temp));
      CUDA_CALL(cudaFree(d_theta));
      CUDA_CALL(cudaFree(d_predictions));
      CUDA_CALL(cudaFree(d_predict_inputs));

      // CUDA_CALL(cudaDeviceSynchronize());
      //
      // // high_resolution_clock::time_point t2 = high_resolution_clock::now();
      // // auto duration = duration_cast<microseconds>( t2 - t1 ).count();
      // // std::cout << "CUDA Logistic Regression - Fit: " << (long)duration << std::endl;
      //
      // CUDA_CALL(cudaMemcpy(&h_theta(0), d_theta, (k + 1) * sizeof(T),
      //   cudaMemcpyDeviceToHost));
      // CUDA_CALL(cudaFree(d_theta));
      // CUDA_CALL(cudaFree(d_xin));
      // CUDA_CALL(cudaFree(d_storage));
      // CUDA_CALL(cudaFree(d_y));
      // theta = h_theta;
      return h_predictions;
  }

  /**template
  void GpuLogisticRegression<float>::GpuFit(const Matrix<float> &xin, const Vector<float> &y,
    int iterations, float alpha);**/

  template
  Vector<float> GpuLogisticRegression<float>::GpuFitMV(const Matrix<float> &xin, const Vector<float> &y,
      const Matrix<float> &predict_inputs, int iterations, float alpha);

  template
  Vector<float> GpuLogisticRegression<float>::GpuPredict(const Matrix<float> &inputs);

  template
  Vector<double> GpuLogisticRegression<double>::GpuFitMV(const Matrix<double> &xin, const Vector<double> &y,
      const Matrix<double> &predict_inputs, int iterations, double alpha);

  template
  Vector<double> GpuLogisticRegression<double>::GpuPredict(const Matrix<double> &inputs);


}; // namespace Nice
#endif  //CUDA_AND_GPU
