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
#include "include/util.h"
#include "include/matrix.h"
#include "include/vector.h"
#include <cmath>
#include <chrono>
#include "include/gpu_operations.h"
#include "include/gpu_util.h"

using namespace std::chrono;

namespace Nice {
  // Used to be able to use templates with shared memory
  template <>
  struct SharedMemory <float>
  {
      __device__ float *getPointer()
      {
          extern __shared__ float s_float[];
          return s_float;
      }
  };

  // Used to be able to use templates with shared memory
  template <>
  struct SharedMemory <double>
  {
      __device__ double *getPointer()
      {
          extern __shared__ double s_double[];
          return s_double;
      }
  };

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

  template <typename T>
  __global__ void preMultiply(T * d_result, T *d_y, T *d_temp, T *d_theta){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    T value = d_result[col] + d_theta[0];
    d_temp[col] = h(value) - d_y[col];
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
    T theta_0 = theta_[0];
    Vector<T> new_theta = (theta_.bottomRows(theta_.rows()-1));
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
    dim3 dimBlock(block_size_ * block_size_);
    dim3 dimGrid(std::ceil((T) m / (block_size_ * block_size_)));

    PredictKernel<<<dimGrid, dimBlock, block_size_ * block_size_ * sizeof(T)>>>(d_theta, d_inputs,
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
  __global__ void CudaSharedKernel(T *d_a, T *d_x, T *d_y, int const a_rows, int const x_size) {
    int blockRow = blockIdx.x;
    int threadCol = threadIdx.x;
    int block_size = blockDim.x * blockDim.y;
    SharedMemory<T> shared;
    T* xTile = shared.getPointer();

    __syncthreads();
    T sum = 0.0f;
    for (int p = 0; p < std::ceil((T)x_size / (block_size)); p++){
      for (int i = 0; i < block_size; i++){
        T * aTile = &d_a[(p * block_size * a_rows) + (block_size * blockRow)];
        xTile[threadCol] = d_x[block_size * p + threadCol];
        int xGIndex = p * block_size + i;
        int yGIndex = blockIdx.x * block_size + threadIdx.x;
        if (xGIndex < x_size && yGIndex < a_rows){
          sum += aTile[(a_rows *i) + threadCol] * xTile[i];
        }
      }
    }
    __syncthreads();
    d_y[threadCol + (blockRow * block_size)] = sum;
  }

    template <typename T>
    __global__ void CudaGlobalKernel(T *d_a, T *d_x, T *d_y, int a_rows, int x_size) {
      int row = blockIdx.y * blockDim.y + threadIdx.y;
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      T sum = 0.0f;
      if (row >= x_size || col >= a_rows) return;
      for (int k = 0; k < x_size; k++) {
        sum += (d_a[col + (k * a_rows)] * d_x[k]);
      }
      d_y[row * a_rows + col] = sum;
    }

  template <typename T>
   __global__ void reduce(T *in_vector, T *sum, int n){
    SharedMemory<T> shared;
    T *shared_data = shared.getPointer();
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    T end_sum = (i < n) ? in_vector[i] : 0;

    if (i + blockDim.x < n) end_sum += in_vector[i + blockDim.x];
    shared_data[threadIdx.x] = end_sum;

    for (int pos = blockDim.x / 2; pos>0; pos >>= 1)
    {
        if (threadIdx.x < pos){
            shared_data[threadIdx.x] = end_sum = end_sum +
              shared_data[threadIdx.x + pos];
        }
    }
    if (threadIdx.x == 0){
      sum[blockIdx.x] = end_sum;
    }
}

template <typename T>
__global__ void calculateTheta(T *d_gradient, T *d_theta, T factor, int theta_size){
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < theta_size){
    d_theta[col] = d_theta[col] - (factor * d_gradient[col]);
  }
}

  template <typename T>
  void GpuLogisticRegression<T>::GpuFit(const Matrix<T> &xin, const Vector<T> &y){
      Vector<T> gradient;
      theta_.resize(xin.cols() + 1);
      gradient.resize(theta_.rows());
      theta_.setZero();
      gradient.setZero();

      Matrix<T> xin_trans = xin.transpose();
      Vector<T> temp(xin.rows());

      // Fit
      Vector<T> h_result(xin.rows());
      Vector<T> h_end(xin.cols());
      Vector<T> h_temp(xin.rows());
      Vector<T> h_gradient(xin.cols() + 1);
      Vector<T> h_theta(xin.cols() + 1);

      const T * h_xin = &xin(0);
      const T * h_y = &y(0);
      const T * h_xin_trans = &xin_trans(0);

      T * d_xin;
      CUDA_CALL(cudaMalloc(&d_xin, (xin.rows() * xin.cols() * sizeof(T))));
      CUDA_CALL(cudaMemcpy(d_xin, h_xin, (xin.rows() * xin.cols() * sizeof(T)),
        cudaMemcpyHostToDevice));

      T * d_y;
      CUDA_CALL(cudaMalloc(&d_y, y.size() * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_y, h_y, y.size() * sizeof(T),
        cudaMemcpyHostToDevice));

      T * d_xin_trans;
      CUDA_CALL(cudaMalloc(&d_xin_trans, xin_trans.size() * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_xin_trans, h_xin_trans, xin_trans.size() * sizeof(T),
        cudaMemcpyHostToDevice));

      T * d_theta;

      T * d_temp;
      CUDA_CALL(cudaMalloc(&d_temp, xin.rows() * sizeof(T)));
      CUDA_CALL(cudaMemset(d_temp, 0, xin.rows() * sizeof(T)));

      T * d_result;
      CUDA_CALL(cudaMalloc(&d_result, xin.rows() * sizeof(T)));
      CUDA_CALL(cudaMemset(d_result, 0, xin.rows() * sizeof(T)));

      T * d_end;
      CUDA_CALL(cudaMalloc(&d_end, (xin.cols() + 1) * sizeof(T)));
      CUDA_CALL(cudaMemset(d_end, 0, (xin.cols() + 1) * sizeof(T)));

      dim3 dimBlock(block_size_);
      dim3 dimGrid(std::ceil((T)xin.rows() / (block_size_)));

      dim3 dimBlockTrans(block_size_);
      dim3 dimGridTrans(std::ceil((T)(xin.cols() - 1) / (block_size_)));

      CUDA_CALL(cudaMalloc(&d_theta, theta_.size() * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_theta, &theta_(0), theta_.size() * sizeof(T),
        cudaMemcpyHostToDevice));

      for (int i = 0; i < iterations_; i++) {
        CUDA_CALL(cudaDeviceSynchronize());
        CudaGlobalKernel<<<dimGrid, dimBlock, block_size_ * sizeof(T)>>>(d_xin, d_theta + 1, d_result, xin.rows(), xin.cols());
        CUDA_CALL(cudaDeviceSynchronize());
        preMultiply<<<dimGrid,dimBlock>>>(d_result, d_y, d_temp, d_theta);
        reduce<<< dimGrid, dimBlock, block_size_ * sizeof(T)>>>(d_theta, d_end, theta_.size());
        CUDA_CALL(cudaDeviceSynchronize());
        CudaGlobalKernel<<<dimGridTrans, dimBlockTrans, block_size_ * sizeof(T)>>>(d_xin_trans, d_temp, d_end + 1, xin.cols(), xin.rows());
        CUDA_CALL(cudaDeviceSynchronize());
        calculateTheta<<< dimGrid, dimBlock>>>(d_end, d_theta, alpha_/ y.size(), theta_.size());
      }

      CUDA_CALL(cudaMemcpy(&theta_(0), d_theta, theta_.size() * sizeof(T), cudaMemcpyDeviceToHost));

      CUDA_CALL(cudaFree(d_xin));
      CUDA_CALL(cudaFree(d_y));
      CUDA_CALL(cudaFree(d_xin_trans));
      CUDA_CALL(cudaFree(d_theta));
      CUDA_CALL(cudaFree(d_temp));
      CUDA_CALL(cudaFree(d_result));
      CUDA_CALL(cudaFree(d_end));
  }

  template
  void GpuLogisticRegression<float>::GpuFit(const Matrix<float> &xin, const Vector<float> &y);

  template
  Vector<float> GpuLogisticRegression<float>::GpuPredict(const Matrix<float> &inputs);

  template
  void GpuLogisticRegression<double>::GpuFit(const Matrix<double> &xin, const Vector<double> &y);

  template
  Vector<double> GpuLogisticRegression<double>::GpuPredict(const Matrix<double> &inputs);


}; // namespace Nice
#endif  //CUDA_AND_GPU
