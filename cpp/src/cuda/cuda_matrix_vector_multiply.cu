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
#include "include/cuda_matrix_vector_multiply.h"
#include <chrono>

using namespace std::chrono;

namespace Nice {

  template <typename T>
  __global__ void CudaMatrixVectorMulKernel(T *d_a, T *d_x, T *d_y, int a_rows, int x_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row >= x_size || col >= a_rows) return;
    for (int k = 0; k < x_size; k++) {
      sum += (d_a[k * a_rows + col] * d_x[row * a_rows + k]);
    }
    __syncthreads();
    d_y[row * a_rows + col] = sum;
  }

  template <typename T>
  Vector<T> CudaMatrixVectorMultiply<T>::Multiply(const Matrix<T> &a, const Vector<T> &b) {
    if (a.cols() == b.rows() && !a.isZero()) {
      // Allocate and transfer memories
      int m = a.rows();
      int n = b.cols();
      int k = a.cols();

      const T * h_a = &a(0);
      const T * h_x = &b(0);
      Vector<T> h_y(m);

      T * d_a;
      T * d_x;
      T * d_y;

      // Setup GPU memory
      CUDA_CALL(cudaMalloc(&d_a, (m * k) * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_a, h_a, (m * k) * sizeof(T),
        cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMalloc(&d_x, k * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_x, h_x, k * sizeof(T),
          cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMalloc(&d_y, m * sizeof(T)));
      CUDA_CALL(cudaMemset(d_y, 0, m * sizeof(T)));

      // Launch kernel here

      high_resolution_clock::time_point t1 = high_resolution_clock::now();
      CudaMatrixVectorMulKernel<<<m, 256>>>(d_a, d_x, d_y, m, k);

      // Device sync
      CUDA_CALL(cudaDeviceSynchronize());
      high_resolution_clock::time_point t2 = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>( t2 - t1 ).count();
      std::cout << "CUDA global time: " << (long)duration << std::endl;

      // Transfer memories back, clear memrory, and return result
      CUDA_CALL(cudaMemcpy(&h_y(0), d_y, m * sizeof(T),
        cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaFree(d_a));
      CUDA_CALL(cudaFree(d_x));
      CUDA_CALL(cudaFree(d_y));

      return h_y;
    } else if (a.cols() != b.rows()) {
      std::cerr << "Matricies in gpu matrix multiply's sizes aren't compatible"
                << std::endl;
      exit(1);
    } else if (a.isZero() && b.isZero()) {
      std::cerr << "The maxtrix and the vector are empty"
                << std::endl;
      exit(1);
    } else if (a.isZero()) {
      std::cerr << "The maxtrix is empty"
                << std::endl;
      exit(1);
    } else if (b.isZero()) {
      std::cerr << "The vector is empty"
                << std::endl;
      exit(1);
    } else {
      std::cerr << "Unknown error"
                << std::endl;
      exit(1);
    }
  }
  template
  Vector<float> CudaMatrixVectorMultiply<float>::Multiply(const Matrix<float> &a, const Vector<float> &b);

  template
  Vector<double> CudaMatrixVectorMultiply<double>::Multiply(const Matrix<double> &a, const Vector<double> &b);

}
