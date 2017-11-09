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
#include "include/cuda_matrix_vector_multiply_shared_memory.h"
#define BLOCK_SIZE 32

using namespace std::chrono;

namespace Nice {

  template <typename T>
  __global__ void CudaSharedMVKernel(T *d_a, T *d_x, T *d_y, int const a_rows, int const x_size) {
    int blockRow = blockIdx.x;
    int threadCol = threadIdx.x;

    extern __shared__ T xTile[];

    __syncthreads();
    T sum = 0.0f;
    for (int p = 0; p < std::ceil((float)x_size / (BLOCK_SIZE)); p++){
      for (int i = 0; i < BLOCK_SIZE; i++){
        T * aTile = &d_a[(p * BLOCK_SIZE * a_rows) + (BLOCK_SIZE * blockRow)];
        extern __shared__ T xTile[];
        xTile[threadCol] = d_x[BLOCK_SIZE * p + threadCol];
        int xGIndex = p * BLOCK_SIZE + i;
        int yGIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        if (xGIndex < x_size && yGIndex < a_rows){
          sum += aTile[(a_rows *i) + threadCol] * xTile[i];
        }
      }
    }
    __syncthreads();
    d_y[threadCol + (blockRow * BLOCK_SIZE)] += sum;
  }

  template <typename T>
  Vector<T> CudaSharedMVMultiply<T>::Multiply(const Matrix<T> &a, const Vector<T> &b) {
    if (a.cols() == b.rows() && !a.isZero()) {
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
      CUDA_CALL(cudaMalloc(&d_a, m * k * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_a, h_a, m * k * sizeof(T),
        cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMalloc(&d_x, k * sizeof(T)));
      CUDA_CALL(cudaMemcpy(d_x, h_x, k * sizeof(T),
          cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMalloc(&d_y, m * sizeof(T)));
      CUDA_CALL(cudaMemset(d_y, 0, m * sizeof(T)));

      // Launch kernel here
      dim3 dimBlock(BLOCK_SIZE);
      dim3 dimGrid(std::ceil((float)m / (BLOCK_SIZE)));

      CudaSharedMVKernel<<<dimGrid, dimBlock, BLOCK_SIZE * sizeof(T)>>>
        (d_a, d_x, d_y, m, k);
      // Device sync
      CUDA_CALL(cudaDeviceSynchronize());

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
  Vector<float> CudaSharedMVMultiply<float>::Multiply(const Matrix<float> &a, const Vector<float> &b);

}
