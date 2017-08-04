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
#include "include/cuda_shared_MV_multiply.h"
#define BLOCK_SIZE 17
namespace Nice {

  template <typename T>
  __global__ void CudaSharedMVKernel(T *d_a, T *d_x, T *d_y, int const a_rows, int const x_size) {
    int blockRow = blockIdx.x;
    int blockCol = blockIdx.y;
    int threadCol = threadIdx.x;


    //if (threadRow >= x_size || threadCol >= a_rows) return;
      T * aTile = &d_a[(blockCol * BLOCK_SIZE * a_rows) + (BLOCK_SIZE * blockRow)]; // ACCESSES ALL DO NOT CHANGE
      //T * xTile = &d_x[BLOCK_SIZE * blockCol]; // ACCESSES ALL DO NOT CHANGE
      extern __shared__ T xTile[];
      xTile[threadCol] = d_x[BLOCK_SIZE * blockCol + threadCol];

      __syncthreads();
      float sum = 0.0f;
      for (int i = 0; i < BLOCK_SIZE; i++){
        int xGIndex = blockIdx.y * BLOCK_SIZE + i;
        int yGIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        //int xGIndex = ceil((double)((blockCol * BLOCK_SIZE * a_rows) + (BLOCK_SIZE * blockRow)) / a_rows);
        //int yGIndex = ceil((double)((blockCol * BLOCK_SIZE * a_rows) + (BLOCK_SIZE * blockRow)) / x_size);
        if (xGIndex < x_size && yGIndex < a_rows){
          //printf("Sum = %5.5f pos-add (%i) br: %i bc: %i i = %i tC = %i xG = %i yG = %i : %5.5f * %5.5f = %5.5f\n",
            //sum, threadCol + (blockRow * BLOCK_SIZE), blockRow, blockCol, i, threadCol, xGIndex,
            //yGIndex, aTile[(a_rows * i) + threadCol], xTile[i],
            //aTile[(a_rows * i) + threadCol] * xTile[i]);
          //printf("Sum = %5.5f Row: %i Col %i tx: %i %5.5f * %5.5f = %5.5f\n", sum, xGIndex,
           //yGIndex, threadCol, aTile[(a_rows *i) + threadCol], xTile[i],
           //aTile[(a_rows *i) + threadCol] * xTile[i]);
          atomicAdd(&d_y[threadCol + (blockRow * BLOCK_SIZE)], aTile[(a_rows *i) + threadCol] * xTile[i]);
        }
        __syncthreads();
      }
      //printf("%i: %5.5f\n", threadCol + (blockRow * BLOCK_SIZE), sum);
      __syncthreads();
      //d_y[threadCol + (blockRow * BLOCK_SIZE)] += sum;
  }

  template <typename T>
  Vector<T> CudaSharedMVMultiply<T>::Multiply(const Matrix<T> &a, const Vector<T> &b) {
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
      dim3 dimGrid(std::ceil((float)m / (BLOCK_SIZE)), std::ceil((float)k / (BLOCK_SIZE)));
      std::cout << std::ceil((float)m / (BLOCK_SIZE)) << " " << std::ceil((float)k / (BLOCK_SIZE)) <<"\n";

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
