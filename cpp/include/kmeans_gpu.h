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

// Austin Clyde

#ifndef CPP_INCLUDE_KMEANS_GPU_H_
#define CPP_INCLUDE_KMEANS_GPU_H_

#ifdef CUDA_AND_GPU

#include "include/kdac.h"
#include "include/gpu_util.h"

namespace Nice {
template<typename T>
class KMeansGPU: public KMeans<T> {
 public:
  /// This is the default constructor for KMeansGPU
  /// Number of clusters c and reduced dimension q will be both set to 2
  KMeansGPU() :
      block_limit_(256) {}

  ~KMeansGPU() {
    // Free parameters, intermediate delta and parameters

  }


 private:
  T* x_matrix_d_;  // Input matrix X (n by d) on device
  Tdsx
  // GPUUtil object to setup memory etc.
  GpuUtil<T> *gpu_util_;
  unsigned int block_limit_;


}; // Class KMMEANS_GPU
}  // namespace Nice

#endif  // CUDA_AND_GPU
#endif  // CPP_INCLUDE_KMEANS_GPU_H_