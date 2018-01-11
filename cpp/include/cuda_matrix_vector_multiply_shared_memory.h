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

#ifndef CPP_INCLUDE_CUDA_MATRIX_VECTOR_MULTIPLY_SHARED_MEMORY_H_
#define CPP_INCLUDE_CUDA_MATRIX_VECTOR_MULTIPLY_SHARED_MEMORY_H_

#ifdef CUDA_AND_GPU

#include <iostream>
#include <cmath>
#include <chrono>
#include "include/gpu_util.h"

namespace Nice {

template<typename T>
class CudaSharedMVMultiply {
 private:
  int block_size;

 public:
  explicit CudaSharedMVMultiply(int inBlock) {
     block_size = inBlock;
  }

  Vector<T> Multiply(const Matrix<T> &a, const Vector<T> &b);
  Vector<T> MapMultiply(const Matrix<T> &a, const Vector<T> &b);
};

}  // namespace Nice
#endif  // NEED_CUDA
#endif  // CPP_INCLUDE_CUDA_MATRIX_VECTOR_MULTIPLY_SHARED_MEMORY_H_
