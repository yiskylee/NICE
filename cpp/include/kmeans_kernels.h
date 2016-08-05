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
#ifndef CPP_INCLUDE_KERNELS_H_
#define CPP_INCLUDE_KERNELS_H_

#ifdef NEED_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cusolverDn.h>
#include <ctime>
#include <time.h>
#include <iostream>
#include <sys/time.h>                                                           
#include <string>                                                               
#include <stdint.h>                                                             
#include "include/matrix.h"
#include "include/vector.h"

namespace Nice {

template <typename T>
__global__ void classifyPoints(T *data, T *clusters,                            
                               T *distances, int *labels,                   
                               int numK, int numObjs, int numDims);

template <typename T>
void doKmeans(cudaDeviceProp DeviceProp, T *data, T *clusters, int *labels,     
              T *distances, int numK, int numObjs, int numDims) ;
}  // namespace Nice

#endif  // NEED_CUDA
#endif  // CPP_INCLUDE_KERNELS_H_
