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

#ifndef CPP_INCLUDE_GPU_SVD_SOLVER_H_
#define CPP_INCLUDE_GPU_SVD_SOLVER_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Eigen/Dense"
#include<unistd.h>
#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include<cusolverDn.h>
#include<cuda_runtime_api.h>
#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"



namespace Nice {

template<typename T>
class GpuSvdSolver {
 public:
   GpuSvdSolver();
   void Compute(const Matrix<T> &A);
   Matrix<T> MatrixU() const;
   Matrix<T> MatrixV() const;
   Vector<T> SingularValues() const;
};
}
/*
        template<typename T>
        void GpuSvdSolver<T>::template Compute(const Matrix<T>& A){

                int ZZZ; // ERROR NOT BEING FOUND BY CMAKE
                // Initilize generally needed and misc variables 
                int work_size = 0;
                int M = A.rows();
                int N = A.cols();
                int *devInfo;   gpuErrchk(cudaMalloc(&devInfo,          sizeof(int)));
                float *work;    gpuErrchk(cudaMalloc(&work, work_size * sizeof(float)));


                // Allocate all host and deviec memories 
                T *h_A =        (T *)malloc(M * N * sizeof(T));
                T *h_U =        (T *)malloc(M * M * sizeof(T));
                T *h_V =        (T *)malloc(N * N * sizeof(T));
                T *h_S =        (T *)malloc(N *     sizeof(T));
                T *d_A;         gpuErrchk(cudaMalloc(&d_A,      M * N * sizeof(T)));
                T *d_U;         gpuErrchk(cudaMalloc(&d_U,      M * M * sizeof(T)));
                T *d_V;         gpuErrchk(cudaMalloc(&d_V,      N * N * sizeof(T)));
                T *d_S;         gpuErrchk(cudaMalloc(&d_S,      N *     sizeof(T)));

                // Map Eigan Matrix A to host matrix h_A and transfer to device matrix d_A

                cgpuErrchk(cudaMemcpy(d_A, h_A, M * N * sizeof(T), cudaMemcpyHostToDevice));

                // Initilize cuSolver 
                cusolverStatus_t stat;
                cusolverDnHandle_t solver_handle;
                cusolverDnCreate(&solver_handle);
                stat = cusolverDnSgesvd_bufferSize(solver_handle, M, N, &work_size);
                if(stat != CUSOLVER_STATUS_SUCCESS ) std::cout << "Initialization of cuSolver failed. \n";

                // Execute and check status of SVD of A
                stat = cusolverDnSgesvd(solver_handle, 'A', 'A', M, N, d_A, M, d_S, d_U, M, d_V, N, work, work_size, NULL, devInfo);

                switch(stat){
                        case CUSOLVER_STATUS_SUCCESS:           std::cout << "SVD computation success\n";                       break;
                        case CUSOLVER_STATUS_NOT_INITIALIZED:   std::cout << "Library cuSolver not initialized correctly\n";    break;
                        case CUSOLVER_STATUS_INVALID_VALUE:     std::cout << "Invalid parameters passed\n";                     break;
                        case CUSOLVER_STATUS_INTERNAL_ERROR:    std::cout << "Internal operation failed\n";                     break;
                }

                // Copy device matrices to host matrices 
                gpuErrchk(cudaMemcpy(h_S, d_S, 1 * N * sizeof(float), cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(h_U, d_U, M * M * sizeof(float), cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(h_V, d_V, N * N * sizeof(float), cudaMemcpyDeviceToHost));
        }
}  // namespace Nice
*/
#endif  // CPP_INCLUDE_GPU_SVD_SOLVER_H_

