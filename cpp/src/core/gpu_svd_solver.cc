/*
 * SVD.cu
 *
 *  Created on: Jun 16, 2016
 *      Author: cpurc002
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>

#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include<cusolverDn.h>
#include<cuda_runtime_api.h>

/***********************/
/* CUDA ERROR CHECKING */
/***********************/
/*void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) { exit(code); }
   }
}
void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

void printMatrix(float * a, int m, int n, std::string str){
	std::cout<<str<<std::endl;
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			std::cout <<a[i*n + j] << "\t\t";
		}
		std::cout<<std::endl;
	}

}
*/
/********/
/* MAIN */
/********//*
int main(){

    int M = 10;
    int N = 10;

    // --- Setting the host matrix
    float *h_A = (float *)malloc(M * N * sizeof(float));
    for(unsigned int i = 0; i < M; i++){
        for(unsigned int j = 0; j < N; j++){
            h_A[j*M + i] = (i + j) * (i + j);
        }
    }

    // --- Setting the device matrix and moving the host matrix to the device
    float *d_A;         gpuErrchk(cudaMalloc(&d_A,      M * N * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // --- host side SVD results space
    float *h_U = (float *)malloc(M * M * sizeof(float));
    float *h_V = (float *)malloc(N * N * sizeof(float));
    float *h_S = (float *)malloc(N *     sizeof(float));

    // --- device side SVD workspace and matrices
    int work_size = 0;

    int *devInfo;       gpuErrchk(cudaMalloc(&devInfo,          sizeof(int)));
    float *d_U;         gpuErrchk(cudaMalloc(&d_U,      M * M * sizeof(float)));
    float *d_V;         gpuErrchk(cudaMalloc(&d_V,      N * N * sizeof(float)));
    float *d_S;         gpuErrchk(cudaMalloc(&d_S,      N *     sizeof(float)));

    cusolverStatus_t stat;

    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    stat = cusolverDnSgesvd_bufferSize(solver_handle, M, N, &work_size);
    if(stat != CUSOLVER_STATUS_SUCCESS ) std::cout << "Initialization of cuSolver failed. \N";

    float *work;    gpuErrchk(cudaMalloc(&work, work_size * sizeof(float)));
    //float *rwork; gpuErrchk(cudaMalloc(&rwork, work_size * sizeof(float)));

    // --- CUDA SVD execution
    stat = cusolverDnSgesvd(solver_handle, 'A', 'A', M, N, d_A, M, d_S, d_U, M, d_V, N, work, work_size, NULL, devInfo);
    //stat = cusolverDnSgesvd(solver_handle, 'N', 'N', M, N, d_A, M, d_S, d_U, M, d_V, N, work, work_size, NULL, devInfo);
    cudaDeviceSynchronize();

    int devInfo_h = 0;
    gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "devInfo = " << devInfo_h << "\n";

    switch(stat){
        case CUSOLVER_STATUS_SUCCESS:           std::cout << "SVD computation success\n";                       break;
        case CUSOLVER_STATUS_NOT_INITIALIZED:   std::cout << "Library cuSolver not initialized correctly\n";    break;
        case CUSOLVER_STATUS_INVALID_VALUE:     std::cout << "Invalid parameters passed\n";                     break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:    std::cout << "Internal operation failed\n";                     break;
    }

    if (devInfo_h == 0 && stat == CUSOLVER_STATUS_SUCCESS) std::cout    << "SVD successful\n\n";

    // --- Moving the results from device to host
    gpuErrchk(cudaMemcpy(h_S, d_S, N * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_U, d_U, M * M * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_V, d_V, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    printMatrix(h_S,1,N,"d_S");
    std::cout<<std::endl;
    printMatrix(h_U,M,M,"d_U");
    std::cout<<std::endl;
    printMatrix(h_V,N,N,"d_V");

    cusolverDnDestroy(solver_handle);

    return 0;

}



*/
