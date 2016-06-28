/*
 * GpuSvdSolver.cu
 *
 *  Created on: Jun 16, 2016
 *      Author: cpurc002
 */

#include "include/gpu_svd_solver.h"

namespace Nice {
		
template<typename T>
void GpuSvdSolver<T>::Compute(const Matrix<T> &A){

    	int M = A.rows();
    	int N = A.cols();
    	const T *h_A = &A(0); 

    	// --- Setting the device matrix and moving the host matrix to the device 
    	T *d_A;         gpuErrchk(cudaMalloc(&d_A,      M * N * sizeof(T)));
    	gpuErrchk(cudaMemcpy(d_A, h_A, M * N * sizeof(T), cudaMemcpyHostToDevice));

   	// --- host side SVD results space 
    	s_.resize(M,1); 
    	u_.resize(M,M); 
    	v_.resize(N,N); 

    	// --- device side SVD workspace and matrices 
    	int work_size = 0;
    	int *devInfo;       gpuErrchk(cudaMalloc(&devInfo,          sizeof(int)));
    	T *d_U;         gpuErrchk(cudaMalloc(&d_U,      M * M * sizeof(T)));
    	T *d_V;         gpuErrchk(cudaMalloc(&d_V,      N * N * sizeof(T)));
    	T *d_S;         gpuErrchk(cudaMalloc(&d_S,      N *     sizeof(T)));

    	cusolverStatus_t stat;
    	// --- CUDA solver initialization
    	cusolverDnHandle_t solver_handle;
    	cusolverDnCreate(&solver_handle);
    	stat = cusolverDnSgesvd_bufferSize(solver_handle, M, N, &work_size);
    	if(stat != CUSOLVER_STATUS_SUCCESS ) std::cout << "Initialization of cuSolver failed. \n";
    	T *work;    gpuErrchk(cudaMalloc(&work, work_size * sizeof(T)));

    	// --- CUDA SVD execution
    	stat = cusolverDnSgesvd(solver_handle, 'A', 'A', M, N, d_A, M, d_S, d_U, M, d_V, N, work, work_size, NULL, devInfo);
    	cudaDeviceSynchronize();

    	int devInfo_h = 0;
    	gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    	std::cout << "devInfo = " << devInfo_h << "\n";

    	switch(stat){
        	case CUSOLVER_STATUS_SUCCESS:           std::cout << "SVD computation success\n";                       break;
        	case CUSOLVER_STATUS_NOT_INITIALIZED:   std::cout << "Library cuSolver not initialized correctly\n";    break;
        	case CUSOLVER_STATUS_INVALID_VALUE:     std::cout << "Invalid parameters passed\n";                     break;
        	case CUSOLVER_STATUS_INTERNAL_ERROR:    std::cout << "Internal operation failed\n";                     break;
        	default: break;
        }
    
    	if (devInfo_h == 0 && stat == CUSOLVER_STATUS_SUCCESS) std::cout    << "SVD successful\n\n";
    	std::cout<<std::endl;

    	// --- Moving the results from device to host
    	gpuErrchk(cudaMemcpy(&s_(0,0), d_S, N * sizeof(T), cudaMemcpyDeviceToHost));
    	gpuErrchk(cudaMemcpy(&u_(0,0), d_U, M * M * sizeof(T), cudaMemcpyDeviceToHost));
    	gpuErrchk(cudaMemcpy(&v_(0,0), d_V, N * N * sizeof(T), cudaMemcpyDeviceToHost));

	cudaFree(d_S); cudaFree(d_U); cudaFree(d_V); 
   	cusolverDnDestroy(solver_handle);
}

	template class GpuSvdSolver<float>;
//	template class GpuSvdSolver<double>;

}


