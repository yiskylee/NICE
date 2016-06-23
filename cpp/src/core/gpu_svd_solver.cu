/*
 * GpuSvdSolver.cu
 *
 *  Created on: Jun 16, 2016
 *      Author: cpurc002
 */

#include "include/gpu_svd_solver.h"

namespace Nice {
/*	template<typename T>
	void BufferToEigenMap(T* h_M, Matrix<T> E_M, int M, int N){
		for(int i = 0; i < M; ++i){
			for(int j = 0; j < N; ++j){
				E_M(i,j) = *(h_M + i + j*N); 
			}
		}
	}

	template<typename T>
	void BufferToEigenMap(T* h_M, Vector<T> E_M, int M, int N){
		for(int i = 0; i < M; ++i){
			for(int j = 0; j < N; ++j){
				E_M(i,j) = *(h_M + i + j*N); 
			}
		}
	}
        
        				
	template<typename T>
	void EigenToBufferMap(T* h_M, Matrix<T> E_M, int M, int N){
		for(int i = 0; i < M; ++i){
			for(int j = 0; j < N; ++j){ 
				*(h_M + j + i*N) = E_M(i,j); 
			}
		}  
	}
*/				
	template<typename T>
	void GpuSvdSolver<T>::Compute(const Matrix<T>& A){
		//------------------------------------------------------------------------
		// TO BE DONE
		// cuSolver cant take ints, only float doublt cuComplex cuComplexDouble
		//------------------------------------------------------------------------		
//		int i
		// Initilize generally needed and misc variables 
		int work_size = 0;		
		int M = A.rows();
    		int N = A.cols();
		int *devInfo;   gpuErrchk(cudaMalloc(&devInfo,          sizeof(int)));	
		float *work;    gpuErrchk(cudaMalloc(&work, work_size * sizeof(float)));
			
		// Allocate all host and deviec memories 
		//T *h_A = 	(T *)malloc(M * N * sizeof(T)); 
		const T *h_A = &A(0); 
		T *h_U = &u_(0); 	//(T *)malloc(M * M * sizeof(T));
    		T *h_V = &v_(0);	//(T *)malloc(N * N * sizeof(T));
    		T *h_S = &s_(0);	//(T *)malloc(N *     sizeof(T));
		T *d_A;         gpuErrchk(cudaMalloc(&d_A,      M * N * sizeof(T)));
    		T *d_U;         gpuErrchk(cudaMalloc(&d_U,      M * M * sizeof(T)));
    		T *d_V;         gpuErrchk(cudaMalloc(&d_V,      N * N * sizeof(T)));
    		T *d_S;         gpuErrchk(cudaMalloc(&d_S,      N *     sizeof(T)));

		// Map Eigan Matrix A to host matrix h_A and transfer to device matrix d_A
	        //EigenToBufferMap(h_A,A,M,N); 
		
		gpuErrchk(cudaMemcpy(d_A, h_A, M * N * sizeof(T), cudaMemcpyHostToDevice));
	
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
			default: break; 
    		}
		
		// Copy device matrices to host matrices 
		gpuErrchk(cudaMemcpy(h_S, d_S, 1 * N * sizeof(T), cudaMemcpyDeviceToHost));
    		gpuErrchk(cudaMemcpy(h_U, d_U, M * M * sizeof(T), cudaMemcpyDeviceToHost));
	    	gpuErrchk(cudaMemcpy(h_V, d_V, N * N * sizeof(T), cudaMemcpyDeviceToHost));
		
		// Linking host matrices to Eigen matrices
		//&s_(0) = h_S; 
		// Mapping host buffer to Eigen matrix
		// SingularValues() = h_S; 	
		//BufferToEigenMap(h_S,s_,1,N); 
		// MatrixU() = h_U; 
		//BufferToEigenMap(h_U,u_,M,M); 
		// MatrixV() = h_V; 
		//BufferToEigenMap(h_V,v_,N,N); 

	}
	
	template class GpuSvdSolver<float>;
//	template class GpuSvdSolver<double>;

}


