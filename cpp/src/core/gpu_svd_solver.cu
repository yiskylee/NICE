/*
 * GpuSvdSolver.cu
 *
 *  Created on: Jun 16, 2016
 *      Author: cpurc002
 */

#include "include/gpu_svd_solver.h"

/*
#include<cuda_runtime.h>
#include "device_launch_parameters.h"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"
#include "Eigen/Dense"
#include<unistd.h>
#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include<cusolverDn.h>
#include<cuda_runtime_api.h>


*/
namespace Nice {
/*
template<typename T>
class GpuSvdSolver {
 public:
   GpuSvdSolver();
   void Compute(const Matrix<T> &A);
   Matrix<T> MatrixU() const;
   Matrix<T> MatrixV() const;
   Vector<T> SingularValues() const;
};
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
		T *h_A = 	(T *)malloc(M * N * sizeof(T)); 
		T *h_U = 	(T *)malloc(M * M * sizeof(T));
    		T *h_V = 	(T *)malloc(N * N * sizeof(T));
    		T *h_S = 	(T *)malloc(N *     sizeof(T));
		T *d_A;         gpuErrchk(cudaMalloc(&d_A,      M * N * sizeof(T)));
    		T *d_U;         gpuErrchk(cudaMalloc(&d_U,      M * M * sizeof(T)));
    		T *d_V;         gpuErrchk(cudaMalloc(&d_V,      N * N * sizeof(T)));
    		T *d_S;         gpuErrchk(cudaMalloc(&d_S,      N *     sizeof(T)));

		// Map Eigan Matrix A to host matrix h_A and transfer to device matrix d_A

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
		
	
	}
	
//	template class GpuSvdSolver<int>;
	template class GpuSvdSolver<float>;
//	template class GpuSvdSolver<double>;

}


