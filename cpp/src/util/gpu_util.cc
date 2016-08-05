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

#ifdef NEED_CUDA

#include "include/gpu_util.h"

namespace Nice {

//
// Helper functions
//
void gpuAssert(cudaError_t code, const char *file,
               int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
    if (abort) { exit(code); }
    }
}

void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }
void startTime(Timer* timer) {                                                  
    gettimeofday(&(timer->startTime), NULL);                                    
}                                                                               
                                                                                
void stopTime(Timer* timer) {                                                   
    gettimeofday(&(timer->endTime), NULL);                                      
}                                                                               
                                                                                
float elapsedTime(Timer timer) {                                                
    stopTime(&timer);                                                           
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));    
}  
void copyHostToDevice(int * d_data, const int * h_data, int size) {                   
  gpuErrchk(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));          
}                                                                               
void copyHostToDevice(float * d_data, const float * h_data, int size) {               
  gpuErrchk(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));          
}                                                                               
void copyHostToDevice(double * d_data, const double * h_data, int size) {              
  gpuErrchk(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));          
}                                                                               
void copyDeviceToHost(int * h_data, const  int * d_data, int size) {                   
  gpuErrchk(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));          
}                                                                               
void copyDeviceToHost(float * h_data, const float * d_data, int size) {               
  gpuErrchk(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));          
}                                                                               
void copyDeviceToHost(double * h_data, const double * d_data, int size) {              
  gpuErrchk(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));          
} 
                                                 
cudaDeviceProp setDevice() {                                                    
  int deviceCount;                                                              
  int bestDeviceIndex = 0;                                                      
  int maxThreadsPerBlock = 0;                                                   
  int *maxGridDims = 0;                                                         
  int maxThreads = 0;                                                           
  int maxMultiProcessors = 0;                                                   
  char *name;                                                                   
  size_t maxSharedMem = 0;                                                      
                                                                                
  gpuErrchk(cudaGetDeviceCount(&deviceCount));                                  
  cudaDeviceProp gpuDevices[deviceCount];                                       
  for (int i = 0; i < deviceCount; ++i) {                                       
    gpuErrchk(cudaGetDeviceProperties(&gpuDevices[i], i));                      
    if(gpuDevices[i].sharedMemPerBlock >= maxSharedMem &&                       
            gpuDevices[i].multiProcessorCount * 2048 >= maxThreads) {           
      maxThreadsPerBlock = gpuDevices[i].maxThreadsPerBlock;                    
      maxGridDims = gpuDevices[i].maxGridSize;                                  
      maxMultiProcessors = gpuDevices[i].multiProcessorCount;                   
      maxThreads = 2048 * maxMultiProcessors;                                   
      maxSharedMem = gpuDevices[i].sharedMemPerBlock;                           
      gpuErrchk(cudaSetDevice(i));                                              
      name = gpuDevices[i].name;                                                
      bestDeviceIndex = i;                                                      
    }                                                                           
  }                                                                             
  std::cout<<"Number of Devices: "<<deviceCount<<std::endl;                     
  std::cout<<"Name of Best Device: "<<name<<std::endl;                          
  std::cout<<"Max Grid Dims: "<<maxGridDims[0]<<" ";                            
  std::cout<<                   maxGridDims[1]<<" "<<maxGridDims[2]<<std::endl; 
  std::cout<<"Max Number of Threads Per Block: "<<maxThreadsPerBlock<<std::endl;
  std::cout<<"Max Number of Multi Processors: "<<maxMultiProcessors<<std::endl; 
  std::cout<<"Max Number of Threads: "<<maxThreads<<std::endl;                  
  std::cout<<"Max Shared Memory: "<<maxSharedMem<<std::endl;                    
  return gpuDevices[bestDeviceIndex];                                           
}                            
//
// Cusolver wraper functions
//
cusolverStatus_t GpuSvd(cusolverDnHandle_t solver_handle,
           int M,
           int N,
           float * d_A,
           float * d_S,
           float * d_U,
           float * d_V,
           float * work,
           int work_size,
           int * devInfo) {
  return cusolverDnSgesvd(solver_handle,
                          'A', 'A', M, N, d_A, M, d_S,
                          d_U, M, d_V, N, work, work_size,
                          NULL, devInfo);
}

cusolverStatus_t GpuSvd(cusolverDnHandle_t solver_handle,
           int M,
           int N,
           double * d_A,
           double * d_S,
           double * d_U,
           double * d_V,
           double * work,
           int work_size,
           int * devInfo) {
  return cusolverDnDgesvd(solver_handle,
                         'A', 'A', M, N, d_A, M, d_S,
                          d_U, M, d_V, N, work, work_size,
                          NULL, devInfo);
}

cusolverStatus_t GpuGetLUDecompWorkspace(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    float *A,
                                    int lda,
                                    int *Lwork) {
  return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t GpuGetLUDecompWorkspace(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    double *A,
                                    int lda,
                                    int *Lwork) {
  return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t GpuLUDecomposition(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    float *A,
                                    int lda,
                                    float *Workspace,
                                    int *devIpiv, int *devInfo) {
  return cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

cusolverStatus_t GpuLUDecomposition(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    double *A,
                                    int lda,
                                    double *Workspace,
                                    int *devIpiv, int *devInfo) {
  return cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

cusolverStatus_t GpuLinearSolver(cusolverDnHandle_t handle,
                                 cublasOperation_t trans,
                                 int n,
                                 int nrhs,
                                 const float *A,
                                 int lda,
                                 const int *devIpiv,
                                 float *B,
                                 int ldb,
                                 int *devInfo) {
  return cusolverDnSgetrs(handle, trans, n, nrhs, A, lda,
                          devIpiv, B, ldb, devInfo);
}

cusolverStatus_t GpuLinearSolver(cusolverDnHandle_t handle,
                                 cublasOperation_t trans,
                                 int n,
                                 int nrhs,
                                 const double *A,
                                 int lda,
                                 const int *devIpiv,
                                 double *B,
                                 int ldb,
                                 int *devInfo) {
  return cusolverDnDgetrs(handle, trans, n, nrhs, A, lda,
                          devIpiv, B, ldb, devInfo);
}
cusolverStatus_t GpuLuWorkspace(cusolverDnHandle_t handle,
                                int m,
                                int n,
                                float *a,
                                int *worksize) {
  return cusolverDnSgetrf_bufferSize(handle, m, n, a, m, &(*worksize));
}
cusolverStatus_t GpuLuWorkspace(cusolverDnHandle_t handle,
                                int m,
                                int n,
                                double *a,
                                int *worksize) {
  return cusolverDnDgetrf_bufferSize(handle, m, n, a, m, &(*worksize));
}
cusolverStatus_t GpuDeterminant(cusolverDnHandle_t handle,
                                int m,
                                int n,
                                float *a,
                                float *workspace,
                                int *devIpiv,
                                int *devInfo) {
  return cusolverDnSgetrf(handle, m, n, a, m, workspace, devIpiv, devInfo);
}
cusolverStatus_t GpuDeterminant(cusolverDnHandle_t handle,
                                int m,
                                int n,
                                double *a,
                                double *workspace,
                                int *devIpiv,
                                int *devInfo) {
  return cusolverDnDgetrf(handle, m, n, a, m, workspace, devIpiv, devInfo);
}

//
// Cublas wraper functions
//
cublasStatus_t GpuMatrixVectorMul(cublasHandle_t handle,
                                  cublasOperation_t trans,
                                  int m, int n,
                                  const float *alpha,
                                  const float *A, int lda,
                                  const float *x, int incx,
                                  const float *beta,
                                  float *y, int incy) {
  return cublasSgemv(handle, trans, m, n, alpha,
                     A, lda, x, incx, beta, y, incy);
}

cublasStatus_t GpuMatrixVectorMul(cublasHandle_t handle,
                                  cublasOperation_t trans,
                                  int m, int n,
                                  const double *alpha,
                                  const double *A, int lda,
                                  const double *x, int incx,
                                  const double *beta,
                                  double *y, int incy) {
  return cublasDgemv(handle, trans, m, n, alpha,
                     A, lda, x, incx, beta, y, incy);
}

cublasStatus_t GpuMatrixScalarMul(cublasHandle_t handle,
                                  int n,
                                  const float &scalar,
                                  float *a) {
  return cublasSscal(handle, n, &scalar, a, 1);
}

cublasStatus_t GpuMatrixScalarMul(cublasHandle_t handle,
                                  int n,
                                  const double &scalar,
                                  double *a) {
  return cublasDscal(handle, n, &scalar, a, 1);
}

cublasStatus_t GpuMatrixMatrixMul(cublasHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  float *a,
                                  float *b,
                                  float *c) {
  const float alpha = 1.0; const float beta = 0.0;
  return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, k, &alpha, a, m, b, k, &beta, c, m);
}

cublasStatus_t GpuMatrixMatrixMul(cublasHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  double *a,
                                  double *b,
                                  double *c) {
  const double alpha = 1.0; const double beta = 0.0;
  return cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, k, &alpha, a, m, b, k, &beta, c, m);
}

cublasStatus_t GpuMatrixAdd(cublasHandle_t handle,
                            int m,
                            int n,
                            const float *alpha,
                            const float *A, int lda,
                            const float *beta,
                            const float *B, int ldb,
                            float *C, int ldc) {
  return cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t GpuMatrixAdd(cublasHandle_t handle,
                            int m,
                            int n,
                            const double *alpha,
                            const double *A, int lda,
                            const double *beta,
                            const double *B, int ldb,
                            double *C, int ldc) {
  return cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t GpuMatrixMatrixSub(cublasHandle_t handle, int m, int n,
                                  const float *alpha, float *a, int lda,
                                  const float *beta, float *b, int ldb,
                                  float *c, int ldc ) {
  return cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, alpha, a, lda,
                     beta, b, ldb, c, ldc);
}

cublasStatus_t GpuMatrixMatrixSub(cublasHandle_t handle, int m, int n,
                                  const double *alpha, double *a, int lda,
                                  const double *beta, double *b, int ldb,
                                  double *c, int ldc ) {
  return cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, alpha, a, lda,
                     beta, b, ldb, c, ldc);
}

cublasStatus_t GpuVectorVectorDot(cublasHandle_t handle,
                                  int n,
                                  float *a,
                                  float *b,
                                  float *c) {
  return cublasSdot(handle, n, a, 1.0, b, 1.0, c);
}

cublasStatus_t GpuVectorVectorDot(cublasHandle_t handle,
                                  int n,
                                  double *a,
                                  double *b,
                                  double *c) {
  return cublasDdot(handle, n, a, 1.0, b, 1.0, c);
}

cublasStatus_t GpuFrobeniusNorm(cublasHandle_t handle,
                                int n,
                                int incx,
                                float *a,
                                float *c) {
  return cublasSnrm2(handle, n, a, incx, c);
}

cublasStatus_t GpuFrobeniusNorm(cublasHandle_t handle,
                                int n,
                                int incx,
                                double *a,
                                double *c) {
  return cublasDnrm2(handle, n, a, incx, c);
}

}  // namespace Nice
#endif  // Need Cuda
