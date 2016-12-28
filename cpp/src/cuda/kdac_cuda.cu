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

#include "include/kdac_cuda.h"
#include "include/gpu_util.h"

namespace Nice {

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

template <typename T>
__global__ void GPUGenDeltaKernel(const T *x_matrix_d,
                                      const int n,
                                      const int d,
                                      T *all_delta_ijs_d) {

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  // This is to index an n x n matrix where each cell is a
  // d x d matrix. No matter what orientation (row or column) the
  // d x d matrix is, to find the starting location of the (i, j)
  // matrix, we just need to use the following to do so
  if (i < n && j < n) {
    T *delta_ij = all_delta_ijs_d + IDXR(i, j, n) * d;
    // x_matrix_d is column major
    for (int k = 0; k < d; k++)
      delta_ij[k] = x_matrix_d[IDXC(i, k, n)] - x_matrix_d[IDXC(j, k, n)];
  }
}

template <typename T>
__global__ void GPUGenAMatricesKernel(const T *x_matrix_d,
                                      const int n,
                                      const int d,
                                      T *a_matrices_d) {
  T *delta_ij = SharedMemory<T>();
  int tx = threadIdx.x;
  int i = blockIdx.y;
  int j = blockIdx.x;

  if (tx < d) {
    T *a_ij = a_matrices_d + IDXR(i, j, n) * (d * d);
    delta_ij[tx] = x_matrix_d[IDXC(i, tx, n)] - x_matrix_d[IDXC(j, tx, n)];
    syncthreads();
    // thread tx calculates a whole row tx of the output matrix a_ij
    for (int col = 0; col < d; col++)
      a_ij[IDXC(tx, col, d)] = delta_ij[col] * delta_ij[tx];

  }
}

template <typename T>
__global__ void GPUGenPhiCoeffKernel(const T *w_l_d,
                                     const T *gradient_d,
                                     const T *a_matrices_d,
                                     const int n,
                                     const int d,
                                     const float *alpha_d,
                                     const float *beta_d,
                                     const int *incx_d,
                                     const int *incy_d,
                                     T *waw_matrix_d,
                                     T *waf_matrix_d,
                                     T *faf_matrix_d,
                                     T *temp_d,
                                     cublasStatus_t *return_status) {

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i < n) && (j < n)) {
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      *(return_status + IDXR(i, j, n)) = status;
      return;
    }
    const T *a_ij_matrix = a_matrices_d + IDXR(i, j, n) * (d * d);
    T *temp_ij = temp_d + IDXR(i, j, n) * d;

    // Calculate waw
//    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//                1, d, d,
//                &alpha,
//                w_l_d, 1,
//                a_ij_matrix, d,
//                &beta,
//                temp_ij, 1);

    status = cublasSgemv(handle, CUBLAS_OP_N,
                         d, d,
                         alpha_d,
                         a_ij_matrix, d,
                         w_l_d, *incx_d,
                         beta_d,
                         temp_ij, *incy_d);

    cublasSdot(handle, d,
               temp_ij, *incx_d,
               w_l_d, *incy_d,
               &waw_matrix_d[IDXC(i, j, n)]);

    // Calculate waf
    // temp_ij is the intermediate result of w_l.transpose() * a_matrix_ij
    // So here we are only going to use cublasSdot and reuse temp_ij
    cublasSdot(handle, d,
               temp_ij, *incx_d,
               gradient_d, *incy_d,
               &waf_matrix_d[IDXC(i, j, n)]);

    // Calculate faf
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                1, d, d,
                alpha_d,
                gradient_d, 1,
                a_ij_matrix, d,
                beta_d,
                temp_ij, 1);

    cublasSdot(handle, d,
               temp_ij, *incx_d,
               gradient_d, *incy_d,
               &faf_matrix_d[IDXC(i, j, n)]);

    cublasDestroy(handle);
    *(return_status + IDXR(i, j, n)) = status;
  }
}

template <typename T>
__global__ void GPUGenPhiKernel(const T alpha,
                                const T sqrt_one_minus_alpha,
                                const T denom,
                                const T *waw_matrix_d,
                                const T *waf_matrix_d,
                                const T *faf_matrix_d,
                                const T *gamma_matrix_d,
                                const int n,
                                const int d,
                                bool w_l_changed,
                                float *phi_of_alphas_d,
                                float *phi_of_zeros_d,
                                float *phi_of_zero_primes_d) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i < n) && (j < n)) {
    T waw = waw_matrix_d[IDXC(i, j, n)];
    T waf = waf_matrix_d[IDXC(i, j, n)];
    T faf = faf_matrix_d[IDXC(i, j, n)];
    T gammaij = gamma_matrix_d[IDXC(i, j, n)];
    T kij = expf(denom * ((faf - waw) * (alpha*alpha) +
        2 * waf * sqrt_one_minus_alpha * alpha + waw));
    phi_of_alphas_d[IDXC(i, j, n)] = gammaij * kij;
    if(w_l_changed) {
      T kij = expf(denom * waw);
      phi_of_zeros_d[IDXC(i, j, n)] = gammaij * kij;
      phi_of_zero_primes_d[IDXC(i, j, n)] =
          gammaij * denom * 2 * waf * kij;
    }
  }
}

template <typename T>
__global__ void GPUGenPhiTransposeKernel(const T alpha,
                                const T sqrt_one_minus_alpha,
                                const T denom,
                                const T *waw_matrix_d,
                                const T *waf_matrix_d,
                                const T *faf_matrix_d,
                                const T *gamma_matrix_d,
                                const int n,
                                const int d,
                                bool w_l_changed,
                                float *phi_of_alphas_d,
                                float *phi_of_zeros_d,
                                float *phi_of_zero_primes_d) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i < n) && (j < n)) {
    T waw = waw_matrix_d[IDXC(j, i, n)];
    T waf = waf_matrix_d[IDXC(j, i, n)];
    T faf = faf_matrix_d[IDXC(j, i, n)];
    T gammaij = gamma_matrix_d[IDXC(j, i, n)];
    T kij = expf(denom * ((faf - waw) * (alpha*alpha) +
        2 * waf * sqrt_one_minus_alpha * alpha + waw));
    phi_of_alphas_d[IDXC(j, i, n)] = gammaij * kij;
    if(w_l_changed) {
      T kij = expf(denom * waw);
      phi_of_zeros_d[IDXC(j, i, n)] = gammaij * kij;
      phi_of_zero_primes_d[IDXC(j, i, n)] =
          gammaij * denom * 2 * waf * kij;
    }
  }
}



// From CUDA SDK
template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce_kernel(T *g_idata, T *g_odata, unsigned int n) {
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n)
  {
    mySum += g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + blockSize < n)
      mySum += g_idata[i+blockSize];

    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  __syncthreads();


  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256))
  {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  __syncthreads();

  if ((blockSize >= 256) &&(tid < 128))
  {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  __syncthreads();

  if ((blockSize >= 128) && (tid <  64))
  {
    sdata[tid] = mySum = mySum + sdata[tid +  64];
  }

  __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
  if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
  // fully unroll reduction within a single warp
  if ((blockSize >=  64) && (tid < 32))
  {
    sdata[tid] = mySum = mySum + sdata[tid + 32];
  }

  __syncthreads();

  if ((blockSize >=  32) && (tid < 16))
  {
    sdata[tid] = mySum = mySum + sdata[tid + 16];
  }

  __syncthreads();

  if ((blockSize >=  16) && (tid <  8))
  {
    sdata[tid] = mySum = mySum + sdata[tid +  8];
  }

  __syncthreads();

  if ((blockSize >=   8) && (tid <  4))
  {
    sdata[tid] = mySum = mySum + sdata[tid +  4];
  }

  __syncthreads();

  if ((blockSize >=   4) && (tid <  2))
  {
    sdata[tid] = mySum = mySum + sdata[tid +  2];
  }

  __syncthreads();

  if ((blockSize >=   2) && ( tid <  1))
  {
    sdata[tid] = mySum = mySum + sdata[tid +  1];
  }

  __syncthreads();
#endif

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = mySum;
}


template<typename T>
void GPUGenAMatrices(const T *x_matrix_d,
                     const int n,
                     const int d,
                     T *a_matrices_d) {

  int block_size = nextPow2(d);
  int shared_mem_size = d * sizeof(T);

  dim3 dim_block(block_size, 1);
  dim3 dim_grid(n, n);
  GPUGenAMatricesKernel
      <<<dim_grid, dim_block, shared_mem_size>>>
      (x_matrix_d, n, d, a_matrices_d);
}

// Explicit Instantiation
template
void GPUGenAMatrices<float>(const float *x_matrix_d,
                            const int n,
                            const int d,
                            float *a_matrices_d);

template <typename T>
void GPUGenPhiCoeff(const T *w_l_d,
                    const T *gradient_d,
                    const T *a_matrices_d,
                    const CUBLASParams &params,
                    const int n,
                    const int d,
                    T *a_mul_w_d,
                    T *a_mul_grad_d,
                    T *waw_matrix,
                    T *waf_matrix,
                    T *faf_matrix) {

  int num_streams = n * n;
  cudaStream_t *streams = new cudaStream_t [num_streams];
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  for (int i = 0; i < num_streams; i++)
    CUDA_CALL(cudaStreamCreate(&streams[i]));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      const T *a_ij_matrix = a_matrices_d + IDXR(i, j, n) * (d * d);
      T *a_mul_w_ij = a_mul_w_d + IDXR(i, j, n) * d;
      T *a_mul_grad_ij = a_mul_grad_d + IDXR(i, j, n) * d;
      CUBLAS_CALL(cublasSetStream(handle, streams[i * n + j]));
      CUBLAS_CALL(cublasSgemv(handle, CUBLAS_OP_N,
                              d, d,
                              &(params.alpha),
                              a_ij_matrix, d,
                              w_l_d, params.incx,
                              &(params.beta),
                              a_mul_w_ij, params.incy));
//      CUBLAS_CALL(cublasSdot(handle, d,
//                             a_mul_w_ij, params.incx,
//                             w_l_d, params.incy,
//                             &waw_matrix[IDXC(i, j, n)]));
//      CUBLAS_CALL(cublasSdot(handle, d,
//                             a_mul_w_ij, params.incx,
//                             gradient_d, params.incy,
//                             &waf_matrix[IDXC(i, j, n)]));
      CUBLAS_CALL(cublasSgemv(handle, CUBLAS_OP_N,
                              d, d,
                              &(params.alpha),
                              a_ij_matrix, d,
                              gradient_d, params.incx,
                              &(params.beta),
                              a_mul_grad_ij, params.incy));
//      CUBLAS_CALL(cublasSdot(handle, d,
//                             a_mul_grad_ij, params.incx,
//                             gradient_d, params.incy,
//                             &faf_matrix[IDXC(i, j, n)]));
    }
  }
  CUDA_CALL(cudaMemset(a_mul_w_d, 0, n*n*d*sizeof(T)));
  CUDA_CALL(cudaMemset(a_mul_grad_d, 0, n*n*d*sizeof(T)));
  for (int i = 0; i < num_streams; i++)
    CUDA_CALL(cudaStreamDestroy(streams[i]));
  CUBLAS_CALL(cublasDestroy(handle));
}

template
void GPUGenPhiCoeff<float>(const float *w_l_d,
                           const float *gradient_d,
                           const float *a_matrices_d,
                           const CUBLASParams &params,
                           const int n,
                           const int d,
                           float *a_mul_w_d,
                           float *a_mul_grad_d,
                           float *waw_matrix,
                           float *waf_matrix,
                           float *faf_matrix);


void GetNumBlocksAndThreads(int num_elements,
                            int max_blocks,
                            int max_threads,
                            int &blocks,
                            int &threads) {
  cudaDeviceProp prop;
  int device;
  CUDA_CALL(cudaGetDevice(&device));
  CUDA_CALL(cudaGetDeviceProperties(&prop, device));

  threads = (num_elements < max_threads) ?
            nextPow2(num_elements) : max_threads;
  blocks = (num_elements + (threads * 2 - 1)) / (threads * 2);

  if ((float)threads*blocks > (float)prop.maxGridSize[0] *
      prop.maxThreadsPerBlock) {
    printf("Number of elements is too large\n");
    exit(1);
  }
  if (blocks > prop.maxGridSize[0]) {
    printf("Grid size <%d> excceeds the device capability <%d>, "
               "set block size as %d (original %d)\n",
           blocks, prop.maxGridSize[0], threads*2, threads);

    blocks /= 2;
    threads *= 2;
  }
  blocks = (max_blocks < blocks) ? max_blocks : blocks;
}

bool isPow2(unsigned int x) {
  return ((x&(x-1))==0);
}


template <typename T>
void reduce(int num_elements, int num_threads, int num_blocks,
            T *input_data_d, T *output_data_d) {



  dim3 dim_block(num_threads, 1, 1);
  dim3 dim_grid(num_blocks, 1, 1);

  int shared_mem_size = (num_threads <= 32) ?
                        2 * num_threads * sizeof(T) : num_threads * sizeof(T);
  if (isPow2(num_elements)) {
    switch (num_threads) {
      case 512:
        reduce_kernel < T, 512, true ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 256:
        reduce_kernel < T, 256, true ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 128:
        reduce_kernel < T, 128, true ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 64:
        reduce_kernel < T, 64, true ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 32:
        reduce_kernel < T, 32, true ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 16:
        reduce_kernel < T, 16, true ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 8:
        reduce_kernel < T, 8, true ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 4:
        reduce_kernel < T, 4, true ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 2:
        reduce_kernel < T, 2, true ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 1:
        reduce_kernel < T, 1, true ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
    }
  } else {
    switch (num_threads) {
      case 512:
        reduce_kernel < T, 512, false
            ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 256:
        reduce_kernel < T, 256, false
            ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 128:
        reduce_kernel < T, 128, false
            ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 64:
        reduce_kernel < T, 64, false ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 32:
        reduce_kernel < T, 32, false ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 16:
        reduce_kernel < T, 16, false ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 8:
        reduce_kernel < T, 8, false ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 4:
        reduce_kernel < T, 4, false ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 2:
        reduce_kernel < T, 2, false ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
      case 1:
        reduce_kernel < T, 1, false ><<<dim_grid, dim_block, shared_mem_size>>>
        (input_data_d, output_data_d, num_elements);
        break;
    }
  }
}

template
void reduce<float>(int num_elements, int num_threads, int num_blocks,
            float *input_data_d, float *output_data_d);



template<typename T>
void GPUGenPhi(const T alpha,
               const T sqrt_one_minus_alpha,
               const T denom,
               const T *waw_matrix_d,
               const T *waf_matrix_d,
               const T *faf_matrix_d,
               const T *gamma_matrix_d,
               const int n,
               const int d,
               const bool w_l_changed,
               T *phi_of_alphas_in_d,
               T *phi_of_zeros_in_d,
               T *phi_of_zero_primes_in_d) {
  int block_size = 16;

  dim3 dim_block(block_size, block_size);
  // If matrix is n x m, then I need an m x n grid for contiguous
  // memory access
  dim3 dim_grid( (n-1) / block_size + 1, (n-1) / block_size + 1);

  GPUGenPhiTransposeKernel<<<dim_grid, dim_block>>>(alpha,
                                           sqrt_one_minus_alpha,
                                           denom,
                                           waw_matrix_d,
                                           waf_matrix_d,
                                           faf_matrix_d,
                                           gamma_matrix_d,
                                           n,
                                           d,
                                           w_l_changed,
                                           phi_of_alphas_in_d,
                                           phi_of_zeros_in_d,
                                           phi_of_zero_primes_in_d);

  // Check if error happens in kernel launch
  CUDA_CALL(cudaGetLastError());

  int num_blocks = 0;
  int num_threads = 0;
  int num_elems = n * n;
  int max_blocks = 64;
  int max_threads = 256;
  GetNumBlocksAndThreads(num_elems,
                         max_blocks,
                         max_threads,
                         num_blocks,
                         num_threads);

  // Each block generates a partial sum
  T *phi_of_alphas_out_d = 0;
  CUDA_CALL(cudaMalloc((void**) &phi_of_alphas_out_d,
                       num_blocks * sizeof(T)));
  reduce<T>(num_elems, num_threads, num_blocks,
            phi_of_alphas_in_d, phi_of_alphas_out_d);
  CUDA_CALL(cudaGetLastError());
  T *phi_of_alphas_out_h = new T[n * n];
  CUDA_CALL(cudaMemcpy(phi_of_alphas_out_h, phi_of_alphas_out_d,
                       num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
  T phi_of_alpha = 0;
  for (int i = 0; i < num_blocks; i++)
    phi_of_alpha += phi_of_alphas_out_h[i];
//  printf("phi(alpha) on gpu: %f\n", phi_of_alpha);
  CUDA_CALL(cudaFree(phi_of_alphas_out_d));
  delete [] phi_of_alphas_out_h;

  if (w_l_changed) {
    T *phi_of_zeros_out_d = 0;
    T *phi_of_zero_primes_out_d = 0;
    CUDA_CALL(cudaMalloc((void**) &phi_of_zeros_out_d,
                         num_blocks * sizeof(T)));
    CUDA_CALL(cudaMalloc((void**) &phi_of_zero_primes_out_d,
                         num_blocks * sizeof(T)));
    reduce<T>(num_elems, num_threads, num_blocks,
              phi_of_zeros_in_d, phi_of_zeros_out_d);
    CUDA_CALL(cudaGetLastError());
    reduce<T>(num_elems, num_threads, num_blocks,
              phi_of_zero_primes_in_d, phi_of_zero_primes_out_d);
    CUDA_CALL(cudaGetLastError());
    T *phi_of_zeros_out_h = new T[n * n];
    T *phi_of_zero_primes_out_h = new T[n * n];
    CUDA_CALL(cudaMemcpy(phi_of_zeros_out_h, phi_of_zeros_out_d,
                         num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(phi_of_zero_primes_out_h, phi_of_zero_primes_out_d,
                         num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
    T phi_of_zero = 0;
    T phi_of_zero_prime = 0;
    for (int i = 0; i < num_blocks; i++) {
      phi_of_zero += phi_of_zeros_out_h[i];
      phi_of_zero_prime += phi_of_zero_primes_out_h[i];
    }
//    printf("phi(0) on gpu: %f\n", phi_of_zero);
//    printf("phi(0)' on gpu: %f\n", phi_of_zero_prime);
    CUDA_CALL(cudaFree(phi_of_zeros_out_d));
    CUDA_CALL(cudaFree(phi_of_zero_primes_out_d));
    delete [] phi_of_zeros_out_h;
    delete [] phi_of_zero_primes_out_h;
  }
//  if (w_l_changed) {
//    reduce<T>(num_elems, num_threads, num_blocks,
//              phi_of_zeros_in_d, phi_of_zeros_out_d);
//    reduce<T>(num_elems, num_threads, num_blocks,
//              phi_of_zero_primes_in_d, phi_of_zero_primes_out_d);
//  }


}



template
void GPUGenPhi<float>(const float alpha,
                      const float sqrt_one_minus_alpha,
                      const float denom,
                      const float *waw_matrix_d,
                      const float *waf_matrix_d,
                      const float *faf_matrix_d,
                      const float *gamma_matrix_d,
                      const int n,
                      const int d,
                      const bool w_l_changed,
                      float *phi_of_alphas_d,
                      float *phi_of_zeros_d,
                      float *phi_of_zero_primes_d);


}  // Namespace NICE