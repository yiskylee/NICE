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

#ifndef CPP_INCLUDE_GPU_KMEANS_H_ 
#define CPP_INCLUDE_GPU_KMEANS_H_                                           
#ifdef NEED_CUDA                                                                
                                                                                
#include <stdlib.h>                                                             
#include <time.h>                                                               
#include<cuda_runtime.h>                                                        
#include<device_launch_parameters.h>                                            
#include<cuda_runtime_api.h>                                                    
#include <cublas_v2.h>                                                          
#include <cusolverDn.h>              
#include <Eigen/Dense>                                            
#include <unistd.h>                                                             
#include <stdexcept>                                                            
#include <ctime>                                                                
                                                                                
#include <iostream>                                                             
                                                                                
#include "include/matrix.h"                                                     
#include "include/vector.h"                                                     
#include "include/gpu_util.h"                                                   
#include "include/gpu_svd_solver.h"                                             
#include "include/util.h" 
#include "include/kmeans_kernels.h"
                                                                               
namespace Nice {                                                                
                                                                                
// Abstract class of common matrix operation interface                          
template <typename T>                                                           
class GpuKmeans {                                                               
 private:                                                                       
   int num_clusters_;                                                           
   int num_iterations_;                                                         
   int num_points_;                                                             
   Matrix<T> *data_points_;                                                     
   Vector<int> *labels_;                                                         
   Matrix<T> *centers_;                                                         
  
   Matrix<T> setInitalClusters(const Matrix<T> &data, Matrix<T> &clusters,
                               int numK, int numObjs, int numDims)     
   {                                                                               
     int stride = numObjs / numK ;
     std::cout<<"stride"<<stride<<std::endl;                                                                                      
     std::cout<<"numK"<<numK<<std::endl; 
     std::cout<<"numDims"<<numDims<<std::endl;
     for (int i = 0; i < numK; ++i) {                                              
       for (int j = 0; j < numDims; ++j) {                                          
        clusters(i, j) = data(stride * i, j);                                      
       }                                                                           
     }                                                                             
     return clusters;                                                              
   } 
   void cuda_main(const Matrix<T> &h_data, int numK) {                                                                               
     Timer timer;                                                                
     int numObjs = h_data.rows(); 
     int numDims = h_data.cols(); 


     std::cout<<"numObjs : "<<numObjs<<std::endl;                                                      
                                                                                 
     // Initialize host variables ----------------------------------------------                                                                             
     startTime(&timer);                                                          
     std::cout<<"\nProfiling for best device"<<std::endl;                        
     cudaDeviceProp deviceProp = setDevice();                                    
     std::cout<<"Finished profiling for best device"<<std::endl;                 
     std::cout<<"  Elapsed time to profile for best device ";                    
     std::cout<<elapsedTime(timer)<<std::endl;                                   
                                                                                 
                                                                                 
     startTime(&timer);                                                          
     std::cout<<"\nSetting inital clusters"<<std::endl;    
     Matrix<T> h_clusters(numK, numDims);                      
     setInitalClusters(h_data, h_clusters, numK, numObjs, numDims);    
     std::cout<<"Finished Setting inital cluters"<<std::endl;                    
     std::cout<<"  Elapsed time to set initial clusters: ";                      
     std::cout<<elapsedTime(timer)<<std::endl;                                   
                                                                                 
                                                                                 
     startTime(&timer);                                                          
     std::cout<<"\nAllocating and transfering memories to the GPU"<<std::endl;   
     int dataSize = numObjs * numDims * sizeof(T);                           
     int clusterSize = numK * numDims * sizeof(T);                           
     int labelSize = numObjs * sizeof(int);
     int distanceSize = numK * numObjs * sizeof(T); 
     T *d_distances;
     gpuErrchk(cudaMalloc((void **)&d_distances, distanceSize));                                 
     cudaMemset(d_distances, 0, distanceSize); 

     T *d_data; gpuErrchk(cudaMalloc((void **)&d_data, dataSize));           
     T *d_clusters; gpuErrchk(cudaMalloc((void**)&d_clusters, clusterSize)); 
     int *d_labels; gpuErrchk(cudaMalloc((void **)&d_labels, labelSize));        
     gpuErrchk(cudaMemcpy(d_data, &h_data(0), dataSize, 
               cudaMemcpyHostToDevice));
     gpuErrchk(cudaMemcpy(d_clusters, &h_clusters(0), clusterSize, 
               cudaMemcpyHostToDevice));                      
     std::cout<<"Finished allocating and transfering memories"<<std::endl;       
     std::cout<<"  Elapsed time to allocate and transfer memories ";             
     std::cout<<elapsedTime(timer)<<std::endl;                                   
                                                                                 
                                                                                 
     startTime(&timer);                                                          
     std::cout<<"\nRunning Kmeans"<<std::endl;                                   
     doKmeans<T>(deviceProp, d_data, d_clusters, d_labels, d_distances, 
              numK, numObjs, numDims); 
     std::cout<<"Finished running Kmeans"<<std::endl;                            
     std::cout<<"  Elapsed time to run Kmeans: ";                                
     std::cout<<elapsedTime(timer)<<std::endl;                                   
                                                                                 
                                                                                 
     startTime(&timer);                                                          
     std::cout<<"\nTransfering data back from GPU"<<std::endl;                   
     Matrix<T> h_data_ret(numObjs, numDims);                            
     Matrix<T> h_clusters_ret(numK, numDims);                        
     Vector<int> h_labels(numObjs);
     Matrix<T> h_distances(numObjs, numK);
     gpuErrchk(cudaMemcpy(&h_data_ret(0, 0), d_data, dataSize, 
               cudaMemcpyDeviceToHost)); 
     gpuErrchk(cudaMemcpy(&h_clusters_ret(0, 0), d_clusters, clusterSize, 
               cudaMemcpyDeviceToHost));
     gpuErrchk(cudaMemcpy(&h_labels(0), d_labels, labelSize, 
               cudaMemcpyDeviceToHost));
     gpuErrchk(cudaMemcpy(&h_distances(0, 0), d_distances, distanceSize, 
               cudaMemcpyDeviceToHost)); 
     std::cout<<"Finished transfering data back from GPU"<<std::endl;            
     std::cout<<"  Elapsed time to tranfer memories ";                           
     std::cout<<elapsedTime(timer)<<std::endl;
  
     int n = numObjs / numK;                                                               
     Vector<int> labels_ref(numObjs);                                 
     for (int i = 0; i < numK; ++i) {                                            
       for (int j = 0; j < n; ++j) {                                          
          labels_ref(i * n + j) = i;                                      
       }                                                                         
     }                                                                           

     std::cout<<"\nPredicted Labels: "<<std::endl;                               
     for (int i = 0; i < numObjs/1000; i = i + 1) {                                   
       std::cout<<h_labels(i)<<std::endl;                                    
     }                                                                           
                                                                                 
//     std::cout<<"\nRef Labels: "<<std::endl;                                     
//     for (int i = 0; i < numObjs; i = i + n/2) {                                   
//       std::cout<<labels_ref(i)<<std::endl;                                  
//     }                                                                           
    
     std::cout<<"\nVerifying Labels"<<std::endl; 
     bool labelsPassed = true; 
     for (int i = 0; i < numObjs; ++i)  
       if (labels_ref(i) != h_labels(i) && labelsPassed) 
         labelsPassed = false; 
     if (labelsPassed) 
        std::cout<<"Labels Passed"<<std::endl; 
     else
        std::cout<<"Labels Failed"<<std::endl;  
     
/*     std::cout<<"\nClusters ret: "<<std::endl;                                   
     for (int i = 0; i < numK; ++i) {                                            
       for (int j = 0; j < numDims; ++j) {                                       
         std::cout<<h_clusters_ret(i, j)<<" ";                                   
       }                                                                         
       std::cout<<std::endl;                                                     
     }
 
                                                                             
     std::cout<<"\nData input / ret: "<<std::endl;   
     std::cout<<h_data<<"\n\n"<<std::endl;                                     
     for (int i = 0; i < numObjs; ++i) {                                         
       for (int j = 0; j < numDims; ++j) {                                       
         std::cout<<h_data_ret(i, j)<<" ";                                       
       }                                                                         
       std::cout<<std::endl;                                                     
     }                                                                           
                                                                                 
  
     std::cout<<"\nDistances ret: "<<std::endl;                                  
     for (int i = 0; i < numObjs; ++i) {                                           
       for (int j = 0; j < numK; ++j) {                                      
         std::cout<<h_distances(i, j)<<" ";                                  
       }                                                                        
       std::cout<<std::endl;                                                    
     }                                                                          
 */                                                                                
//     verify<T>(labels_ref, h_labels, numObjs, 1, "Labels");                         
//     verify(h_data, h_data_ret, numDims, numObjs, "Data");                       
//     verify(h_clusters, h_clusters_ret, numDims, numK, "Clusters");              
                                                                                 
     std::cout<<"\nFreeing data"<<std::endl;                                     
     cudaFree(d_data); cudaFree(d_labels); cudaFree(d_clusters);
     cudaFree(d_distances); 
   }
 public:                                                                        
   Matrix<T>* Fit(const Matrix<T> &input_data, int numK) { 
     cuda_main(input_data, numK);  
     return centers_; 
   }                                  
   Vector<int>* FitPredict(const Matrix<T> &input_data, int numK) { 
     return labels_; 
   }                         
   Vector<int>* Predict(Matrix<T> new_points, int numK) {  
     return labels_; 
   }                                   
                                                                                
                                                                               
};                                                                              

}  // namespace Nice                                                            
#endif  // NEED_CUDA                                                            
#endif  // CPP_INCLUDE_GPU_KMEANS_H_  
