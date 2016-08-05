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
#include "include/kmcuda.h"
                                                                               
namespace Nice {                                                                
                                                                                
// Abstract class of common matrix operation interface                          
template <typename T>                                                           
class GpuKmeans {                                                               
 private:                                                                       
   int num_clusters_;                                                           
   int num_iterations_;                                                         
   int num_points_;                                                             
   Matrix<T> *data_points_;                                                     
   Vector<unsigned int> *labels_;                                                         
   Matrix<T> *centers_;                                                         
  
   void doKmeans(const Matrix<T> &input_data, int numK) { 
     Timer timer; 

     startTime(&timer); 
     std::cout<<"Allocating Memories"<<std::endl;
     Matrix<float> casted_samples = input_data.template cast<float>(); 
     Matrix<float> clusters(numK, input_data.cols()); 
     Matrix<T> casted_clusters(numK, input_data.cols());
     Vector<unsigned int> labels(input_data.rows());  
     bool kmpp = true;                                                           
     float tolerance = 0.001;                                                    
     float yinyang_t = 0.1;                                                      
     uint32_t samples_size = input_data.rows();                                            
     uint16_t features_size = input_data.cols();                                           
     uint32_t clusters_size = numK;                                              
     uint32_t seed = time(NULL);                                                 
     uint32_t device = 0;                                                        
     int32_t  verbosity = 1;                                                     
     float *samples = &casted_samples(0);                                                   
     float *centroids = &clusters(0);   
     uint32_t *assignments = &labels(0);      
     std::cout<<"Finished allocating memories"<<std::endl;                       
     std::cout<<"  Elapsed time to allocate memories ";                          
     std::cout<<elapsedTime(timer)<<std::endl;                                   
                                                                                 
                                                                                 
     startTime(&timer);                                                          
     std::cout<<"\nRunning Kmeans"<<std::endl;                                   
     kmeans_cuda(kmpp, tolerance, yinyang_t, samples_size,                       
                     features_size, clusters_size, seed,                         
                     device, verbosity, samples,                                 
                     centroids, assignments);                                    
     std::cout<<"Finished running Kmeans"<<std::endl;                            
     std::cout<<"  Elapsed time to run Kmeans: ";                                
     std::cout<<elapsedTime(timer)<<std::endl;
    
     num_points_ = input_data.rows(); 
     num_clusters_ = numK;
     casted_clusters = clusters.template cast<T>();  
     centers_ = &casted_clusters; 
     labels_ = &labels; 
   }  

 public:                                                                        
   Matrix<T>* Fit(const Matrix<T> &input_data, int numK) { 
     doKmeans(input_data, numK);  
     return centers_; 
   }                                  
   Vector<unsigned int>* FitPredict(const Matrix<T> &input_data, int numK) { 
     return labels_; 
   }                         
   Vector<unsigned int>* Predict(Matrix<T> new_points, int numK) {  
     return labels_; 
   }                                   
                                                                                
                                                                               
};                                                                              

}  // namespace Nice                                                            
#endif  // NEED_CUDA                                                            
#endif  // CPP_INCLUDE_GPU_KMEANS_H_  
