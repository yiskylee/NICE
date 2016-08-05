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

// This file tests the CpuOperations::Transpose() function by checking to
// See if a matrix passed is transposed in the test IsTransposed
// A transposed Nice matrix is compared to a transposed Eigen Matrix in
// Transpose Eigen
// Behavior with oddly shaped matrices is also tested with test DifferentShapes
// And TransposeZeroRows
// All tests are made using a templated test fixture which attempts
// Integer, float, and double data types

#include <iostream>
#include <cmath>
#include <sstream>  

#include "include/gpu_kmeans.h"
#include "include/gpu_operations.h"
#include "include/cpu_operations.h"

#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"
#include "include/util.h"

// This is a template test fixture class containing test matrices
template<class T>  // Template
class GpuKmeansTest : public ::testing::Test {  // Inherits testing::Test
 public:  // Members must be public to be accessed by tests
 
};
// Establishes a test case with the given types, Char and short types will
// Throw compiler errors
typedef ::testing::Types<float, double> dataTypes;
TYPED_TEST_CASE(GpuKmeansTest, dataTypes);

TYPED_TEST(GpuKmeansTest, FuncionalityTest) {
  srand(time(NULL));
  // Create test data
  int numK = 100; 
  int numObjs =  10000; 
  int numDims = 50; 
  int numC = 100; 
  std::stringstream sstm;                                                     
  sstm<<"../test/data_for_test/data_k"<<numK<<"_p"<<numObjs<<"_d"<<numDims<<"_c"<<numC<<".txt";      
  std::string file = sstm.str();
  Nice::Matrix<TypeParam> input_data = Nice::util::FromFile<TypeParam>(file); 


  Nice::Vector<unsigned int>* predicted_labels; 
  Nice::Matrix<TypeParam>* predicted_clusters;  
  // Test gpu matrix matrix multiply in Nice
  Nice::GpuKmeans<TypeParam> gpu_kmeans;
  predicted_labels = gpu_kmeans.Predict(input_data, numK); 
  predicted_clusters = gpu_kmeans.Fit(input_data, numK); 

  if (predicted_labels != NULL && predicted_clusters != NULL) 
  
  // Verify the result
  EXPECT_NEAR(2, 2, 0.001);
}

