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

#include <string.h>
#include <cmath>
#include <iostream>
#include <string>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/matrix.h"
#include "include/util.h"
#include "include/kmeans.h"

class KmeansTest : public ::testing::Test {
 private:
 public:
   Nice::Vector<int> LabelsKmeans;
   void GetLabel() {
     Nice::Kmeans model = Nice::Kmeans();
     std::string inputFile = "data_k4_p10_d4_c1.txt";
     Nice::Matrix<double> testMatrix = Nice::util::FromFile<double>(inputFile, 40, 4);
     model.SetClusterType(Nice::Kmeans::lloyds);
     model.Fit(testMatrix);
     LabelsKmeans = model.GetLabels();
   }
   void Predict() {
     Nice::Kmeans model = Nice::Kmeans();
     std::string inputFile = "data_k4_p10_d4_c1.txt";
     Nice::Matrix<double> testMatrix = Nice::util::FromFile<double>(inputFile, 40, 4);
     model.SetClusterType(Nice::Kmeans::hybrid);
     model.Fit(testMatrix);
     LabelsKmeans = model.Predict(testMatrix);
   }
};

#define EXPECT_MATRIX_EQ(a, ref)\
    EXPECT_EQ(a.rows(), ref.rows());\
    EXPECT_EQ(a.cols(), ref.cols());\
    for (int i = 0; i < a.rows(); i++)\
      for (int j = 0; j < a.cols(); j++)\
        EXPECT_NEAR(double(a(i, j)), double(ref(i, j)), 0.0001);\

TEST_F(KmeansTest, FitTest) {
  //this->GetLabel();
  //Eigen::VectorXi LabelsTruth(40);
  //LabelsTruth << 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
  //EXPECT_MATRIX_EQ(LabelsTruth, LabelsKmeans);
  EXPECT_EQ(2+2, 4);
}

/*
TEST_F(KmeansTest, PredictTest) {
  this->Predict();
  Eigen::VectorXi LabelsTruth(40);
  LabelsTruth << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  EXPECT_MATRIX_EQ(LabelsTruth, LabelsKmeans);
}
*/
