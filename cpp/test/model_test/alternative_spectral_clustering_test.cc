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
#include <iostream>
#include <memory>
#include <cmath>
#include "include/util.h"
#include "gtest/gtest.h"
#include "include/alternative_spectral_clustering.h"


template <typename T>
class AltSpectralClusteringTest : public ::testing::Test {
 protected:
  Nice::Matrix<T> data_matrix_;
  Nice::Matrix<T> h_matrix_ref_;
  std::shared_ptr<Nice::AlternativeSpectralClustering<T>> asc;
  int k_;
  virtual void SetUp() {
    data_matrix_ = Nice::util::FromFile<T>(
        "../test/data_for_test/alternative_spectral_clustering/"
        "alt_spec_data_40_2.txt", 40, 2);
    k_ = 2;
    asc = std::make_shared<Nice::AlternativeSpectralClustering<T>>
        (data_matrix_, k_);
  }
};

typedef ::testing::Types<float, int, long, double> AllTypes;
typedef ::testing::Types<int, long> IntTypes;
typedef ::testing::Types<float, double> FloatTypes;
TYPED_TEST_CASE(AltSpectralClusteringTest, FloatTypes);

#define EXPECT_MATRIX_EQ(a, ref)\
    EXPECT_EQ(a.rows(), ref.rows());\
    EXPECT_EQ(a.cols(), ref.cols());\
    for (int i = 0; i < a.rows(); i++)\
      for (int j = 0; j < a.cols(); j++)\
        EXPECT_NEAR(a(i, j), ref(i, j), 0.0001);\

#define EXPECT_MATRIX_ABS_EQ(a, ref, error)\
    EXPECT_EQ(a.rows(), ref.rows());\
    EXPECT_EQ(a.cols(), ref.cols());\
    for (int i = 0; i < a.rows(); i++)\
      for (int j = 0; j < a.cols(); j++)\
        EXPECT_NEAR(std::abs(a(i, j)), std::abs(ref(i, j)), error);\


//TYPED_TEST(AltSpectralClusteringTest, SimpleTest) {
//  Nice::Vector<unsigned long> assignments = this->asc->FitPredict();
//}

TYPED_TEST(AltSpectralClusteringTest, CenterData) {
  Nice::Matrix<TypeParam> data_matrix = this->asc->data_matrix_;
  Nice::Matrix<TypeParam> data_matrix_ref = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "CenterData/data_ref_40_2.txt", 40, 2);
  EXPECT_MATRIX_EQ(data_matrix, data_matrix_ref);
}

TYPED_TEST(AltSpectralClusteringTest, InitHMatrix) {
  this->asc->InitHMatrix();
  Nice::Matrix<TypeParam> h_matrix = this->asc->h_matrix_;
  Nice::Matrix<TypeParam> h_matrix_ref = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "InitHMatrix/h_matrix_ref_40_40.txt", 40, 40);
  EXPECT_MATRIX_EQ(h_matrix, h_matrix_ref);
}

TYPED_TEST(AltSpectralClusteringTest, InitWMatrix) {
  // no input, directly calls the function
  this->asc->InitWMatrix();
  Nice::Matrix<TypeParam> w_matrix = this->asc->w_matrix_;
  Nice::Matrix<TypeParam> w_matrix_ref = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "InitWMatrix/w_matrix_ref_2_2.txt", 2, 2);
  EXPECT_MATRIX_EQ(w_matrix, w_matrix_ref);
}

TYPED_TEST(AltSpectralClusteringTest, CalcGaussianKernel) {
  // input w_matrix
  this->asc->w_matrix_ = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "CalcGaussianKernel/w_matrix_input_2_2.txt", 2, 2);
  // calls the function
  this->asc->CalcGaussianKernel();
  // output matrices
  Nice::Matrix<TypeParam> kernel_matrix = this->asc->kernel_matrix_;
  Nice::Matrix<TypeParam> kernel_matrix_ref = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "CalcGaussianKernel/kernel_matrix_ref_40_40.txt", 40, 40);
  EXPECT_MATRIX_EQ(kernel_matrix, kernel_matrix_ref);
  Nice::Matrix<TypeParam> d_matrix = this->asc->d_matrix_;
  Nice::Matrix<TypeParam> d_matrix_ref = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "CalcGaussianKernel/d_matrix_ref_40_40.txt", 40, 40);
  EXPECT_MATRIX_EQ(d_matrix, d_matrix_ref);
}

TYPED_TEST(AltSpectralClusteringTest, UOptimize) {
  this->asc->h_matrix_ = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "/UOptimize/h_matrix_input_40_40.txt", 40, 40);
  this->asc->d_matrix_ = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "/UOptimize/d_matrix_input_40_40.txt", 40, 40);
  this->asc->kernel_matrix_ = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "/UOptimize/kernel_matrix_input_40_40.txt", 40, 40);
  this->asc->w_matrix_ = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "/UOptimize/w_matrix_input_2_2.txt", 2, 2);

  this->asc->UOptimize();
//  Nice::Matrix<TypeParam> l_matrix = this->asc->l_matrix_;
//  Nice::Matrix<TypeParam> l_matrix_ref = Nice::util::FromFile<TypeParam>(
//      "../test/data_for_test/alternative_spectral_clustering/"
//      "/UOptimize/l_matrix_ref_40_40.txt", 40, 40);
//  EXPECT_MATRIX_EQ(l_matrix, l_matrix_ref);
  Nice::Matrix<TypeParam> u_matrix = this->asc->u_matrix_;
  Nice::Matrix<TypeParam> u_matrix_ref = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "/UOptimize/u_matrix_ref_40_2.txt", 40, 2);
  EXPECT_MATRIX_ABS_EQ(u_matrix, u_matrix_ref, 0.1);
}

TYPED_TEST(AltSpectralClusteringTest, CreateYTilde) {
  this->asc->h_matrix_ = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "CreateYTilde/h_matrix_input_40_40.txt", 40, 40);
  this->asc->kernel_matrix_ = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "CreateYTilde/kernel_matrix_input_40_40.txt", 40, 40);
  this->asc->d_matrix_ = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "CreateYTilde/d_matrix_input_40_40.txt", 40, 40);

  Nice::Matrix<TypeParam> y_tilde = this->asc->CreateYTilde();
  Nice::Matrix<TypeParam> y_tilde_ref = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/alternative_spectral_clustering/"
      "CreateYTilde/y_tilde_ref_40_40.txt", 40, 40);
  EXPECT_MATRIX_EQ(y_tilde, y_tilde_ref);
}
