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
        "../test/data_for_test/alt_spec_data_40_2.txt", 40, 2);
    k_ = 2;
    asc = std::make_shared<Nice::AlternativeSpectralClustering<T>>
        (data_matrix_, k_);
  }
};

typedef ::testing::Types<float, int, long, double> AllTypes;
typedef ::testing::Types<int, long> IntTypes;
typedef ::testing::Types<float, double> FloatTypes;
TYPED_TEST_CASE(AltSpectralClusteringTest, FloatTypes);

TYPED_TEST(AltSpectralClusteringTest, SimpleTest) {
  Nice::Vector<unsigned long> assignments = this->asc->FitPredict();
}

TYPED_TEST(AltSpectralClusteringTest, InitHMatrix) {
  this->asc->initialize_h_matrix();
  Nice::Matrix<TypeParam> h_matrix = this->asc->h_matrix_;
  Nice::Matrix<TypeParam> h_matrix_ref = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/h_matrix_ref_40_40.txt", 40, 40);
  EXPECT_EQ(h_matrix.rows(), h_matrix_ref.rows());
  EXPECT_EQ(h_matrix.cols(), h_matrix_ref.cols());
  for (int i = 0; i < h_matrix.rows(); i++)
    for (int j = 0; j < h_matrix.cols(); j++)
      EXPECT_EQ(h_matrix(i, j), h_matrix_ref(i, j));
}

TYPED_TEST(AltSpectralClusteringTest, InitWMatrix) {
  this->asc->initialize_w_matrix();
  Nice::Matrix<TypeParam> w_matrix = this->asc->w_matrix_;
  Nice::Matrix<TypeParam> w_matrix_ref = Nice::util::FromFile<TypeParam>(
      "../test/data_for_test/w_matrix_ref_2_2.txt", 2, 2);
  EXPECT_EQ(w_matrix.rows(), w_matrix_ref.rows());
  EXPECT_EQ(w_matrix.cols(), w_matrix_ref.cols());
  for (int i = 0; i < w_matrix.rows(); i++)
    for (int j = 0; j < w_matrix.cols(); j++)
      EXPECT_EQ(w_matrix(i, j), w_matrix_ref(i, j));
}





