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
  std::string base_dir;
  int k_;
  int num_samples_;
  int num_features_;
  int num_clusters_;
  virtual void SetUp() {
    data_matrix_ = Nice::util::FromFile<T>(
        "../test/data_for_test/alternative_spectral_clustering/"
        "CenterData/data_matrix_ref_40_2.txt", 40, 2);
    k_ = 2;
    asc = std::make_shared<Nice::AlternativeSpectralClustering<T>>
        (data_matrix_, k_);
    num_samples_ = asc->num_samples_;
    num_features_ = asc->num_features_;
    num_clusters_ = asc->num_clusters_;
    base_dir = "../test/data_for_test/alternative_spectral_clustering";

  }
  Nice::Matrix<T> ReadTestData(std::string matrix_name, std::string func_name,
                   std::string test_data_type, int num_rows, int num_cols) {
    // test_data_type is either "ref" or "input"
    std::string dir = base_dir + "/" + func_name;
    std::string file_name = matrix_name + "_"
        + test_data_type + "_"
        + std::to_string(num_rows) + "_"
        + std::to_string(num_cols) + ".txt";
    std::string file_path = dir + "/" + file_name;
    return Nice::util::FromFile<T>(file_path, num_rows, num_cols);
    }

  Nice::Vector<T> ReadTestData(std::string matrix_name, std::string func_name,
                   std::string test_data_type, int num_elements) {
    // test_data_type is either "ref" or "input"
    std::string dir = base_dir + "/" + func_name;
    std::string file_name = matrix_name + "_"
        + test_data_type + "_"
        + std::to_string(num_elements) + "_"
        + std::to_string(1) + ".txt";
    std::string file_path = dir + "/" + file_name;
    return Nice::util::FromFile<T>(file_path, num_elements);
    }
//  Nice::Vector<int> ReadTestDataInt(std::string matrix_name,
//                                    std::string func_name,
//                                    std::string test_data_type,
//                                    int num_elements) {
//    // test_data_type is either "ref" or "input"
//    std::string dir = base_dir + "/" + func_name;
//    std::string file_name = matrix_name + "_"
//        + test_data_type + "_"
//        + std::to_string(num_elements) + "_"
//        + std::to_string(1) + ".txt";
//    std::string file_path = dir + "/" + file_name;
//    return Nice::util::FromFile<int>(file_path, num_elements);
//    }
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
        EXPECT_NEAR(double(a(i, j)), double(ref(i, j)), 0.0001);\


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
  Nice::Matrix<TypeParam> data_matrix_ref =
      this->ReadTestData("data_matrix", "CenterData", "ref",
                         this->num_samples_, this->num_features_);
  EXPECT_MATRIX_EQ(data_matrix, data_matrix_ref);
}

TYPED_TEST(AltSpectralClusteringTest, InitHMatrix) {
  this->asc->InitHMatrix();
  Nice::Matrix<TypeParam> h_matrix = this->asc->h_matrix_;
  Nice::Matrix<TypeParam> h_matrix_ref =
      this->ReadTestData("h_matrix", "InitHMatrix", "ref",
                         this->num_samples_, this->num_samples_);
  EXPECT_MATRIX_EQ(h_matrix, h_matrix_ref);
}

TYPED_TEST(AltSpectralClusteringTest, InitWMatrix) {
  // no input, directly calls the function
  this->asc->InitWMatrix();
  Nice::Matrix<TypeParam> w_matrix = this->asc->w_matrix_;
  Nice::Matrix<TypeParam> w_matrix_ref =
      this->ReadTestData("w_matrix", "InitWMatrix", "ref",
                         this->num_features_, this->num_features_);
  EXPECT_MATRIX_EQ(w_matrix, w_matrix_ref);
}

TYPED_TEST(AltSpectralClusteringTest, CalcGaussianKernel) {
  // input w_matrix
  this->asc->w_matrix_ =
      this->ReadTestData("w_matrix", "CalcGaussianKernel", "input",
                         this->num_features_, this->num_features_);
  // calls the function
  this->asc->CalcGaussianKernel();
  // output matrices
  Nice::Matrix<TypeParam> kernel_matrix = this->asc->kernel_matrix_;
  Nice::Matrix<TypeParam> kernel_matrix_ref =
      this->ReadTestData("kernel_matrix", "CalcGaussianKernel", "ref",
                         this->num_samples_, this->num_samples_);
  EXPECT_MATRIX_EQ(kernel_matrix, kernel_matrix_ref);
  Nice::Matrix<TypeParam> d_matrix = this->asc->d_matrix_;
  Nice::Matrix<TypeParam> d_matrix_ref =
      this->ReadTestData("d_matrix", "CalcGaussianKernel", "ref",
                         this->num_samples_, this->num_samples_);
  EXPECT_MATRIX_EQ(d_matrix, d_matrix_ref);
}

TYPED_TEST(AltSpectralClusteringTest, UOptimize) {
  // input h_matrix, d_matrix, kernel_matrix, w_matrix
  this->asc->h_matrix_ =
      this->ReadTestData("h_matrix", "UOptimize", "input",
                         this->num_samples_, this->num_samples_);
  this->asc->d_matrix_ =
      this->ReadTestData("d_matrix", "UOptimize", "input",
                         this->num_samples_, this->num_samples_);
  this->asc->kernel_matrix_ =
      this->ReadTestData("kernel_matrix", "UOptimize", "input",
                         this->num_samples_, this->num_samples_);
  this->asc->w_matrix_ =
      this->ReadTestData("w_matrix", "UOptimize", "input",
                         this->num_features_, this->num_features_);
  // call function
  this->asc->UOptimize();
  // output u_matrix
  Nice::Matrix<TypeParam> u_matrix = this->asc->u_matrix_;
  Nice::Matrix<TypeParam> u_matrix_ref =
      this->ReadTestData("u_matrix", "UOptimize", "ref",
                         this->num_samples_, this->num_features_);
  EXPECT_MATRIX_ABS_EQ(u_matrix, u_matrix_ref, 0.1);
}

TYPED_TEST(AltSpectralClusteringTest, CreateYTilde) {
  this->asc->h_matrix_ =
      this->ReadTestData("h_matrix", "CreateYTilde", "input",
                         this->num_samples_, this->num_samples_);
  this->asc->kernel_matrix_ =
      this->ReadTestData("kernel_matrix", "CreateYTilde", "input",
                         this->num_samples_, this->num_samples_);
  this->asc->d_matrix_ =
      this->ReadTestData("d_matrix", "CreateYTilde", "input",
                         this->num_samples_, this->num_samples_);
  Nice::Matrix<TypeParam> y_tilde = this->asc->CreateYTilde();
  Nice::Matrix<TypeParam> y_tilde_ref =
      this->ReadTestData("y_tilde", "CreateYTilde", "ref",
                         this->num_samples_, this->num_samples_);
  EXPECT_MATRIX_EQ(y_tilde, y_tilde_ref);
}

TYPED_TEST(AltSpectralClusteringTest, NormalizeEachURow) {
  this->asc->u_matrix_ =
      this->ReadTestData("u_matrix", "NormalizeEachURow", "input",
                         this->num_samples_, this->num_features_);
  this->asc->NormalizeEachURow();
  Nice::Matrix<TypeParam> normalized_u_matrix =
      this->asc->normalized_u_matrix_;
  Nice::Matrix<TypeParam> normalized_u_matrix_ref =
      this->ReadTestData("normalized_u_matrix", "NormalizeEachURow", "ref",
                         this->num_samples_, this->num_features_);
  EXPECT_MATRIX_EQ(normalized_u_matrix, normalized_u_matrix_ref);
}

TYPED_TEST(AltSpectralClusteringTest, RunKMeans) {
  this->asc->normalized_u_matrix_ =
      this->ReadTestData("normalized_u_matrix", "RunKMeans", "input",
                         this->num_samples_, this->num_clusters_);
  this->asc->RunKMeans();
  Nice::Vector<TypeParam> allocation =
      this->asc->allocation_;
  Nice::Vector<TypeParam> allocation_ref =
      this->ReadTestData("allocation", "RunKMeans", "ref", this->num_samples_);
  Nice::Matrix<TypeParam> y_matrix =
      this->asc->y_matrix_;
  Nice::Matrix<TypeParam> y_matrix_ref =
      this->ReadTestData("y_matrix", "RunKMeans", "ref",
                         y_matrix.rows(), y_matrix.cols());
  std::cout << "dlib allocation: " << std::endl;
  for (int i = 0; i < 20; i++)
    std::cout << allocation(i) << " ";
  std::cout << std::endl;
  std::cout << "numpy allocation: " << std::endl;
  for (int i = 0; i < 20; i++)
    std::cout << allocation_ref(i) << " ";
  std::cout << std::endl;
//  EXPECT_MATRIX_EQ(allocation, allocation_ref);
}
