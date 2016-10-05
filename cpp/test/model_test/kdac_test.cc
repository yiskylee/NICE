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


#include <stdio.h>
#include <iostream>
#include <memory>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/kdac.h"
#include "include/util.h"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/kernel_types.h"

template<typename T>
class KDACTest : public ::testing::Test {
 protected:
//  Nice::Matrix<T> data_matrix_;
  std::shared_ptr<Nice::KDAC<T>> kdac_;
  int c_;
  std::string base_dir_;

  virtual void SetUp() {
//    data_matrix_ = Nice::util::FromFile<T>(
//        "../test/data_for_test/kdac/data_400.csv", ",");
//    c_ = 2;
    kdac_ = std::make_shared<Nice::KDAC<T>>();
    kdac_->SetC(c_);
    base_dir_ = "../test/data_for_test/kdac";
  }

  Nice::Matrix<T> ReadTestData(
      std::string matrix_name,
      std::string func_name,
      std::string test_data_type) {
    // test_data_type is either "ref" or "input"
    std::string dir = base_dir_ + "/" + func_name;
    std::string file_name = matrix_name + "_"
        + test_data_type + ".txt";
    std::string file_path = dir + "/" + file_name;
    return Nice::util::FromFile<T>(file_path);
  }

};

typedef ::testing::Types<float, int, long, double> AllTypes;
typedef ::testing::Types<int, long> IntTypes;
typedef ::testing::Types<float> FloatTypes;
typedef ::testing::Types<float, double> BothTypes;


TYPED_TEST_CASE(KDACTest, FloatTypes);

#define EXPECT_MATRIX_EQ(a, ref)\
    EXPECT_EQ(a.rows(), ref.rows());\
    EXPECT_EQ(a.cols(), ref.cols());\
    for (int i = 0; i < a.rows(); i++)\
      for (int j = 0; j < a.cols(); j++)\
        EXPECT_NEAR(double(a(i, j)), double(ref(i, j)), 0.0001);\

#define PRINTV(v, num_per_line)\
  for (int i = 0; i < v.rows(); i++) {\
    if (i % num_per_line == 0 && i != 0)\
      std::cout << std::endl;\
    std::cout << v(i) << ",";\
  }\
  std::cout << std::endl;\


#define EXPECT_MATRIX_ABS_EQ(a, ref, error)\
    EXPECT_EQ(a.rows(), ref.rows());\
    EXPECT_EQ(a.cols(), ref.cols());\
    for (int i = 0; i < a.rows(); i++)\
      for (int j = 0; j < a.cols(); j++)\
        EXPECT_NEAR(std::abs(a(i, j)), std::abs(ref(i, j)), error);\

TYPED_TEST(KDACTest, PredGaussian) {
  int num_clusters = 3;
  int num_samples_per_cluster = 100;
  int num_samples = num_clusters * num_samples_per_cluster;
  int dim = 6;
  this->kdac_->SetQ(num_clusters);
  this->kdac_->SetC(num_clusters);
  std::string root_dir("../test/data_for_test/kdac/");
  std::string file_name = "data_gaussian_" + std::to_string(num_samples) + "_"
  + std::to_string(dim) + "_" + std::to_string(num_clusters) + ".csv";
//  Nice::Matrix<TypeParam> data_matrix = Nice::util::FromFile<TypeParam>(
//      "../test/data_for_test/kdac/data_gaussian_150_6_3.csv", ",");
  Nice::Matrix<TypeParam> data_matrix = Nice::util::FromFile<TypeParam>(
      root_dir + file_name, ",");
  Nice::Matrix<TypeParam> first_y =
      Nice::Matrix<TypeParam>::Zero(data_matrix.rows(), this->kdac_->GetC());

  for (int center = 0; center < num_clusters; center++) {
    for (int sample = 0; sample < num_samples_per_cluster; sample ++) {
      first_y(center * num_samples_per_cluster + sample, center) =
          static_cast<TypeParam>(1);
    }
  }
//  for (int i = 0; i < 40; i++)
//    first_y(i, 0) = static_cast<TypeParam>(1);
//  for (int i = 40; i < 80; i++)
//    first_y(i, 1) = static_cast<TypeParam>(1);
//  for (int i = 80; i < 120; i++)
//    first_y(i, 2) = static_cast<TypeParam>(1);
//  this->kdac_->Print(first_y, "y_after");
//  for (float sigma = 1; sigma < 20; sigma++) {
  this->kdac_->SetKernel(Nice::kGaussianKernel, 1.0);
  this->kdac_->Fit(data_matrix, first_y);
  PRINTV(this->kdac_->Predict(), num_samples_per_cluster);
//  }

//
//  this->kdac_->Fit(data_matrix);
//  PRINTV(this->kdac_->Predict(), 40);

//  Nice::Matrix<TypeParam>
//
//  Nice::Matrix<TypeParam> first_y = this->kdac_->GetY();
//  this->kdac_->SetLambda(1.0);
//  this->kdac_->Fit(data_matrix, first_y);
//  PRINTV(this->kdac_->Predict(), 40);
//
//  this->kdac_->SetLambda(2.0);
//  this->kdac_->Fit(data_matrix, first_y);
//  PRINTV(this->kdac_->Predict());
//
//  this->kdac_->SetLambda(3.0);
//  this->kdac_->Fit(data_matrix, first_y);
//  PRINTV(this->kdac_->Predict());
}

TYPED_TEST(KDACTest, WlRef) {
  Nice::Matrix<TypeParam> w_matrix(3,3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      w_matrix(i, j) = i * 3 + j;
  std::cout << w_matrix << std::endl;
  Nice::Vector<TypeParam> w_l = w_matrix.col(0);
  w_l(0) = 88;
  std::cout << w_l << std::endl;
  std::cout << w_matrix << std::endl;
}

//TYPED_TEST(KDACTest, FitUMatrix) {
//  this->kdac_->Fit(this->data_matrix_);
//  Nice::Matrix<TypeParam> u_matrix = this->kdac_->GetU();
//  Nice::Matrix<TypeParam> u_matrix_ref =
//      this->ReadTestData("u_matrix", "Fit", "ref");
//  EXPECT_MATRIX_ABS_EQ(u_matrix, u_matrix_ref, 0.01);
//
////  std::cout << u_matrix.block(0, 0, 5, 2) << std::endl << std::endl;
////  std::cout << u_matrix_ref.block(0, 0, 5, 2) << std::endl << std::endl;
//}
//
//TYPED_TEST(KDACTest, FitLMatrix) {
//  this->kdac_->Fit(this->data_matrix_);
//  Nice::Matrix<TypeParam> l_matrix = this->kdac_->GetL();
//  Nice::Matrix<TypeParam> l_matrix_ref =
//      this->ReadTestData("l_matrix", "Fit", "ref");
//  EXPECT_MATRIX_EQ(l_matrix, l_matrix_ref);
//}
//
//TYPED_TEST(KDACTest, FitKMatrix) {
//  this->kdac_->Fit(this->data_matrix_);
//  Nice::Matrix<TypeParam> k_matrix = this->kdac_->GetK();
//  Nice::Matrix<TypeParam> k_matrix_ref =
//      this->ReadTestData("kernel_matrix", "Fit", "ref");
//  EXPECT_MATRIX_EQ(k_matrix, k_matrix_ref);
////  std::cout << k_matrix.block(0, 0, 5, 2)
////      << std::endl << std::endl;
////  std::cout << k_matrix_ref.block(0, 0, 5, 2)
////      << std::endl << std::endl;
//}
//
//TYPED_TEST(KDACTest, FitDMatrix) {
//  this->kdac_->Fit(this->data_matrix_);
//  Nice::Matrix<TypeParam> d_matrix =
//      this->kdac_->GetDToTheMinusHalf();
//  Nice::Matrix<TypeParam> d_matrix_ref =
//      this->ReadTestData("d_matrix", "Fit", "ref");
//  EXPECT_MATRIX_EQ(d_matrix, d_matrix_ref);
//}
//
//TYPED_TEST(KDACTest, FitAMatrixList) {
//  this->kdac_->Fit(this->data_matrix_);
//  int n = this->data_matrix_.rows();
//  std::vector<Nice::Matrix<TypeParam>> a_matrix_list = this->kdac_->GetAList();
//  Nice::Matrix<TypeParam> a_matrix = a_matrix_list[2 * n + 3];
//  Nice::Matrix<TypeParam> a_matrix_ref =
//      this->ReadTestData("a_matrix", "Fit", "ref");
//  EXPECT_MATRIX_EQ(a_matrix, a_matrix_ref);
////  std::cout << a_matrix_list[10] << std::endl << std::endl;
////  for (int i = 0; i < n; i++)
////    for (int j = 0; j < n; j++)
////      std::cout << a_matrix_list[i * n + j] << std::endl << std::endl;
//}

//TYPED_TEST(KDACTest, FitWMatrix) {
//  this->kdac_->Fit(this->data_matrix_);
//  this->kdac_->Fit();
//  Nice::Matrix<TypeParam> w_matrix =
//      this->kdac_->GetW();
//  Nice::Matrix<TypeParam> w_matrix_ref =
//      this->ReadTestData("w_matrix", "Fit", "ref");
//  EXPECT_MATRIX_EQ(w_matrix, w_matrix_ref);
//}

//TYPED_TEST(KDACTest, FitYMatrixTilde) {
//  this->kdac_->Fit(this->data_matrix_);
//  Nice::Matrix<TypeParam> y_matrix_tilde = this->kdac_->GetYTilde();
//  Nice::Matrix<TypeParam> y_matrix_tilde_ref =
//      this->ReadTestData("y_matrix_tilde", "Fit", "ref");
//  EXPECT_MATRIX_EQ(y_matrix_tilde, y_matrix_tilde_ref);
//}
//
//TYPED_TEST(KDACTest, FitGammaMatrix) {
//  this->kdac_->Fit(this->data_matrix_);
//  Nice::Matrix<TypeParam> gamma_matrix = this->kdac_->GetGamma();
//  Nice::Matrix<TypeParam> gamma_matrix_ref =
//      this->ReadTestData("gamma_matrix", "Fit", "ref");
////  std::cout << gamma_matrix << std::endl;
//  EXPECT_MATRIX_EQ(gamma_matrix.col(0), gamma_matrix_ref.col(0));
//}

//TEST(KDACTest, ReferTest) {
//  Nice::Matrix<float> a(2,2);
//  a << 1,2,
//       3,4;
//  std::cout << a << std::endl;
//  Nice::Matrix<float> &a_ref = a;
//  a_ref(0,0) = 88;
//  std::cout << a << std::endl;
//  Nice::Matrix<float> &b = a_ref;
//  b(0,0) = 99;
//  std::cout << a << std::endl;
//  std::cout << b << std::endl;
//}



//TYPED_TEST(KDACTest, Ortho) {
//  Nice::Matrix<TypeParam> m(3, 2);
//  m << 1.0,0.0,
//       0.0,-1.0,
//       0.0,0.0;
//  Nice::Vector<TypeParam> c(3);
//  c << 3,2,3;
//  Nice::Vector<TypeParam> vertical = this->kdac_->GenOrthogonal(m, c);
//  std::cout << vertical << std::endl;
//}
