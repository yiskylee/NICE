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

template<typename T>
class KDACTest : public ::testing::Test {
 protected:
  Nice::Matrix<T> data_matrix_;
  std::shared_ptr<Nice::KDAC<T>> kdac_;
  int c_;
  int n_;
  int d_;
  std::string base_dir_;

  virtual void SetUp() {
    data_matrix_ = Nice::util::FromFile<T>(
        "../test/data_for_test/kdac/data_4.csv", ",");
    c_ = 2;
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
typedef ::testing::Types<float, double> FloatTypes;


TYPED_TEST_CASE(KDACTest, FloatTypes);

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



TYPED_TEST(KDACTest, FitUMatrix) {
  this->kdac_->Fit(this->data_matrix_);
  Nice::Matrix<TypeParam> u_matrix = this->kdac_->GetU();
  Nice::Matrix<TypeParam> u_matrix_ref =
      this->ReadTestData("u_matrix", "Fit", "ref");
  EXPECT_MATRIX_ABS_EQ(u_matrix, u_matrix_ref, 0.01);

//  std::cout << u_matrix.block(0, 0, 5, 2) << std::endl << std::endl;
//  std::cout << u_matrix_ref.block(0, 0, 5, 2) << std::endl << std::endl;
}

TYPED_TEST(KDACTest, FitLMatrix) {
  this->kdac_->Fit(this->data_matrix_);
  Nice::Matrix<TypeParam> l_matrix = this->kdac_->GetL();
  Nice::Matrix<TypeParam> l_matrix_ref =
      this->ReadTestData("l_matrix", "Fit", "ref");
  EXPECT_MATRIX_EQ(l_matrix, l_matrix_ref);
}

TYPED_TEST(KDACTest, FitKMatrix) {
  this->kdac_->Fit(this->data_matrix_);
  Nice::Matrix<TypeParam> k_matrix = this->kdac_->GetK();
  Nice::Matrix<TypeParam> k_matrix_ref =
      this->ReadTestData("kernel_matrix", "Fit", "ref");
  EXPECT_MATRIX_EQ(k_matrix, k_matrix_ref);
//  std::cout << k_matrix.block(0, 0, 5, 2)
//      << std::endl << std::endl;
//  std::cout << k_matrix_ref.block(0, 0, 5, 2)
//      << std::endl << std::endl;
}

TYPED_TEST(KDACTest, FitDMatrix) {
  this->kdac_->Fit(this->data_matrix_);
  Nice::Matrix<TypeParam> d_matrix =
      this->kdac_->GetDToTheMinusHalf();
  Nice::Matrix<TypeParam> d_matrix_ref =
      this->ReadTestData("d_matrix", "Fit", "ref");
  EXPECT_MATRIX_EQ(d_matrix, d_matrix_ref);
}
