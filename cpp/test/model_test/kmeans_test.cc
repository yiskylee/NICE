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
#include "include/kmeans.h"
#include "gtest/gtest.h"
#include "include/util.h"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/kernel_types.h"
#include "include/stop_watch.h"


template<typename T>
class KMeansTest : public ::testing::Test {
 protected:
//  Nice::Matrix<T> data_matrix_;
  std::shared_ptr<Nice::KMeans<T>> kmeans_;
  int k_;
  std::string device_type_;
  std::string data_file_path_;
  Nice::Matrix<T> data_;
  Nice::Vector<T> labels_;


  virtual void SetUp() {
  }

  void SetupInputData(int k, std::string base_dir,
                      std::string file_name,
                      std::string device_type) {
    k_ = k;
    device_type_ = device_type;

    if (device_type_ == "cpu")
      kmeans_ = std::make_shared<Nice::KMeans<T>>();
    else if (device_type_ == "gpu")
      kmeans_ = nullptr;

    data_file_path_ = base_dir + file_name;
    std::cout << "data_file_path: " << data_file_path_ << std::endl;
    data_ = Nice::util::FromFile<T>(data_file_path_, ",");
  }
};

typedef ::testing::Types<float> FloatTypes;

TYPED_TEST_CASE(KMeansTest, FloatTypes);


TYPED_TEST(KMeansTest, CPU5_10_3) {
  std::string base_dir = "../test/data_for_test/";
  // std::string file_name = "clustering_k5_10_d3.txt";
  // std::string file_name = "data_k50_p10000_d100_c1.txt";
  // std::string file_name = "data_k5_p500_d10_c1.txt";
  std::string file_name = "data_k5_p10_d3_c1.txt";
  this->SetupInputData(5, base_dir, file_name, "cpu");
  this->kmeans_->Fit(this->data_, this->k_);
  this->labels_ = this->kmeans_->GetLabels();
  // std::cout << this->labels_ << std::endl;
}


