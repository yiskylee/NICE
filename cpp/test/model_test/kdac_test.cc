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
#include "include/kdac_cpu.h"
#include "include/kdac_gpu.h"
#include "include/util.h"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/kernel_types.h"


template<typename T>
class KDACTest : public ::testing::Test {
 protected:
  std::shared_ptr<Nice::KDAC<T>> kdac_;
  std::string device_type_;
  std::string base_dir_;
  int num_clusters_;
  int num_samples_per_cluster_;
  int num_samples_;
  int dim_;
  std::string data_file_path_;
  std::string label_file_path_;
  std::string data_type_;
  std::string label_type_;
  Nice::Matrix<T> data_matrix_;
  Nice::Matrix<T> existing_y_;


  virtual void SetUp() {
  }

  void SetupInputData(int num_clusters, int num_samples_per_cluster,
                      int dim,
                      std::string device_type) {
    num_clusters_ = num_clusters;
    num_samples_per_cluster_ = num_samples_per_cluster;
    num_samples_ = num_clusters_ * num_samples_per_cluster_;
    dim_ = dim;
    device_type_ = device_type;

    if (device_type_ == "cpu")
      kdac_ = std::make_shared<Nice::KDACCPU<T>>();
    else if (device_type_ == "gpu")
      kdac_ = std::make_shared<Nice::KDACGPU<T>>();

    kdac_->SetQ(num_clusters_);
    kdac_->SetC(num_clusters_);
    kdac_->SetKernel(Nice::kGaussianKernel, 1.0);
    base_dir_ = "../test/data_for_test/kdac/";
    data_type_ = "data_gaussian";
    label_type_ = "y1_gaussian";
    GenFilePath();
    std::cout << "data_file_path: " << data_file_path_ << std::endl;
    data_matrix_ = Nice::util::FromFile<T>(data_file_path_, ",");
    std::cout << "label_file_path: " << label_file_path_ << std::endl;
    existing_y_ = Nice::util::FromFile<T>(label_file_path_, ",");
  }

  void Output() {
    Nice::KDACProfiler profiler = kdac_->GetProfiler();
    std::cout << "GenPhi: "
              << profiler.gen_phi.GetTotalTime() << std::endl;
    std::cout << "GenGradient: "
              << profiler.gen_grad.GetTotalTime() << std::endl;
    std::cout << "Update g(w): "
              << profiler.update_g_of_w.GetTotalTime() << std::endl;
    std::cout << "Fit: "
              << profiler.fit.GetTotalTime() << std::endl;
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

  Nice::Matrix<T> GenExistingY(void) {
    Nice::Matrix<T> existing_y =
        Nice::Matrix<T>::Zero(num_samples_, num_clusters_);
    for (int center = 0; center < num_clusters_; center++) {
      for (int sample = 0; sample < num_samples_per_cluster_; sample++) {
        existing_y(center * num_samples_per_cluster_ + sample, center) =
            static_cast<T>(1);
      }
    }
    return existing_y;
  }

  void GenFilePath(void) {
    std::string suffix = "_" + std::to_string(num_samples_) + "_"
        + std::to_string(dim_) + "_" + std::to_string(num_clusters_) + ".csv";
    data_file_path_ = base_dir_ + data_type_ + suffix;
    label_file_path_ = base_dir_ + label_type_ + suffix;
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




TYPED_TEST(KDACTest, CPU3_10_600_KDAC) {
  this->SetupInputData(3, 10, 600, "cpu");
  this->kdac_->SetVerbose(true);
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}

TYPED_TEST(KDACTest, CPU3_10_600_ISM) {
  this->SetupInputData(3, 10, 600, "cpu");
  this->kdac_->SetVerbose(true);
  this->kdac_->SetMethod("ISM");
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}

TYPED_TEST(KDACTest, CPU3_10_40_ISM) {
  this->SetupInputData(3, 10, 40, "cpu");
  this->kdac_->SetVerbose(true);
  this->kdac_->SetMethod("ISM");
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}

TYPED_TEST(KDACTest, CPU3_10_30_ISM) {
  this->SetupInputData(3, 10, 30, "cpu");
  this->kdac_->SetVerbose(true);
  this->kdac_->SetMethod("ISM");
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}

TYPED_TEST(KDACTest, CPU3_10_29_ISM) {
  this->SetupInputData(3, 10, 29, "cpu");
  this->kdac_->SetVerbose(true);
  this->kdac_->SetMethod("ISM");
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}


TYPED_TEST(KDACTest, CPU3_10_20_ISM) {
  this->SetupInputData(3, 10, 20, "cpu");
  this->kdac_->SetVerbose(true);
  this->kdac_->SetMethod("ISM");
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}

TYPED_TEST(KDACTest, CPU3_10_6_ISM) {
  this->SetupInputData(3, 10, 6, "cpu");
  this->kdac_->SetVerbose(true);
  this->kdac_->SetMethod("ISM");
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}

TYPED_TEST(KDACTest, CPU3_100_10_ISM) {
  this->SetupInputData(3, 100, 10, "cpu");
  this->kdac_->SetVerbose(true);
  this->kdac_->SetMethod("ISM");
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}

TYPED_TEST(KDACTest, CPU3_100_10) {
  this->SetupInputData(3, 100, 10, "cpu");
  this->kdac_->SetVerbose(true);
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}

TYPED_TEST(KDACTest, GPU3_10_600) {
  this->SetupInputData(3, 10, 600, "gpu");
  this->kdac_->SetVerbose(true);
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}

TYPED_TEST(KDACTest, CPU3_20_600) {
  this->SetupInputData(3, 20, 600, "cpu");
  this->kdac_->SetVerbose(true);
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}

TYPED_TEST(KDACTest, GPU3_20_600) {
  this->SetupInputData(3, 20, 600, "gpu");
  this->kdac_->SetVerbose(true);
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}

TYPED_TEST(KDACTest, CPU3_30_600) {
  this->SetupInputData(3, 30, 600, "cpu");
  this->kdac_->SetVerbose(true);
  this->kdac_->SetDebug(true);
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}

TYPED_TEST(KDACTest, GPU3_30_600) {
  this->SetupInputData(3, 30, 600, "gpu");
  this->kdac_->SetVerbose(true);
  this->kdac_->SetDebug(true);
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}