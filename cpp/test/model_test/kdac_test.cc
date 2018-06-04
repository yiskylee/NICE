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
  std::shared_ptr<Nice::KDAC<T>> kdac_cpu_;
  std::shared_ptr<Nice::KDAC<T>> kdac_gpu_;
  std::string device_type_;
  std::string base_dir_;
  int n_;
  int d_;
  int k_;
  std::string data_file_path_;
  std::string label_file_path_;
  std::string data_type_;
  std::string label_type_;
  Nice::Matrix<T> data_matrix_;
  Nice::Matrix<T> existing_y_;

  virtual void SetUp() {
  }

  void SetupInputData(int n, int d, int k, std::string device_type,
                      bool label=true) {
    n_ = n;
    d_ = d;
    k_ = k;
    device_type_ = device_type;

    if (device_type_ == "cpu") {
      kdac_ = std::make_shared<Nice::KDACCPU<T>>();
      kdac_->SetQ(k_);
      kdac_->SetC(k_);
      kdac_->SetKernel(Nice::kGaussianKernel, 1.0);
    }
#ifdef CUDA_AND_GPU
    else if (device_type_ == "gpu") {
      kdac_ = std::make_shared<Nice::KDACGPU<T>>();
      kdac_->SetQ(k_);
      kdac_->SetC(k_);
      kdac_->SetKernel(Nice::kGaussianKernel, 1.0);
    } else if (device_type == "both") {
      kdac_cpu_ = std::make_shared<Nice::KDACCPU<T>>();
      kdac_gpu_ = std::make_shared<Nice::KDACGPU<T>>();
      kdac_cpu_->SetQ(k_);
      kdac_cpu_->SetC(k_);
      kdac_cpu_->SetKernel(Nice::kGaussianKernel, 1.0);
      kdac_gpu_->SetQ(k_);
      kdac_gpu_->SetC(k_);
      kdac_gpu_->SetKernel(Nice::kGaussianKernel, 1.0);
    }
#endif

    base_dir_ = "../test/data_for_test/kdac/";
    data_type_ = "data_gaussian";
    label_type_ = "y1_gaussian";
    GenFilePath();
    std::cout << "data_file_path: " << data_file_path_ << std::endl;
    data_matrix_ = Nice::util::FromFile<T>(data_file_path_, ",");
    if (label) {
      std::cout << "label_file_path: " << label_file_path_ << std::endl;
      existing_y_ = Nice::util::FromFile<T>(label_file_path_, ",");
    }
  }

  void Output() {
    if (device_type_ != "both") {
      Nice::ACLProfiler profiler = kdac_->GetProfiler();
      std::cout << "GenPhi(alpha): "
                << profiler["gen_phi(alpha)"].GetTotalTime() << std::endl;
      std::cout << "GenGradient: "
                << profiler["gen_grad"].GetTotalTime() << std::endl;
      std::cout << "Fit: "
                << profiler["fit"].GetTotalTime() << std::endl;
    } else {
      Nice::ACLProfiler profiler_cpu = kdac_cpu_->GetProfiler();
      std::cout << "\n CPU: \n";
      std::cout << "GenPhi(alpha): "
                << profiler_cpu["gen_phi(alpha)"].GetTotalTime() << std::endl;
      std::cout << "GenGradient: "
                << profiler_cpu["gen_grad"].GetTotalTime() << std::endl;
      std::cout << "LineSearch: "
                << profiler_cpu["line_search"].GetTotalTime() << std::endl;
      std::cout << "Fit: "
                << profiler_cpu["fit"].GetTotalTime() << std::endl;


      Nice::ACLProfiler profiler_gpu = kdac_gpu_->GetProfiler();
      std::cout << "\n GPU: \n";
      std::cout << "GenPhi(alpha): "
                << profiler_gpu["gen_phi(alpha)"].GetTotalTime() << std::endl;
      std::cout << "GenGradient: "
                << profiler_gpu["gen_grad"].GetTotalTime() << std::endl;
      std::cout << "LineSearch: "
                << profiler_gpu["line_search"].GetTotalTime() << std::endl;
      std::cout << "Fit: "
                << profiler_gpu["fit"].GetTotalTime() << std::endl;
    }
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
        Nice::Matrix<T>::Zero(n_, k_);
    for (int center = 0; center < k_; center++) {
      for (int sample = 0; sample < n_ / k_; sample++) {
        existing_y(center * n_ / k_ + sample, center) =
            static_cast<T>(1);
      }
    }
    return existing_y;
  }

  void GenFilePath(void) {
    std::string suffix = "_" + std::to_string(n_) + "_"
        + std::to_string(d_) + "_" +
        std::to_string(k_) + ".csv";
    data_file_path_ = base_dir_ + data_type_ + suffix;
    label_file_path_ = base_dir_ + label_type_ + suffix;
  }

};

typedef ::testing::Types<float, int, long, double> AllTypes;
typedef ::testing::Types<int, long> IntTypes;
typedef ::testing::Types<float> FloatTypes;
typedef ::testing::Types<float, double> BothTypes;

TYPED_TEST_CASE(KDACTest, BothTypes);

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


TYPED_TEST(KDACTest, CPU_30_100_3) {
  this->SetupInputData(30, 100, 3, "cpu");
  this->kdac_->SetQ(3);
  this->kdac_->SetKernel(Nice::kGaussianKernel, 1.0);
  this->kdac_->SetVerbose(true);
  this->kdac_->SetMode("gtest");
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
  Nice::util::Print(this->kdac_->Predict(), "Alternative Solution");
}

TYPED_TEST(KDACTest, GPU_30_100_3) {

  this->SetupInputData(30, 100, 3, "gpu");
  this->kdac_->SetQ(3);
  this->kdac_->SetKernel(Nice::kGaussianKernel, 1.0);
  this->kdac_->SetVerbose(true);
  this->kdac_->SetMode("gtest");
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
  Nice::util::Print(this->kdac_->Predict(), "Alternative Solution");
}

TYPED_TEST(KDACTest, BOTH_30_100_3) {
  this->SetupInputData(30, 100, 3, "both");
  this->kdac_cpu_->SetVerbose(true);
  this->kdac_gpu_->SetVerbose(true);
  std::cout << "\nCPU:\n";
  this->kdac_cpu_->Fit(this->data_matrix_, this->existing_y_);
  std::cout << "\nGPU:\n";
  this->kdac_gpu_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}

TYPED_TEST(KDACTest, BOTH_300_100_3) {
  this->SetupInputData(300, 100, 3, "both");
  this->kdac_cpu_->SetVerbose(true);
  this->kdac_gpu_->SetVerbose(true);
  std::cout << "\nCPU:\n";
  this->kdac_cpu_->Fit(this->data_matrix_, this->existing_y_);
  std::cout << "\nGPU:\n";
  this->kdac_gpu_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
}

TYPED_TEST(KDACTest, CPU_120_100_3) {
  this->SetupInputData(120, 100, 3, "cpu");
  this->kdac_->SetQ(3);
  this->kdac_->SetKernel(Nice::kGaussianKernel, 1.0);
  this->kdac_->SetVerbose(true);
  this->kdac_->SetMode("gtest");
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
  Nice::util::Print(this->kdac_->Predict(), "Alternative Solution");
}

TYPED_TEST(KDACTest, CPU_30_6_3) {
  this->SetupInputData(30, 6, 3, "cpu");
  this->kdac_->SetQ(3);
  this->kdac_->SetKernel(Nice::kGaussianKernel, 1.0);
  this->kdac_->SetVerbose(true);
  this->kdac_->SetMode("gtest");
  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
  this->Output();
  Nice::util::Print(this->kdac_->Predict(), "Alternative Solution");
}

TYPED_TEST(KDACTest, CPU_40_2_2) {
  this->SetupInputData(40, 2, 2, "cpu");
  this->kdac_->SetQ(1);
  this->kdac_->SetKernel(Nice::kGaussianKernel, 0.5);
  this->kdac_->SetVerbose(true);
  this->kdac_->SetMode("gtest");
  this->kdac_->SetDebug(true);
  this->kdac_->Fit(this->data_matrix_);
  Nice::util::Print(this->kdac_->Predict(), "Original Solution");
  this->kdac_->Fit();
  Nice::util::Print(this->kdac_->Predict(), "Alternative Solution");
}

TYPED_TEST(KDACTest, CPU_400_4_2) {
  this->SetupInputData(400, 4, 2, "cpu", false);
  this->kdac_->SetQ(1);
  this->kdac_->SetKernel(Nice::kGaussianKernel, 2.0);
  this->kdac_->SetVerbose(true);
  this->kdac_->SetMode("gtest");
  this->kdac_->SetDebug(false);
  this->kdac_->SetThreshold2(0.01);
  this->kdac_->Fit(this->data_matrix_);
  Nice::util::Print(this->kdac_->Predict(), "Original Solution");
  this->kdac_->SetKernel(Nice::kGaussianKernel, 1.0);
  this->kdac_->SetMaxTime(30);
  this->kdac_->Fit();
  Nice::util::Print(this->kdac_->Predict(), "Alternative Solution");
}

//TYPED_TEST(KDACTest, CPU400_4_2_ISM) {
//  this->SetupInputData(400, 4, 2, "cpu", false);
//  this->kdac_->SetQ(1);
//  this->kdac_->SetKernel(Nice::kGaussianKernel, 2.0);
//  this->kdac_->SetVerbose(true);
//  this->kdac_->SetMethod("ISM");
//  this->kdac_->SetMode("gtest");
////  this->kdac_->SetDebug(true);
//  this->kdac_->Fit(this->data_matrix_);
//  this->kdac_->SetKernel(Nice::kGaussianKernel, 1.0);
//  this->kdac_->Fit();
//}

//TYPED_TEST(KDACTest, CPU40_2_2_ISM) {
//  this->SetupInputData(40, 2, 2, "cpu", false);
//  this->kdac_->SetQ(1);
//  this->kdac_->SetKernel(Nice::kGaussianKernel, 0.5);
//  this->kdac_->SetVerbose(true);
//  this->kdac_->SetMethod("ISM");
//  this->kdac_->SetMode("gtest");
//  this->kdac_->SetDebug(true);
//  this->kdac_->Fit(this->data_matrix_);
//  this->kdac_->Fit();
//}

//TYPED_TEST(KDACTest, CPU400_4_2_ISM_NON_VEC) {
//  this->SetupInputData(400, 4, 2, "cpu", false);
//  this->kdac_->SetQ(1);
//  this->kdac_->SetKernel(Nice::kGaussianKernel, 2.0);
//  this->kdac_->SetVectorization(false);
//  this->kdac_->SetVerbose(true);
//  this->kdac_->SetMethod("ISM");
//  this->kdac_->SetMode("gtest");
////  this->kdac_->SetDebug(true);
//  this->kdac_->Fit(this->data_matrix_);
//  this->kdac_->SetKernel(Nice::kGaussianKernel, 1.0);
//  this->kdac_->Fit();
//}
//
//TYPED_TEST(KDACTest, CPU300_100_3_ISM) {
//  this->SetupInputData(300, 100, 3, "cpu");
//  this->kdac_->SetQ(3);
//  this->kdac_->SetMaxTime(300);
//  this->kdac_->SetKernel(Nice::kGaussianKernel, 1.0);
//  this->kdac_->SetMethod("ISM");
//  this->kdac_->SetMode("gtest");
//  this->kdac_->SetVerbose(true);
//  this->kdac_->SetDebug(false);
//  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
//}
//
//TYPED_TEST(KDACTest, CPU270_100_3_ISM) {
//  this->SetupInputData(270, 100, 3, "cpu");
//  this->kdac_->SetQ(3);
//  this->kdac_->SetMaxTime(300);
//  this->kdac_->SetKernel(Nice::kGaussianKernel, 1.0);
//  this->kdac_->SetMethod("ISM");
//  this->kdac_->SetMode("gtest");
//  this->kdac_->SetVerbose(true);
//  this->kdac_->SetDebug(false);
//  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
//}

//TYPED_TEST(KDACTest, CPU120_100_3_ISM) {
//  this->SetupInputData(120, 100, 3, "cpu");
//  this->kdac_->SetVerbose(true);
//  this->kdac_->SetMethod("ISM");
//  this->kdac_->Fit(this->data_matrix_, this->existing_y_);
//  this->Output();
//}
