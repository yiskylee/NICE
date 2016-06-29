#include "include/gpu_svd_solver.h"
#include <iostream>
#include <stdio.h>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/matrix.h"
#include "include/vector.h"


TEST(SvdTest, floatTest) {
  // GPU SVD solver
  Nice::GpuSvdSolver<float> gpu_svd; 

  // CPU SVD solver from eigen
  Eigen::JacobiSVD<Eigen::MatrixXf> cpu_svd;

  // Define the test matrix
  int row = 10;
  int col = row;
  Nice::Matrix<float> m = Eigen::MatrixXf::Random(row,col); 

  // Solve in GPU
  gpu_svd.Compute(m);

  // Solve in CPU
  cpu_svd.compute(m, Eigen::ComputeFullU|Eigen::ComputeFullV);

  // Get GPU SVD results
  Nice::Vector<float> gpu_s = gpu_svd.SingularValues(); 
  Nice::Matrix<float> gpu_u = gpu_svd.MatrixU();
  Nice::Matrix<float> gpu_v = gpu_svd.MatrixV();  

  // Get CPU SVD results
  Nice::Vector<float> cpu_s = cpu_svd.singularValues();
  Nice::Matrix<float> cpu_u = cpu_svd.matrixU();
  Nice::Matrix<float> cpu_v = cpu_svd.matrixV();  


  // Verify matrix U
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      EXPECT_EQ(gpu_u(i, j), cpu_u(i, j));

  // Verify matrix V
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      EXPECT_EQ(gpu_v(i, j), cpu_v(i, j));


  // Verify vector S
  for (int i = 0; i < row; i++)
    EXPECT_EQ(gpu_s(i), cpu_s(i));
}

