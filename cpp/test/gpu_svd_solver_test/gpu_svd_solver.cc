#include "include/gpu_svd_solver.h"
#include <iostream>
#include <stdio.h>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/matrix.h"
#include "include/vector.h"
/*
// --- Setting the host matrix 
void SetHostMatrix(NICE::Matrix<std::float> &A, char callNum){
    float num = 0.0; 
    int M = A.rows(); 
    int N = A.cols(); 
    for(unsigned int i = 0; i < M; i++){
        for(unsigned int j = 0; j < N; j++){
            A[j*M + i] = num + j*M + i;
        }
    }
    std::cout << "Matrix " << callNum << " is:" << std::endl;
    std::cout << A << std::endl;
}
*/
// This function takes a matrix as a parameter and returns its SVD


TEST(Compute, GpuSvdCorrect) {
  Nice::GpuSvdSolver<float> SvdSolver; 
  Nice::Matrix<float> m1 = Eigen::MatrixXf::Random(2,2); 
  m1 << 1,2,3,4;
  std::cout << "Matrix 1 is:" << std::endl;
    std::cout << m1 << std::endl;
//  SetHostMatrix(m1, '1'); 
  SvdSolver.Compute(m1);

  Nice::Vector<float> m1_S     = SvdSolver.SingularValues(); 
  Nice::Matrix<float> m1_U     = SvdSolver.MatrixU();
  Nice::Matrix<float> m1_V     = SvdSolver.MatrixV();  

  Nice::Vector<float> m1_S_ref; m1_S_ref <<  3.70246 , 0.540182;
  Nice::Matrix<float> m1_U_ref; m1_U_ref << -0.525731, -0.850651, -0.850651, 0.525731; 
  Nice::Matrix<float> m1_V_ref; m1_V_ref << -0.229753, 0.973249, -0.973249, -0.229753; 

  EXPECT_EQ (m1_S_ref, m1_S);
  EXPECT_EQ (m1_U_ref, m1_U);
  EXPECT_EQ (m1_V_ref, m1_V);
}



// Start and run the tests
//int main(int argc, char **argv) {
//  ::testing::InitGoogleTest(&argc, argv);
//  return RUN_ALL_TESTS();
//}

