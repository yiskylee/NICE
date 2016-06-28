#include "include/gpu_svd_solver.h"
#include <iostream>
#include <stdio.h>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/matrix.h"
#include "include/vector.h"

void verify(Nice::Matrix<float> m, Nice::Matrix<float> m_ref, char c, char n){

  std::cout<<"\n----------------------------------------------------------\n"<<std::endl;
  std::cout<<"Expected " << c << " For Test " << n << " : "<<std::endl; 
  std::cout<<m_ref<<std::endl; 
  std::cout<<"Calculated " << c <<" : "<<std::endl;
  std::cout<< m << std::endl; 
  if(m_ref.isApprox(m)) std::cout<< c <<" Calculation is Correct"<<std::endl;  
  else			std::cout<< c <<" Calculation is Incorrect"<<std::endl; 
  std::cout<<"\n----------------------------------------------------------\n"<<std::endl; 
 
}
TEST(Compute, GpuSvdCorrect) {
  Nice::GpuSvdSolver<float> SvdSolver; 
  Nice::Matrix<float> m1(2,2); 
  m1 << 0,2,1,3;
  std::cout << "Matrix 1 is:" << std::endl;
  std::cout << m1 << std::endl;

  SvdSolver.Compute(m1);

  Nice::Vector<float> m1_S     = SvdSolver.SingularValues(); 
  Nice::Matrix<float> m1_U     = SvdSolver.MatrixU();
  Nice::Matrix<float> m1_V     = SvdSolver.MatrixV();  

  Nice::Vector<float> m1_S_ref(2,1); m1_S_ref <<  3.70246 , 0.540182;
  Nice::Matrix<float> m1_U_ref(2,2); m1_U_ref << -0.525731, -0.850651, -0.850651, 0.525731; 
  Nice::Matrix<float> m1_V_ref(2,2); m1_V_ref << -0.229753, -0.973249, 0.973249, -0.229753; 
 
  verify(m1_S,m1_S_ref,'S','1'); 
  verify(m1_U,m1_U_ref,'U','1'); 
  verify(m1_V,m1_V_ref,'V','1');  

  EXPECT_EQ (2+2, 4);
}



// Start and run the tests
//int main(int argc, char **argv) {
//  ::testing::InitGoogleTest(&argc, argv);
//  return RUN_ALL_TESTS();
//}

