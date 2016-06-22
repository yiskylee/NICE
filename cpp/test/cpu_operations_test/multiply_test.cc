#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"
//#include "cpu_operations.cc" 

TEST(MultiplyAns, Multiply){  

  int scalar;
  scalar = 3; 
  
  MatrixXi a(3,3);
  a << 0, 1, 2,
       3, 2, 1,
       1, 3, 0;

  MatrixXi MultiplyAns(3,3);
  MultiplyAns << 0, 3, 6,
                 9, 6, 3,
                 3, 9, 0;

  Nice::Matrix<int> Multiply = scalar * a; //look up in Eigen library document
  for (int i = 0; i < 3; ++i) 
    for (int j = 0; j < 3; ++j)
      EXPECT_EQ(MultiplyAns(i,j), Multiply(i,j)); //check i,j notation w/Eigen

  std::cout << "Here is the scalar: " << scalar << std::endl; 
  std::cout << "\nHere is the initial matrix:\n\n" << a << std::endl << std::endl;
  std::cout << "\nThis is the product matrix: \n\n" << MultiplyAns << std::endl;
  std::cout << "\nThis is the calculated product: \n\n" << Multiply << std::endl;

}
