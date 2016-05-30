//This program is a simple test of an Eigen operation

#include <iostream>
#include "Eigen/Eigen/Dense"
//#include <gtest/gtest.h>

using namespace Eigen;
using namespace std;

//int add(int a, int b);

int main(int argc, char **argv)
{
 //::testing::InitGoogleTest(&argc, argv);
  MatrixXd m = MatrixXd::Random(3,3);
  m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
  cout << "m=" << endl << m << endl;
  int c=add(1, 3);
  return 0;
}
/*
int add(int a, int b)
{
  return (a + b);
}

TEST(Addition, CanAddTwoNumbers)
{
  EXPECT_EQ (5, add(1, 4));
}
*/
