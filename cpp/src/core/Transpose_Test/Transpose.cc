// This is a test file which is used to demonstrate the functionality 
// of the transpose() function from the Eigen library.
// The function has been drafted in the cpu_operations.cc


#include "../../../../Eigen/Eigen/Dense"  // Relative path to Eigen library in project
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace Eigen;  // Improper namespace for testing

int main()
{
  // Creates a randomly initialized 2 by 2 matrix
  Matrix2i m = Matrix2i::Random();
  cout << "Here is the matrix m:" << endl << m << endl;
  cout << "Here is the transpose of m:" << endl << m.transpose() << endl;
  cout << "Here is the coefficient (1,0) in the transpose of m:" << endl
       << m.transpose()(1,0) << endl;
  cout << "Let us overwrite this coefficient with the value 0." << endl;
  m.transpose()(1,0) = 0;
  cout << "Now the matrix m is:" << endl << m << endl;
  return 0;
}
