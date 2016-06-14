#include <iostream>
#include "Eigen/Dense"
#include <stdio.h>

using namespace std;
using namespace Eigen; 

//template<typename T>
//static Matrix<T> Add(const Matrix<T> &a, const Matrix<T> &b)
//{
//	cout << a << endl;
//	cout << b << endl;
//}

int main ()
{
	Matrix2i m;
	m << 1, 2, 3, 4;
	
	cout << m << endl;
	
return 0;
}

