
#include <stdio.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"
#include "include/vector.h"

TEST(InverseTestInt, SimpleTest) {

	Nice::Matrix<int> a;
	a.setRandom(3,3);
	a.inverse();
}
