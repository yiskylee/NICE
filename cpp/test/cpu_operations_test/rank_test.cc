#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"


template<class T>
class RankTest : public ::testing::Test {
  public:
   Nice::Matrix<T> mat_;
   int calculated_ans_;
};

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(RankTest, MyTypes);

TYPED_TEST(RankTest, RankMatrix){
  this->mat_.resize(4, 4);
	this->mat_ <<  1.0, 3.0, 5.0, 2.0,
	     	       0.0, 1.0, 0.0, 3.0,
	     	       0.0, 0.0, 0.0, 1.0,
	     	       0.0, 0.0, 0.0, 0.0;

  int correct_ans = 3;

  this->calculated_ans_ = Nice::CpuOperations<TypeParam>::Rank(this->mat_);
  EXPECT_EQ(correct_ans, this->calculated_ans_);
}

