#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

template<class T>
class NormTest : public ::testing::Test {
  public:
   Nice::Matrix<T> norm_matrix_;
   int p_;
   int axis_;
   Nice::Vector<T> calculated_norm_;
};

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(NormTest, MyTypes);

TYPED_TEST(NormTest, NormMatrix){
  this->norm_matrix_.resize(3,3);
	this->norm_matrix_ << 
      1.0, 2.0, 3.0,
	    4.0, 5.0, 6.0,
	    7.0, 8.0, 9.0;
  float correct_norm[3] = {sqrt(66), sqrt(93), sqrt(126)};
  this->calculated_norm_ = Nice::CpuOperations<TypeParam>::Norm(this->norm_matrix_,this->p_,this->axis_);
  for (int i = 0; i < 3; i++)
    ASSERT_NEAR(correct_norm[i], this->calculated_norm_(i),0.001);
}

TYPED_TEST(NormTest, NormMatrix2){
  this->norm_matrix_.resize(3,4);
  this->norm_matrix_ <<
                 1.0, 2.0, 3.0, 4.0,
      		       5.0, 6.0, 7.0, 8.0,
	               9.0, 10.0, 11.0, 12.0;
  float correct_norm[4] = {sqrt(107), sqrt(140), sqrt(179), sqrt(224)};
  this-> calculated_norm_ =Nice::CpuOperations<TypeParam>::Norm(this->norm_matrix_,this->p_,this->axis_);
  for(int i = 0; i < 4; i++){
	  ASSERT_NEAR(correct_norm[i], this->calculated_norm_(i),0.001);
}
}
