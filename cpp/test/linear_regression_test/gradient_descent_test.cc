#include <stdio.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/linear_regression.h"
#include "include/matrix.h"

template<class T>
class GradientDescentTest : public ::testing::Test {
 public:
  Nice::LinearRegression<T> *lr;
  Nice::Matrix<T> data_;
  Nice::Vector<T> y_;
  Nice::Vector<T> gd_;
  Nice::Vector<T> theta_;
  Nice::Vector<T> expected_ans_;
};

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(GradientDescentTest, MyTypes);

TYPED_TEST(GradientDescentTest, Test1) {
  this->data_.resize(12, 2);
  this->data_<< 1.0, 0.0,
                2.0, 0.0,
                3.0, 0.0,
                4.0, 0.0,
                5.0, 0.0,
                6.0, 0.0,
                7.0, 0.0,
                8.0, 0.0,
                9.0, 0.0,
                10.0, 0.0,
                11.0, 0.0,
                12.0, 0.0;
  this->y_.resize(12, 1);
  this->y_<<7.0,
            9.0,
            11.0,
            13.0,
            15.0,
            17.0,
            19.0,
            21.0,
            23.0,
            25.0,
            27.0,
            29.0;
  this->theta_.resize(3);
  this->theta_<<10.0,
                20.0,
                30.0;
  this->gd_.resize(3);

  this->lr = new Nice::LinearRegression<TypeParam>(this->data_,
                                                   this->y_);
  this->gd_  = this->lr->Fit();

  this->expected_ans_.resize(3);
  this->expected_ans_<< 5.0,
                        2.0,
                        0.0;
  delete this->lr;
  ASSERT_NEAR(this->gd_(0), this->expected_ans_(0), 0.1);
}
