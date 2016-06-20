// The MIT License (MIT)
//
// Copyright (c) 2016 Northeastern University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#include <stdio.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

Nice::Matrix<bool> ma(4,4);  //Matrix for method input
Nice::Matrix<bool> mb(4,4);  //Expected Matrix output
Nice::Matrix<bool> mc(4,4);  //Will not be assigned value
Nice::Vector<bool> va(4); //Vector for method input
Nice::Vector<bool> vb(4); //Vector for method output
Nice::Vector<bool> vc(4);

TEST(MyTest, LogicalNotMatrix) {
  ma << 1, 1, 1, 1,
        1, 1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0;
  mb << 0, 0, 0, 0,
        0, 0, 0, 0,
        1, 1, 1, 1,
        1, 1, 1, 1;
  ASSERT_TRUE( mb.isApprox( Nice::CpuOperations<bool>::LogicalNot( ma ) ) );
}

TEST(MyTest, LogicalNotVector) {
  va << 1, 0, 1, 0;
  vb << 0, 1, 0, 1;
  ASSERT_TRUE( vb.isApprox( Nice::CpuOperations<bool>::LogicalNot( va ) ) );
}

TEST(MyTest, MatrixNoValue) {
  ASSERT_ANY_THROW(Nice::CpuOperations<bool>::LogicalNot(mc));
}
