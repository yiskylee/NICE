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


#include "include/gpu_operations.h"
#include "Eigen/Dense"
#include "gtest/gtest.h"

TEST(GPU_Matrix_Scalar_Multiply, Basic_Test) {
  Nice::Matrix<float> a(3, 3);
  a << 0.0, 1.0, 2.0,
       3.0, 2.0, 1.0,
       1.0, 3.0, 0.0;

  float b = 2.0;

  Nice::Matrix<float> correct_ans(3, 3);
  correct_ans << 0.0, 2.0, 4.0,
                 6.0, 4.0, 2.0,
                 2.0, 6.0, 0.0;
  Nice::Matrix<float> calc_ans = Nice::GpuOperations<float>::Multiply(a, b);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        EXPECT_EQ(correct_ans(i, j), calc_ans(i, j));
}
