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
// #include <stdio.h>
// #include <stdlib.h>
// #include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
// #include "include/matrix.h"
// #include "include/vector.h"


TEST(GPU_Vector_Vector_Dot_Product, Basic_Test) {
  Nice::Vector<float> a(9);
  a << 0.0, 1.0, 2.0,
       3.0, 2.0, 1.0,
       1.0, 3.0, 0.0;
  Nice::Vector<float> b(9);
  b << 1.0, 0.0, 2.0,
       2.0, 1.0, 0.0,
       0.0, 2.0, 1.0;
  float correct_ans = 18;
  float calc_ans = Nice::GpuOperations<float>::DotProduct(a, b);
  EXPECT_EQ(correct_ans, calc_ans);
}
