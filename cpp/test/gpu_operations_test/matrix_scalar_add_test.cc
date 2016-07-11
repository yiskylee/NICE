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

TEST(GPU_MATRIX_SCALAR_ADD, Basic_Test) {
  Nice::Matrix<float> a(3,4);
  a << 1, 2, 3, 4,
       1, 2, 3, 4,
       1, 2, 3, 4;
  Nice::Matrix<float> b = Nice::GpuOperations<float>::Add(a, 1);
  std::cout << "output:\n" << b << std::endl;
  Nice::Matrix<float> answer(3, 4);
  answer << 2, 3, 4, 5,
            2, 3, 4, 5,
            2, 3, 4, 5;
  ASSERT_TRUE(answer.isApprox(b));
}
