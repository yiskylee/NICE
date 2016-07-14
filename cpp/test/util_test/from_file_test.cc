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

#include "include/util.h"
#include "include/matrix.h"
#include "gtest/gtest.h"

TEST(FromFileTest, IfFileNotExist) {
  ASSERT_DEATH(
      {
        Nice::Matrix<int> m = Nice::util::FromFile<int>(
            "../test/data_for_test/matrix_not_exist.txt");
      }
      , "Cannot open file .*, exiting...");
}

TEST(FromFileTest, IfFileExists) {
//  ::testing::internal::CaptureStdout();
  Nice::Matrix<int> m = Nice::util::FromFile<int>(
      "../test/data_for_test/matrix_2_2.txt");
  std::cout << m << std::endl;



//  std::string output = ::testing::internal::GetCapturedStdout();
//  EXPECT_STREQ(output.c_str(), "File Open");
}
