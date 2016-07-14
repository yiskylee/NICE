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

#ifndef CPP_INCLUDE_UTIL_H_
#define CPP_INCLUDE_UTIL_H_

#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "include/matrix.h"
#include "include/vector.h"

namespace Nice {

namespace util {
template<typename T>
Matrix<T> FromFile(const std::string &input_file_path,
                   int num_rows, int num_cols) {
  std::ifstream input_file(input_file_path, std::ifstream::in);
  Matrix<T> m(num_rows, num_cols);
  if (input_file) {
    for (int i = 0; i < num_rows; i++)
      for (int j = 0; j < num_cols; j++)
        input_file >> m(i, j);
    return m;
  } else {
    std::cerr << "Cannot open file " + input_file_path + ", exiting...";
    exit(1);
  }
}

}  // namespace util

}  // namespace Nice

#endif  // CPP_INCLUDE_UTIL_H_
