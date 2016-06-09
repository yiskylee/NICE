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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include "include/matrix.h"
#include "include/vector.h"

namespace Nice {

namespace util {

template<typename T>
Matrix<T> FromFile(const std::string &input_file_path, int num_rows,
                   int num_cols) {
  std::ifstream input_file(input_file_path, std::ifstream::in);
  if (input_file) {
    std::cout << "File Open";
  } else {
    std::cout << "Cannot open file " + input_file_path + " , exiting...";
//    exit(1);

  }
  return Matrix<T>::Random(num_rows, num_cols);
}

template Matrix<int> FromFile<int>(const std::string &input_file_path,
                                   int num_rows, int num_cols);

}  // namespace util
}  // namespace Nice

//}
//  std::ifstream input_file(input_file_path);
//  if (input_file) {
////    for (int i = 0; i < num_rows_; i++)
////      for (int j = 0; j < num_cols_; j++)
////        input_file >> (*matrix_)(i, j);
//    input_file.close();
//    return true;
//  } else
//    return false;
//  }

