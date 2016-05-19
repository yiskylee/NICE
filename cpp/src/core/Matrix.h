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

#include <string>
#include <Eigen/Dense>
#ifndef CPP_SRC_CORE_MATRIX_H_
#define CPP_SRC_CORE_MATRIX_H_

#include <string>

namespace nice {

template<typename T>
class Matrix {
 public:
  Matrix();
  virtual ~Matrix();
  T GetRawBuffer() const;
  Matrix(int num_rows, int num_cols);
  Matrix(int num_rows, int num_cols, std::string);
  ~Matrix();
  int GetNumRows() const;
  int GetNumCols() const;
  bool FromFile(std::string);
  bool FromSql(std::string);
  void Print() const;
 private:
  std::shared_ptr<T> raw_buffer_;
  int num_rows_;
  int num_cols_;
};

} // namespace nice

#endif // CPP_SRC_CORE_MATRIX_H_

