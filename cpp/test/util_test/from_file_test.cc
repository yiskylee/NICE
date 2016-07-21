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
#include "include/util.h"
#include "include/matrix.h"
#include "gtest/gtest.h"

template<class T>
class FromFileTest : public ::testing::Test {
 public:
  Nice::Matrix<T> expected;
  Nice::Matrix<T> result;

  void Filer(std::string input_file_path, std::string d) {
    result = Nice::util::FromFile<T>(input_file_path, d);
  }
  void Filer(std::string input_file_path, int rows, int cols, std::string d) {
    result = Nice::util::FromFile<T>(input_file_path, rows, cols, d);
  }
};

typedef ::testing::Types<int, float, double> MyTypes;
TYPED_TEST_CASE(FromFileTest, MyTypes);

TYPED_TEST(FromFileTest, IfFileNotExist) {
  ASSERT_DEATH(this->Filer(
                           "../test/data_for_test/FromFile/matrix_not_exist.txt"
                           , " "), "Cannot open file .*, exiting...");
}

TYPED_TEST(FromFileTest, IfFileExists) {
  this->Filer("../test/data_for_test/FromFile/matrix_2_2.txt", " ");
  this->expected.resize(2, 2);
  this->expected << 1, 2,
                    3, 99;
  for (int i = 0; i < this->expected.rows(); ++i) {
    for (int j = 0; j < this->expected.cols(); ++j) {
      EXPECT_NEAR(this->expected(i, j), this->result(i, j), 0.0001);
    }
  }
}

TYPED_TEST(FromFileTest, NonSquareMatrix) {
  this->Filer("../test/data_for_test/FromFile/matrix_2_3.txt", " ");
  this->expected.resize(2, 3);
  this->expected << 1, 2, 3,
                    4, 5, 6;
  for (int i = 0; i < this->expected.rows(); ++i) {
    for (int j = 0; j < this->expected.cols(); ++j) {
      EXPECT_NEAR(this->expected(i, j), this->result(i, j), 0.0001);
    }
  }
}

TYPED_TEST(FromFileTest, MatrixWrongSize) {
  ASSERT_DEATH(this->Filer(
                          "../test/data_for_test/FromFile/matrix_wrong_size.txt"
                          , " "), ".*");
}

TYPED_TEST(FromFileTest, IfFileNotExistsRowsAndColsMethod) {
  ASSERT_DEATH(
      {
        this->Filer(
            "../test/data_for_test/FromFile/matrix_not_exist.txt", 2, 2, " ");
      }
      , "Cannot open file .*, exiting...");
}

TYPED_TEST(FromFileTest, IfFileExistsRowsAndColsMethod) {
  this->Filer("../test/data_for_test/FromFile/matrix_2_2.txt", 2, 2, " ");
  this->expected.resize(2, 2);
  this->expected << 1, 2,
                    3, 99;
  for (int i = 0; i < this->expected.rows(); ++i) {
    for (int j = 0; j < this->expected.cols(); ++j) {
      EXPECT_NEAR(this->expected(i, j), this->result(i, j), 0.0001);
    }
  }
}

TYPED_TEST(FromFileTest, CSVTest) {
  this->Filer("../test/data_for_test/FromFile/matrix_2_2.csv", ",");
  this->expected.resize(2, 2);
  this->expected << 1, 2,
                    3, 99;
  for (int i = 0; i < this->expected.rows(); ++i) {
    for (int j = 0; j < this->expected.cols(); ++j) {
      EXPECT_NEAR(this->expected(i, j), this->result(i, j), 0.0001);
    }
  }
}

TYPED_TEST(FromFileTest, CSVRowsAndColsMethodTest) {
  this->Filer("../test/data_for_test/FromFile/matrix_2_2.csv", 2, 2, ",");
  this->expected.resize(2, 2);
  this->expected << 1, 2,
                    3, 99;
  for (int i = 0; i < this->expected.rows(); ++i) {
    for (int j = 0; j < this->expected.cols(); ++j) {
      EXPECT_NEAR(this->expected(i, j), this->result(i, j), 0.0001);
    }
  }
}

TYPED_TEST(FromFileTest, OtherTest) {
  ASSERT_DEATH(this->Filer("../test/data_for_test/FromFile/matrix_2_2.csv"
                          , "."), ".*");
}

TYPED_TEST(FromFileTest, OtherTestRowsAndCols) {
  ASSERT_DEATH(this->Filer("../test/data_for_test/FromFile/matrix_2_2.csv",
                           2, 2, "."), ".*");
}

TYPED_TEST(FromFileTest, WrongDelimiterSpace) {
  ASSERT_DEATH(this->Filer("../test/data_for_test/FromFile/matrix_2_2.csv",
                           " "), ".*");
}

TYPED_TEST(FromFileTest, WrongDelimiterComma) {
  ASSERT_DEATH(this->Filer("../test/data_for_test/FromFile/matrix_2_2.txt",
                           ","), ".*");
}

TYPED_TEST(FromFileTest, WrongDelimiterSpaceRowsAndCols) {
  ASSERT_DEATH(this->Filer("../test/data_for_test/FromFile/matrix_2_2.csv",
                           2, 2, " "), ".*");
}

TYPED_TEST(FromFileTest, WrongDelimiterCommaRowsAndcols) {
  ASSERT_DEATH(this->Filer("../test/data_for_test/FromFile/matrix_2_2.txt",
                           2, 2, ","),
                           ".*");
}
