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
// Position for Column-Major index
#define IDXC(i,j,ld) (((j)*(ld))+(i))
// Position for Row-Major index
#define IDXR(i,j,ld) (((i)*(ld))+(j))
// Pass in a timer and a function, the time taken by that function is then
// recorded in the timer
#define PROFILE(func, timer)\
  timer.Start();\
  func;\
  timer.Stop();\

#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <vector>
#include <math.h>

#include "include/matrix.h"
#include "include/vector.h"
//#include "include/cpu_operations.h"

namespace Nice {

namespace util {

/// This function reads and creates a matrix from a file
///
/// \param &input_file_path
/// Input string to file location
/// \param num_rows
/// The number of rows in the read-in matrix
/// \param num_cols
/// The number of columns in the read-in matrix
/// \param delimiter
/// Char that stands in-between coefficients
///
/// \return
/// This function returns a matrix of type T, that was created from a file
template<typename T>
Matrix<T> FromFile(const std::string &input_file_path,
                   int num_rows, int num_cols,
                   const std::string delimiter = " ") {
  // Reads in the file from "input_file_path"
  std::ifstream input_file(input_file_path, std::ifstream::in);
  Matrix<T> m(num_rows, num_cols);
  T coef;
  if (input_file) {
    int i = 0;
    // Iterates over every line in the input file
    while ( !input_file.eof() ) {
      std::string line;
      getline(input_file, line);
      if (line.find_first_not_of(' ') == std::string::npos)
        continue;
      if (delimiter == " " && line.find_first_of(',') != std::string::npos) {
        std::cerr << "File uses different delimiter than parameter! Use ','!";
        exit(1);
      } else if (delimiter == ",") {
        if (line.find_first_of(',') == std::string::npos) {
          std::cerr << "File uses different delimiter than parameter! Use ' '!";
          exit(1);
        }
        // Replaces every instance of the "delimiter" with whitespace for comma
        std::replace(line.begin(), line.end(), ',', ' ');
      } else if (delimiter != " ") {
        // If the function is called with an invalid delimiter, create an error
        std::cerr << "'" + delimiter + "' isn't an accepted delimiter";
        exit(1);
      }
      // Creates a stringstream out of every line in the file
      std::stringstream stream(line);
      int j = 0;
      // Reads in every coefficient in the string stream and puts it into
      //  the matrix
      while (stream >> coef) {
        m(i, j) = coef;
        ++j;
      }
      ++i;
    }
    return m;
  } else {
    // Error for when the file doesn't exist
    std::cerr << "Cannot open file " + input_file_path + ", exiting..."
              << std::endl;
    exit(1);
  }
}

/// This function reads and creates a matrix from a file
///
/// \param &input_file_path
/// Input string to file location
/// \param delimiter
/// Char that stands in-between coefficients
///
/// \return
/// This function returns a matrix of type T, that was created from a file
template<typename T>
Matrix<T> FromFile(const std::string &input_file_path,
                   const std::string delimiter = " ") {
  // Reads in the file from "input_file_path"
  std::ifstream input_file(input_file_path, std::ifstream::in);
  Matrix<T> m;
  std::string line;
  std::vector<T> temp_buffer;
  T coef;
  int num_cols = 0;
  int num_rows = 0;
  int cols_in_row;
  if (input_file) {
    // Iterates over every line in the input file
    while (!input_file.eof()) {
      getline(input_file, line);
      if (line.find_first_not_of(' ') == std::string::npos)
        continue;
      if (delimiter == " " && line.find_first_of(',') != std::string::npos) {
        std::cerr << "File uses different delimiter than parameter! Use ','!";
        exit(1);
      } else if (delimiter == ",") {
        if (line.find_first_of(',') == std::string::npos) {
          std::cerr << "File uses different delimiter than parameter! Use ' '!";
          exit(1);
        }
        // Replaces every instance of the "delimiter" with whitespace for comma
        std::replace(line.begin(), line.end(), ',', ' ');
      } else if (delimiter != " ") {
        // If the function is called with an invalid delimiter, create an error
        std::cerr << "'" + delimiter + "' isn't an accepted delimiter";
        exit(1);
      }
      // Creates a stringstream out of every line in the file
      std::stringstream stream(line);
      cols_in_row = 0;
      // Reads every coefficient in the stringstream into the temporary buffer
      while (stream >> coef) {
        temp_buffer.push_back(coef);
        ++cols_in_row;
      }
      // If the number of columns in the matrix hasn't been set, make it the
      // current number of columns in the row
      if (num_cols == 0) {
        num_cols = cols_in_row;
      // If the matrix in the file is shaped incorrectly, throw an error
      } else if (num_cols != cols_in_row) {
        std::cerr << "Problem with Matrix in: " + input_file_path +
                     ", exiting..." << std::endl;
        exit(1);
      }
      ++num_rows;
    }
    // Instantiate the matrix's size and feed it the coefficients in the
    // temporary buffer
    m.resize(num_rows, num_cols);
    for (int i = 0; i < num_rows; ++i)
      for (int j = 0; j < num_cols; ++j)
        m(i, j) = temp_buffer[i * num_cols + j];

    return m;
  } else {
    // Error for when the file doesn't exist
    std::cerr << "Cannot open file " + input_file_path + ", exiting..."
              << std::endl;
    exit(1);
  }
}

template<typename T>
Vector<T> FromFile(const std::string &input_file_path,
                   int num_elements) {
  std::ifstream input_file(input_file_path, std::ifstream::in);
  Vector<T> m(num_elements);
  if (input_file) {
    for (int i = 0; i < num_elements; i++)
        input_file >> m(i);
    return m;
  } else {
    std::cerr << "Cannot open file " + input_file_path + ", exiting..." <<
        std::endl;
    exit(1);
  }
}

template <typename T>
void ToFile(const Matrix<T> a,
            const std::string &output_file_path,
            const std::string delimiter = ",") {
  std::ofstream output(output_file_path, std::ofstream::out);
  if (output) {
    for (int i = 0; i < a.rows(); i++) {
      for (int j = 0; j < a.cols(); j++) {
        output << a(i, j);
        if (j == a.cols() - 1)
          output << std::endl;
        else
          output << delimiter;
      }
    }
  } else {
    std::cerr << "Cannot open file " + output_file_path + ", exiting..." <<
        std::endl;
    exit(1);
  }
  output.close();
}

template <typename T>
void ToFile(const Vector<T> a,
            const std::string &output_file_path) {
  std::ofstream output(output_file_path, std::ofstream::out);
  if (output) {
    for (int i = 0; i < a.rows(); i++) {
      output << a(i);
      output << std::endl;
    }
  } else {
    std::cerr << "Cannot open file " + output_file_path + ", exiting..." <<
        std::endl;
    exit(1);
  }
  output.close();
}

template<typename T>
static T reciprocal(T x) {
  return T(1) / x;
}

template <typename T>
void PrintMatrix(T* matrix, int row, int col, bool row_major = true) {
  for (int i = 0; i <= row; i++) {
    for (int j = 0; j <= col; j++) {
      if (row_major)
        std::cout << matrix[IDXR(i, j, col)] << "\t";
      else
        std::cout << matrix[IDXC(i, j, row)] << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T>
void Print(const Vector<T> &vector, std::string name) {
  std::cout << name << std::endl;
  for (int i = 0; i < vector.rows(); i++) {
    std::cout << vector(i) << " ";
  }
  std::cout << std::endl;
}

template <typename T>
void Print(const Matrix<T> &matrix, std::string name) {
  std::cout << name << std::endl;
  std::cout << matrix << " ";
  std::cout << std::endl;
}

template <typename T>
void Print(const T &scalar, std::string name) {
  std::cout << name << std::endl;
  std::cout << scalar << std::endl;
}

template <typename T>
bool CheckConverged(const Matrix<T> &matrix, const Matrix<T> &pre_matrix,
                    const T &threshold) {
  if ( (matrix.rows() != pre_matrix.rows()) ||
      (matrix.cols() != pre_matrix.cols()) )
    return false;
  T change = static_cast<T>((matrix - pre_matrix).norm()) /
      static_cast<T>(pre_matrix.norm());
  bool converged = (change < threshold);
  return converged;
}

template <typename T>
bool CheckConverged(const Vector<T> &vector, const Vector<T> &pre_vector,
                    const T &threshold) {
  if ( vector.rows() != pre_vector.rows() )
    return false;
  T change = static_cast<T>((vector - pre_vector).norm()) /
      static_cast<T>(pre_vector.norm());
  bool converged = (change < threshold);
  return converged;
}

template <typename T>
bool CheckConverged(const T &scalar, const T &pre_scalar, const T &threshold) {
  T change = fabs(scalar - pre_scalar) / fabs(scalar);
  bool converged = (change < threshold);
  return converged;
}

template <typename T>
void CheckFinite(const Matrix<T> &matrix, std::string name) {
  if (!matrix.allFinite()) {
    std::cout << name << " not finite: " << std::endl << matrix << std::endl;
    exit(1);
  }
}

template <typename T>
void CheckFinite(const Vector<T> &vector, std::string name) {
  if (!vector.allFinite()) {
    std::cout << name << " not finite: " << std::endl << vector << std::endl;
    exit(1);
  }
}


}  // namespace util

}  // namespace Nice

#endif  // CPP_INCLUDE_UTIL_H_
