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

#include "Matrix.h"
#include <stdlib.h>
#include <fstream>
#include <iostream>



namespace nice {
template<typename T>
Matrix<T>::Matrix(int num_rows, int num_cols) {
	num_rows_ = num_rows;
	num_cols_ = num_cols;
	matrix_.resize(num_rows_, num_cols_);
	raw_buffer_ = &matrix_(0);
}

template<typename T>
Matrix<T>::Matrix(int num_rows, int num_cols, std::string input_file_path) {
	num_rows_ = num_rows;
	num_cols_ = num_cols;
	matrix_.resize(num_rows_, num_cols_);
	raw_buffer_ = &matrix_(0);
	if (!FromFile(input_file_path)) {
		std::cerr << "Error opening file: " + input_file_path << std::endl;
		exit(-1);
	}
}

template<typename T>
Matrix<T>::~Matrix() {
}
template<typename T>
int Matrix<T>::GetNumRows() const {
	return num_rows_;
}
template<typename T>
int Matrix<T>::GetNumCols() const {
	return num_cols_;
}
template<typename T>
bool Matrix<T>::FromFile(std::string input_file_path) {
	std::ifstream input_file(input_file_path);
	if (input_file) {
		for (int i = 0; i < num_rows_; i++)
			for (int j = 0; j < num_cols_; j++)
				input_file >> (*matrix_)(i, j);
		input_file.close();
		return true;
	}
	else
		return false;
}
template<typename T>
T Matrix<T>::Get(int row_num, int col_num) const {
	// Need to use return some Map stuff?
	Eigen::Matrix m = Eigen::Map<Eigen::Matrix<T, num_rows_, num_cols_>> (raw_buffer_);
	return m(row_num, col_num);
}
}
