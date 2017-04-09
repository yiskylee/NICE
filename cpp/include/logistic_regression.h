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

#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <string>
#include <iostream>
#include <cmath>
#include "include/matrix.h"
#include "include/vector.h"
#include "include/kernel_types.h"
#include "Eigen/SVD"
#include "include/svd_solver.h"
#include "include/util.h"
#include <typeinfo>

namespace Nice {

// Abstract class of common matrix operation interface
template<typename T>
class LogisticRegression {
  private:
    // Holds the data needed to process and get a fit.
    Matrix<T> training_x;
    Vector<T> training_y;
    Vector<T> theta_params;

    // Parameters used for the calculation
    int number_iterations = 10000000;
    double alpha = 0.001;
    
    // Returns the value of the sigmoid function
    T get_sigmoid(T sum){
    
      return (1 / (1 + exp(-1 * sum)));
    }

    T get_hypothesis(Vector<T> curr_x){
    
      T temp_val = 0;
      for (int i = 0; i < theta_params.size(); i++){
      	temp_val += curr_x[i] * theta_params[i];
      }
      
      T sigmoid_val = get_sigmoid(temp_val);
      return sigmoid_val;
    }
    
    T get_cost(){
      T error_sum = 0;
      for (int i = 0; i <= training_y.rows(); i++){
      	Vector<T> curr_x = training_x.col(i);
      	T hypothesis = get_hypothesis(curr_x);
      	if (training_y[i] == 1){
      	  error_sum += (training_y[i] * log(hypothesis));
      	}
      	else if (training_y[i] == 0){
      	  error_sum += ((1 - training_y[i]) * log(1-hypothesis));
        }
	  } 
	  return (error_sum * (-1/training_y.rows()));
    }

    T get_cost_function_derivative(int j){
      T error_sum = 0;
      for (int i = 0; i < training_y.rows(); i++){
      	Vector<T> curr_x = training_x.col(i);
      	T curr_j = curr_x[j];
      	T hypothesis = get_hypothesis(curr_x);
      	error_sum += (hypothesis - training_y[i]) * curr_j;
      } 
      T cost;
      cost = (alpha / training_y.rows()) * error_sum;
      return cost;
    }
    
    void get_gradient(){
    
      Vector<T> new_theta_params;
      new_theta_params.resize(2);
      for (int j = 0; j < theta_params.size(); j++){
        T CDF = get_cost_function_derivative(j);
        //std::cout << "CDF is: " << get_cost();
        //std::cout << CDF << std::endl; 
        new_theta_params[j] = theta_params[j] - CDF;
        
      }    
      theta_params = new_theta_params;
    }
    
  public:
    LogisticRegression() {}
    
    Vector<T> logistic_regression(Matrix<T> in_training_x, Vector<T> in_training_y, Vector<T> in_theta_params){
    
    training_x = in_training_x;
    training_y = in_training_y;
    theta_params = in_theta_params;
    training_x.resize(10, 2);
    training_y.resize(10);
    theta_params.resize(2);
    std::cout << training_x << std::endl;
    std::cout << training_y << std::endl;
      for (int i = 0; i <= number_iterations; i++){
        get_gradient();
        if (i % 1000 == 0){
          std::cout << "Theta is: " <<std::endl; 
          for (int j = 0; j < theta_params.size(); j++){
            std::cout << theta_params[j] << ", "<< std::endl;
          }
          std::cout << "Iteration is: " <<std::endl; 
          std::cout << i <<std::endl; 
          
        }
      }
      std::cout << training_y.rows() <<std::endl; 
      std::cout << training_y.cols() <<std::endl; 
      return theta_params;
    }
};
} 
#endif
