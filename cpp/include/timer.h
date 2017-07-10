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

#ifndef CPP_INCLUDE_TIMER_H_
#define CPP_INCLUDE_TIMER_H_

#include <vector>
#include <numeric>
#include "include/stop_watch.h"
#include "include/matrix.h"

namespace Nice {
// A timer includes a Stop Watch and a vector to store the time
class Timer {
 public:
  std::vector<double> vec_;
  std::vector<double> vec_temp_;


  void Start() {
    watch_.Start();
  }

  // Record the current elapsed time and store it temporary in vec_temp_
  void Record() {
    watch_.Stop();
    vec_temp_.push_back(watch_.DiffInMs());
  }

  void Stop() {
    watch_.Stop();
    vec_.push_back(watch_.DiffInMs());
  }

  // Sum up all the temporarily recorded times and store the sum to vec_
  // Then erase the temporary vector
  void SumRecords() {
    double total_time = std::accumulate(vec_temp_.begin(),
                                        vec_temp_.end(), 0.0);
    vec_.push_back(total_time);
    vec_temp_.clear();
  }

  double GetTotalTime() {
    return std::accumulate(vec_.begin(), vec_.end(), 0.0);
  }

  double GetAvgTimePerIter() {
    return GetTotalTime() / GetNumIters();
  }

  Matrix<double> GetTimePerIter() {
    Matrix<double> m(vec_.size(), 1);
    for (unsigned int i = 0; i < vec_.size(); i++)
      m(i, 0) = vec_[i];
    return m;
  }

  int GetNumIters() {
    return vec_.size();
  }

 private:
  StopWatch watch_;
};

}  // namespace Nice

#endif  // CPP_INCLUDE_TIMER_H_
