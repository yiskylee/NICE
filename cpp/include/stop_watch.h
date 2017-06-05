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

#ifndef CPP_INCLUDE_STOP_WATCH_H_
#define CPP_INCLUDE_STOP_WATCH_H_

#include <sys/time.h>

namespace Nice {

// Class StopWatch
class StopWatch {
 private:
  struct timeval start_;
  struct timeval end_;
  bool started_;
 public:
  StopWatch() :
    started_(false) {}

  void Start() {
    gettimeofday(&start_, NULL);
    started_ = true;
  }
  void Stop() {
    if (started_) {
      gettimeofday(&end_, NULL);
    } else {
      std::cerr << "Make sure to start the timer before stopping it. "
                << std::endl;
      exit(1);
    }
  }
  double DiffInMs() {
    return (double)(end_.tv_sec * 1000 + static_cast<double>(end_.tv_usec) / 1000) -
      (double)(start_.tv_sec * 1000 + static_cast<double>(start_.tv_usec) / 1000);
  }
};

} // namespace Nice

#endif  // CPP_INCLUDE_STOP_WATCH_H_

