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

#ifndef CPP_INCLUDE_KDAC_PROFILER_H
#define CPP_INCLUDE_KDAC_PROFILER_H

#include <vector>
#include <numeric>
#include "include/timer.h"
#include "include/stop_watch.h"

namespace Nice {
// A profiler includes one timer for a function or a partition of code
struct KDACProfiler {
  Timer fit;
  Timer fit_loop;
  Timer u;
  Timer w;
  Timer w_part1;
  Timer w_part2;
  Timer w_part3;
  Timer w_part4;
  Timer w_part5;
  Timer w_part6;
  Timer w_part7;
  Timer w_part8;
  Timer init;
  Timer init_a_gpu;
  Timer init_a_cpu;
  Timer kmeans;
};
} // namespace Nice
#endif  // CPP_INCLUDE_KDAC_PROFILER_H