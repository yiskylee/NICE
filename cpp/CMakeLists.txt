cmake_minimum_required(VERSION 2.8.9)
project(nice)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(../Eigen include)

set(SOURCES src/core/cpu_operations.cc src/util/util.cc)

add_library(nice_lib SHARED ${SOURCES})
