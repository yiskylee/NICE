cmake_minimum_required(VERSION 2.8.9)
project(nice)
get_cmake_property(_variableNames VARIABLES)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(../Eigen)

set(SOURCES src/core/cpu_operations.cc src/util/util.cc)


#Xiangyu
file(GLOB ABS src/core/*.cc)
file(GLOB REL RELATIVE {CMAKE_SOURCE_DIR} src/core/*.cc)
#file(GLOB REL RELATIVE ${CMAKE_SOURCE_DIR})
#message(STATUS "ABS: ${ABS}")
#message(STATUS "RELATIVE: ${REL}")
#find_file(CPP_LINT_PY 
#  NAMES cpplint.py 
#  HINTS ${CMAKE_SOURCE_DIR}
#  DOC "Google cpp style scan program.")
#if(NOT CPP_LINT_PY)
#  message ("cpplint.py not found")
#endif()
#message(STATUS "cpp-lint: ${CPP_LINT_PY}")
message(STATUS "cmake-command: ${CMAKE_COMMAND}")

# Test

if (enable-test)

#Xiangyu

#add_library(nice_lib SHARED ${SOURCES})


