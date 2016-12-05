cmake_minimum_required(VERSION 2.8.9)
if (${CMAKE_VERSION} VERSION_GREATER 3.5)
  cmake_policy(SET CMP0046 OLD)
endif()
set(PROJECT_NAME Nice)
project(Nice)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(eigen)
include(check)
include(doc)
include(dlib/cmake)
find_package(Threads REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
if(CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-Wall -ansi -Wno-deprecated -pthread -std=c++11)
  #  add_definitions(-Wall -ansi -Wno-deprecated -std=c++11)
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  add_definitions(-Wall -ansi -Wno-deprecated -pthread -std=c++11) 
endif()
if(${APPLE})
  add_definitions(-Wall -ansi -Wno-deprecated -pthread -std=c++11)
endif()
message(STATUS "Eigen path: " ${EIGEN_INCLUDE_DIRS})
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${COMMON_INCLUDES}) 


set(enable-mkl)
option (enable-mkl "Compile NICE library with MKL" OFF)

# Locate MKL
if (enable-mkl AND EXISTS ${MKL_ROOT})
  message(STATUS "MKL Found")
  add_definitions(-m64 -test "Compile all Nice unit tests" ON)

if (enable-test)
  include(gtest)
  enable_testing()
  set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)
  include_directories(${GTEST_INCLUDE_DIRS})

  # Add source files according to machine setup
  if (CUDA_HAVE_GPU)
    file(GLOB_RECURSE TEST_SRC_FILES RELATIVE 
         ${PROJECT_SOURCE_DIR} 
         test/cpu_operations_test/*.cc
         test/util_test/*.cc
         test/cpu_solver_test/*.cc
         test/gpu_operations_test/*.cc
         test/gpu_solver_test/*.cc
         test/model_test/*.cc
         )
  else()
    file(GLOB_RECURSE TEST_SRC_FILES RELATIVE 
         ${PROJECT_SOURCE_DIR} 
         test/cpu_operations_test/*.cc
         test/util_test/*.cc
         test/cpu_solver_test/*.cc
         test/model_test/*.cc
         )
  endif()
  message(STATUS "Test case source files: " ${TEST_SRC_FILES})

  add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})

  add_dependencies(${PROJECT_TEST_NAME} googletest)
  add_dependencies(${PROJECT_TEST_NAME} eigen)
  add_dependencies(${PROJECT_TEST_NAME} ${PROJECT_NAME})

  # Link libraries
  target_link_libraries(${PROJECT_TEST_NAME}
      ${GTEST_LIBS_DIR}/libgtest.a
      ${GTEST_LIBS_DIR}/libgtest_main.a
      ${PROJECT_NAME} 
      ${CMAKE_THREAD_LIBS_INIT})

  # Add test
  add_test(test1 ${PROJECT_TEST_NAME} "--gtest_color=yes")
endif()

# Build Python interface
add_subdirectory(interface)
