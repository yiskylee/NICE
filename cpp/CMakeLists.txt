cmake_minimum_required(VERSION 3.0.0)
set(PROJECT_NAME Nice)
project(Nice)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(eigen)
include(check)
include(doc)
find_package(Threads REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
if(CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-Wall -ansi -Wno-deprecated -pthread -fPIC -std=c++11)
  #  add_definitions(-Wall -ansi -Wno-deprecated -std=c++11)
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  add_definitions(-Wall -ansi -Wno-deprecated -pthread -std=c++11) 
endif()
if(${APPLE})
  add_definitions(-Wall -ansi -Wno-deprecated -pthread -std=c++11)
endif()

set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${COMMON_INCLUDES})

# This is a Hack for Clion to know where eigen is
include_directories(./build/eigen/src/eigen)
include_directories(./)
include_directories(./build/gtest/src/googletest/googletest/include)

#get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#foreach(dir ${dirs})
#  message(STATUS "dir='${dir}'")
#endforeach()

set(enable-intelmkl)
option (enable-intelmkl "Compile NICE library with Intel MKL" OFF)
set(enable-condamkl)
option (enable-condamkl "Compile NICE library with Intel MKL in Anaconda" OFF)

#add_definitions(-DEIGEN_USE_LAPACKE)
#find_library(OPENBLAS openblas HINTS /home/xiangyu/anaconda/lib)

# Locate MKL in Intel dir
if (enable-intelmkl AND DEFINED ENV{MKLROOT})
# AND EXISTS ${MKLROOT})
  message(STATUS "INTEL MKL Found")
#  message(STATUS ${MKLROOT})
  message(STATUS $ENV{MKLROOT})
  add_definitions(-m64 -DEIGEN_USE_MKL_ALL -Wno-unused-local-typedefs)
  include_directories($ENV{MKLROOT}/include)
  find_library(MKL_LP64 mkl_intel_lp64 HINTS $ENV{MKLROOT}/lib/intel64)
  find_library(MKL_CORE mkl_core HINTS $ENV{MKLROOT}/lib/intel64)
  find_library(MKL_THREAD mkl_intel_thread HINTS $ENV{MKLROOT}/lib/intel64)
#  find_library(GNU_THREAD mkl_gnu_thread HINTS $ENV{MKLROOT}/lib/intel64)
  get_filename_component(INTEL_DIR $ENV{MKLROOT} DIRECTORY)
  message(STATUS INTEL_DIR: ${INTEL_DIR})
  find_library(INTEL_OMP iomp5 HINTS ${INTEL_DIR}/compiler/lib/intel64_lin)
#  find_library(INTEL_OMP iomp5 HINTS ${MKLROOT}/lib/intel64)
#  message(STATUS "INTEL_DIR: ${INTEL_DIR}")
else()
  message(STATUS "INTEL MKL NOT FOUND")
endif()

# Locate MKL in Anaconda
if (enable-condamkl AND DEFINED ENV{MKLROOTCONDA})
  message(STATUS "CONDA MKL Found")
  message(STATUS $ENV{MKLROOTCONDA})
  add_definitions(-m64 -DEIGEN_USE_MKL_ALL -Wno-unused-local-typedefs)
  include_directories($ENV{MKLROOT}/include)
  find_library(MKL_LP64 mkl_intel_lp64 HINTS $ENV{MKLROOTCONDA})
  find_library(MKL_CORE mkl_core HINTS $ENV{MKLROOTCONDA})
  find_library(MKL_THREAD mkl_intel_thread HINTS $ENV{MKLROOTCONDA})
#  find_library(GNU_THREAD mkl_gnu_thread HINTS $ENV{MKLROOT}/lib/intel64)
  find_library(INTEL_OMP iomp5 HINTS $ENV{MKLROOTCONDA})
else()
  message(STATUS "CONDA MKL NOT FOUND")
endif()
# Locate CUDA package
find_package(CUDA)

# Detect GPU device
# Dirty Hack to disable GPU
if(CUDA_FOUND)
  message("CUDA_FOUND")
  try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/has_cuda_gpu.c
    CMAKE_FLAGS
      -DINCLUDE_DIRECTORIES:STRING=${CUDA_TOOLKIT_INCLUDE}
      -DLINK_LIBRARIES:STRING=${CUDA_CUDART_LIBRARY}
    COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
    RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)

  # Display number of GPUs found
  message("${RUN_OUTPUT_VAR}")

  # COMPILE_RESULT_VAR is TRUE when compile succeeds
  # RUN_RESULT_VAR is zero when a GPU is found
  if(COMPILE_RESULT_VAR AND NOT RUN_RESULT_VAR)
      set(CUDA_AND_GPU TRUE CACHE BOOL "Whether CUDA-capable GPU is present")
  else()
      set(CUDA_AND_GPU FALSE CACHE BOOL "Whether CUDA-capable GPU is present")
  endif()
else()
  set(CUDA_AND_GPU FALSE CACHE BOOL "Whether CUDA-capable GPU is present")
endif()

# Dirty hack to not compile CUDA code
set(CUDA_AND_GPU FALSE)
# Set GPU True
# set(CUDA_AND_GPU TRUE)

# Build NICE library with/without CUDA
if (CUDA_AND_GPU)
#   Print messge to indicate that CUDA exists
  message(STATUS "GPU device exists")
  # Set source files
  file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} src/*.cc src/*.cu)

  # Add definition
  add_definitions(-DCUDA_AND_GPU)

  #include the built-in findCUDA cmake scripts
  include(FindCUDA)

  # Set NVCC flags
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -O3
#          -gencode arch=compute_20,code=sm_20
          -gencode arch=compute_30,code=sm_30
#          -gencode arch=compute_35,code=sm_35
#          -gencode arch=compute_37,code=sm_37
          -gencode arch=compute_50,code=sm_50
          --relaxed-constexpr
#          -gencode arch=compute_52,code=sm_52
          )


  # Find cusolver library
  find_library(CUDA_SOLVER_LIBRARY cusolver HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  find_library(CUDA_BLAS_LIBRARY   cublas   HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

  # Add NICE library together with CUDA
  cuda_include_directories(${CUDA_INCLUDE_DIR})
  cuda_add_library(${PROJECT_NAME} SHARED ${SOURCES})
  target_link_libraries(${PROJECT_NAME} 
                        ${CUDA_SOLVER_LIBRARY} 
                        ${CUDA_BLAS_LIBRARY} 
                        ${CUDA_LIBRARIES} 
                        ${OPENBLAS}
                        ${MKL_LP64} 
                        ${MKL_CORE} 
                        ${MKL_THREAD}
                        #${GNU_THREAD}
                        ${INTEL_OMP}
                        pthread m dl gomp)

else()

  message(STATUS "GPU device does not exist")

  # Set source files
  file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} src/*.cc)

  # Add NICE library
  add_library(${PROJECT_NAME} SHARED ${SOURCES} include/ism_cpu.h)
  target_link_libraries(${PROJECT_NAME}
                        ${MKL_LP64}
                        ${MKL_CORE}
                        ${MKL_THREAD}
                        ${INTEL_OMP}
                        ${OPENBLAS}
                        pthread m dl)

endif()

add_dependencies(${PROJECT_NAME} eigen)

# Test
option (enable-test "Compile all Nice unit tests" OFF)

message(STATUS "ENABLE-TEST: ${enable-test}")

if (enable-test)
  include(gtest)
  enable_testing()
  set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)
  include_directories(${GTEST_INCLUDE_DIRS})

  # Add source files according to machine setup
  if (CUDA_AND_GPU)
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
  add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
  add_dependencies(${PROJECT_TEST_NAME} googletest)
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
if (enable-interface)
	add_subdirectory(interface)
endif()
