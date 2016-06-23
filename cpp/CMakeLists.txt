cmake_minimum_required(VERSION 2.8.9)
set(PROJECT_NAME_STR Nice)
project(Nice)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(eigen)
include(check)
find_package(Threads REQUIRED)

set(CMAKE_BUILD_TYPE Release)
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wall -ansi -Wno-deprecated -pthread -std=c++11)
endif()
 
message(STATUS ${EIGEN_INCLUDE_DIRS})
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${COMMON_INCLUDES}) 


if(EXISTS /usr/local/cuda/)
    message("-- CUDA Found")
   
    find_package(CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lcuda -lcudart -std=c++11 -O3 -gencode arch=compute_50,code=sm_50)
    set(SOURCES src/core/cpu_operations.cc src/core/gpu_svd_solver.cu src/util/util.cc)
    set(CUDA_TEST_FILES ${PROJECT_SOURCE_DIR}/test/gpu_svd_solver_test/*.cc)
    find_library(CUDA_LIBRARY cusolver HINTS /usr/local/cuda/lib64)
    find_library(CUDA_ERROR cudart HINTS /usr/local/cuda/lib64)
    message(STATUS ${CUDA_LIBRARY})
    message(STATUS ${CUDA_ERROR})
    cuda_include_directories(${CUDA_INCLUDE_DIR})
    cuda_add_library(${PROJECT_NAME_STR} SHARED ${SOURCES})
    add_dependencies(${PROJECT_NAME_STR} ${CUDA_ERROR})
    add_dependencies(${PROJECT_NAME_STR} ${CUDA_LIBRARY})
    target_link_libraries(${PROJECT_NAME_STR} ${CUDA_ERROR} ${CUDA_LIBRARY})
else()
   message("-- CUDA Not Found")

   set(SOURCES src/core/cpu_operations.cc src/util/util.cc)
   add_library(${PROJECT_NAME_STR} SHARED ${SOURCES})

endif()

add_dependencies(${PROJECT_NAME_STR} eigen)

# Test
option (enable-test "Compile all Nice unit tests" ON)

if (enable-test)
  include(gtest)
  enable_testing()
  set(PROJECT_TEST_NAME ${PROJECT_NAME_STR}_test)
  include_directories(${GTEST_INCLUDE_DIRS})
  file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/test/cpu_operations_test/*.cc ${PROJECT_SOURCE_DIR}/test/util_test/*.cc ${CUDA_TEST_FILES})
  #file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/test *.cc)
  add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
  add_dependencies(${PROJECT_TEST_NAME} ${CUDA_ERROR})
  add_dependencies(${PROJECT_TEST_NAME} ${CUDA_LIBRARY})
  add_dependencies(${PROJECT_TEST_NAME} googletest)
  add_dependencies(${PROJECT_TEST_NAME} eigen)
  add_dependencies(${PROJECT_TEST_NAME} ${PROJECT_NAME_STR})
  target_link_libraries(${PROJECT_TEST_NAME}
      ${CUDA_ERROR}
      ${CUDA_LIBRARY}
      ${PROJECT_SOURCE_DIR}/build/libNice.so
      ${GTEST_LIBS_DIR}/libgtest.a
      ${GTEST_LIBS_DIR}/libgtest_main.a
      )
  target_link_libraries(${PROJECT_TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})
  add_test(test1 ${PROJECT_TEST_NAME} "--gtest_color=yes")
endif()
