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
 
message(STATUS "Eigen path: " ${EIGEN_INCLUDE_DIRS})
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${COMMON_INCLUDES}) 

# Locate CUDA package 
find_package(CUDA)
if (CUDA_FOUND)

  # Print messge to indicate that CUDA exists
  message(STATUS "CUDA exists")

  #include the built-in findCUDA cmake scripts
  include(FindCUDA)
  
  # Set NVCC flags  
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -O3 -gencode arch=compute_50,code=sm_50)

  # Set source files
  file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} src/*.cc src/*.cu)
  message(STATUS "Source files: " ${SOURCES})

  # Find cusolver library
  find_library(CUDA_SOLVER_LIBRARY cusolver HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

  # Add NICE library together with CUDA
  cuda_include_directories(${CUDA_INCLUDE_DIR})
  cuda_add_library(${PROJECT_NAME_STR} SHARED ${SOURCES})
  add_dependencies(${PROJECT_NAME_STR} ${CUDA_SOLVER_LIBRARY})
  add_dependencies(${PROJECT_NAME_STR} ${CUDA_LIBRARIES})
  target_link_libraries(${PROJECT_NAME_STR} ${CUDA_SOLVER_LIBRARY} ${CUDA_LIBRARIES})

else()

  message(STATUS "CUDA not exists")

  # Set source files
  file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} src/*.cc)
  message(STATUS "Source files: " ${SOURCES})

  # Add NICE library
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

  # Add source files according to machine setup
  if (CUDA_FOUND)
    file(GLOB_RECURSE TEST_SRC_FILES RELATIVE 
         ${PROJECT_SOURCE_DIR} 
         test/cpu_operations_test/*.cc
         test/util_test/*.cc
         test/gpu_solver_test/*.cc)
  else()
    file(GLOB_RECURSE TEST_SRC_FILES RELATIVE 
         ${PROJECT_SOURCE_DIR} 
         test/cpu_operations_test/*.cc
         test/util_test/*.cc)
  endif()
  message(STATUS "Test case source files: " ${TEST_SRC_FILES})

  add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})

  add_dependencies(${PROJECT_TEST_NAME} googletest)
  add_dependencies(${PROJECT_TEST_NAME} eigen)
  add_dependencies(${PROJECT_TEST_NAME} ${PROJECT_NAME_STR})

  # Add dependencies and link libraries according to machine setup
  if (CUDA_FOUND)
    add_dependencies(${PROJECT_TEST_NAME} ${CUDA_SOLVER_LIBRARY})
    add_dependencies(${PROJECT_TEST_NAME} ${CUDA_LIBRARIES})
    target_link_libraries(${PROJECT_TEST_NAME}
        ${CUDA_SOLVER_LIBRARY}
        ${CUDA_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/build/libNice.so
        ${GTEST_LIBS_DIR}/libgtest.a
        ${GTEST_LIBS_DIR}/libgtest_main.a
        )
  else()
    target_link_libraries(${PROJECT_TEST_NAME}
        ${PROJECT_SOURCE_DIR}/build/libNice.so
        ${GTEST_LIBS_DIR}/libgtest.a
        ${GTEST_LIBS_DIR}/libgtest_main.a
        )
  endif()

  target_link_libraries(${PROJECT_TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})
  add_test(test1 ${PROJECT_TEST_NAME} "--gtest_color=yes")
endif()
