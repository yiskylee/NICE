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
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)                                                   
 
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MACOSX_RPATH 1)
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wall -ansi -Wno-deprecated -pthread ${OpenMP_CXX_FLAGS} -std=c++11)
endif()
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -Werror  -fopenmp")
add_definitions(-std=c++11)
#if("${CMAKE_COMPILER_ID}" STREQUAL "Clang")
#    add_definitions(-std=c++11)
#endif()
message(STATUS "Eigen path: " ${EIGEN_INCLUDE_DIRS})
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${COMMON_INCLUDES}) 

# Set source files
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} src/*.cc src/*.cu)
message(STATUS "Source files: " ${SOURCES})

# Locate CUDA package 
find_package(CUDA)
if (CUDA_FOUND)

  # Print messge to indicate that CUDA exists
  message(STATUS "CUDA exists")

  set(Python_ADDITIONAL_VERSIONS 3.4)                                             
#  find_package(OpenMP REQUIRED)                                                   
  find_package(CUDA QUIET REQUIRED)                                               
  find_package(PythonLibs 3 REQUIRED) 


  # Add defination
  add_definitions(-DNEED_CUDA)

  #include the built-in findCUDA cmake scripts
  include(FindCUDA)
 
  # Set NVCC flags  
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -O3 -gencode arch=compute_50,code=sm_50
                      -Xptxas=-v -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES)

  # Find cusolver library
  find_library(CUDA_SOLVER_LIBRARY cusolver HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  find_library(CUDA_BLAS_LIBRARY   cublas   HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
#  find_library(CUDA_KMEANS_LIBRARY libKMCUDA.so HINTS libraries/)

  # Add NICE library together with CUDA
  cuda_include_directories(${CUDA_INCLUDE_DIR})
  cuda_add_library(${PROJECT_NAME} SHARED ${SOURCES})
  add_dependencies(${PROJECT_NAME} ${CUDA_SOLVER_LIBRARY})
  add_dependencies(${PROJECT_NAME} ${CUDA_BLAS_LIBRARY})
  add_dependencies(${PROJECT_NAME} ${CUDA_LIBRARIES})
#  add_dependencies(${PROJECT_NAME} ${CUDA_KMEANS_LIBRARY})
  target_link_libraries(${PROJECT_NAME} 
    ${CUDA_SOLVER_LIBRARY} 
    ${CUDA_BLAS_LIBRARY} 
    ${CUDA_LIBRARIES} 
#    ${CUDA_KMEANS_LIBRARY}
    )
  if(PYTHONLIBS_FOUND)
    message("Python exists")
    include_directories(${PYTHON_INCLUDE_DIRS})                                   
    target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})                             
  endif() 

else()

  message(STATUS "CUDA not exists")
  
  # Add NICE library
  add_library(${PROJECT_NAME} SHARED ${SOURCES})

endif()

add_dependencies(${PROJECT_NAME} eigen)

# Test
option (enable-test "Compile all Nice unit tests" ON)

if (enable-test)
  include(gtest)
  enable_testing()
  set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)
  include_directories(${GTEST_INCLUDE_DIRS})

  # Add source files according to machine setup
  if (CUDA_FOUND)
    file(GLOB_RECURSE TEST_SRC_FILES RELATIVE 
         ${PROJECT_SOURCE_DIR} 
         test/cpu_operations_test/*.cc
         test/util_test/*.cc
         test/cpu_solver_test/*.cc
         test/gpu_operations_test/*.cc
         test/gpu_solver_test/*.cc
         test/gpu_kmeans_test/*.cc
         )
  else()
    file(GLOB_RECURSE TEST_SRC_FILES RELATIVE 
         ${PROJECT_SOURCE_DIR} 
         test/cpu_operations_test/*.cc
         test/util_test/*.cc
         test/cpu_solver_test/*.cc)
  endif()
  message(STATUS "Test case source files: " ${TEST_SRC_FILES})

  add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})

  add_dependencies(${PROJECT_TEST_NAME} googletest)
  add_dependencies(${PROJECT_TEST_NAME} eigen)
  add_dependencies(${PROJECT_TEST_NAME} ${PROJECT_NAME})

  # Add dependencies and link libraries according to machine setup
  if (CUDA_FOUND)
    add_dependencies(${PROJECT_TEST_NAME} ${CUDA_SOLVER_LIBRARY})
    add_dependencies(${PROJECT_TEST_NAME} ${CUDA_BLAS_LIBRARY})
    add_dependencies(${PROJECT_TEST_NAME} ${CUDA_LIBRARIES})
#    add_dependencies(${PROJECT_TEST_NAME} ${CUDA_KMEANS_LIBRARY})
    target_link_libraries(${PROJECT_TEST_NAME}
        ${CUDA_SOLVER_LIBRARY}
        ${CUDA_BLAS_LIBRARY}
        ${CUDA_LIBRARIES}
#        ${CUDA_KMEANS_LIBRARY}
        ${GTEST_LIBS_DIR}/libgtest.a
        ${GTEST_LIBS_DIR}/libgtest_main.a
        ${PROJECT_NAME}
        )
  else()
    target_link_libraries(${PROJECT_TEST_NAME}
        ${GTEST_LIBS_DIR}/libgtest.a
        ${GTEST_LIBS_DIR}/libgtest_main.a
        ${PROJECT_NAME}
        )
  endif()

  target_link_libraries(${PROJECT_TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})
  add_test(test1 ${PROJECT_TEST_NAME} "--gtest_color=yes")
endif()
