cmake_minimum_required(VERSION 2.8.9)
set(PROJECT_NAME_STR Nice)
project(Nice)
find_package(Threads REQUIRED)

set(CMAKE_BUILD_TYPE Release)
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wall -ansi -Wno-deprecated -pthread -std=c++11)
endif()



set(SOURCES src/core/cpu_operations.cc src/util/util.cc)
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR} ../Eigen)

include_directories(${COMMON_INCLUDES})

add_library(Nice SHARED ${SOURCES})

# Test
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
option (enable-test "Compile all Nice unit tests" OFF)

if (enable-test)
  include(gtest)
  enable_testing()
  set(PROJECT_TEST_NAME ${PROJECT_NAME_STR}_test)
  include_directories(${GTEST_INCLUDE_DIRS} ${COMMON_INCLUDES})
  file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/test/cpu_operations_test/*.cc ${PROJECT_SOURCE_DIR}/test/util_test/*.cc)
  #file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/test *.cc)
  add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
  add_dependencies(${PROJECT_TEST_NAME} googletest)
  target_link_libraries(${PROJECT_TEST_NAME}
      ${GTEST_LIBS_DIR}/libgtest.a
      ${GTEST_LIBS_DIR}/libgtest_main.a
      ${PROJECT_SOURCE_DIR}/build/libNice.so
      )
  target_link_libraries(${PROJECT_TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})
  add_test(test1 ${PROJECT_TEST_NAME} "--gtest_color=yes")
endif()
