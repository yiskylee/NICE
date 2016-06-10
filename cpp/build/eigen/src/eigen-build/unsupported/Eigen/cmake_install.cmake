# Install script for directory: /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/usr/local")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "Release")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/AdolcForward"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/AlignedVector3"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/ArpackSupport"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/AutoDiff"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/BVH"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/FFT"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/IterativeSolvers"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/KroneckerProduct"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/LevenbergMarquardt"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/MatrixFunctions"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/MoreVectorization"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/MPRealSupport"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/NonLinearOptimization"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/NumericalDiff"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/OpenGLSupport"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/Polynomials"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/Skyline"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/SparseExtra"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/Eigen/Splines"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")

IF(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  INCLUDE("/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/unsupported/Eigen/src/cmake_install.cmake")

ENDIF(NOT CMAKE_INSTALL_LOCAL_ONLY)

