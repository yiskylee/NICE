# Install script for directory: /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen

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
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE FILE FILES
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/MetisSupport"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/Geometry"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/StdDeque"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/StdList"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/SparseQR"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/QR"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/Array"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/Eigen"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/Cholesky"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/Householder"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/Sparse"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/SVD"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/PardisoSupport"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/StdVector"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/SparseCholesky"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/IterativeLinearSolvers"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/Core"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/Eigen2Support"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/Eigenvalues"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/UmfPackSupport"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/SparseLU"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/SparseCore"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/Jacobi"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/QtAlignedMalloc"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/Dense"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/OrderingMethods"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/SuperLUSupport"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/LeastSquares"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/LU"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/SPQRSupport"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/CholmodSupport"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/PaStiXSupport"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")

IF(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  INCLUDE("/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/Eigen/src/cmake_install.cmake")

ENDIF(NOT CMAKE_INSTALL_LOCAL_ONLY)

