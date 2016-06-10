# Install script for directory: /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core

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
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen/src/Core" TYPE FILE FILES
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/NestByValue.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Block.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Dot.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/DenseBase.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/TriangularMatrix.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/NoAlias.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/DiagonalMatrix.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/StableNorm.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Fuzzy.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Flagged.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Stride.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Map.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/SelfCwiseBinaryOp.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Ref.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/PermutationMatrix.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Assign.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Transpose.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/ReturnByValue.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Visitor.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Transpositions.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/ArrayBase.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/CwiseBinaryOp.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/ForceAlignedAccess.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Functors.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Array.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/CwiseUnaryOp.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/DenseStorage.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/EigenBase.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/GenericPacketMath.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Random.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/ProductBase.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/MapBase.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Matrix.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/DiagonalProduct.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/VectorwiseOp.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/NumTraits.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/GlobalFunctions.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/CommaInitializer.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/ArrayWrapper.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/VectorBlock.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/MatrixBase.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/MathFunctions.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/IO.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Redux.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/SolveTriangular.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Assign_MKL.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Swap.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/GeneralProduct.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/BooleanRedux.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Replicate.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/SelfAdjointView.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Diagonal.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/CwiseUnaryView.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Reverse.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/BandMatrix.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/CwiseNullaryOp.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/CoreIterators.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/DenseCoeffsBase.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/Select.h"
    "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/Eigen/src/Core/PlainObjectBase.h"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")

IF(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  INCLUDE("/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/Eigen/src/Core/products/cmake_install.cmake")
  INCLUDE("/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/Eigen/src/Core/util/cmake_install.cmake")
  INCLUDE("/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/Eigen/src/Core/arch/cmake_install.cmake")

ENDIF(NOT CMAKE_INSTALL_LOCAL_ONLY)

