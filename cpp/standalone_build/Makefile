CUDA_VERSION = cuda-7.0
CPP = g++
NVCC = /usr/local/${CUDA_VERSION}/bin/nvcc
NICE_DIR = ${HOME}/Dropbox/git_project/NICE/cpp
CUDA_LIB = /usr/local/${CUDA_VERSION}/lib64
GTEST_LIB = ${NICE_DIR}/build/gtest/src/googletest-build/googlemock/gtest/
GTEST_INC = ${NICE_DIR}/build/gtest/src/googletest/googletest/include/
EIGEN_INC = ${NICE_DIR}/build/eigen/src/eigen
DLIB = ${NICE_DIR}/dlib/build
CCSOURCES = ${NICE_DIR}/test/model_test/kdac_test.cc ${NICE_DIR}/src/util/gpu_util.cc
CUSOURCES = ${NICE_DIR}/src/cuda/kdac_cuda.cu
ARCH = -gencode archi=compute_35,code=sm_35
BINARY = ./test_kdac.out
NVCCFLAGS = -arch=sm_50 -g -O3 -std=c++11

all:
	${NVCC} ${CCSOURCES} ${CUSOURCES} ${NVCCFLAGS} -I${EIGEN_INC} -I${GTEST_INC} -I${NICE_DIR} -L${CUDA_LIB}  -L${GTEST_LIB} -lcublas -lcusolver -lgtest -lgtest_main -o ${BINARY}
