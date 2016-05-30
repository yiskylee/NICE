################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Eigen/bench/basicbenchmark.cpp \
../Eigen/bench/benchBlasGemm.cpp \
../Eigen/bench/benchCholesky.cpp \
../Eigen/bench/benchEigenSolver.cpp \
../Eigen/bench/benchFFT.cpp \
../Eigen/bench/benchGeometry.cpp \
../Eigen/bench/benchVecAdd.cpp \
../Eigen/bench/bench_gemm.cpp \
../Eigen/bench/bench_norm.cpp \
../Eigen/bench/bench_reverse.cpp \
../Eigen/bench/bench_sum.cpp \
../Eigen/bench/benchmark.cpp \
../Eigen/bench/benchmarkSlice.cpp \
../Eigen/bench/benchmarkX.cpp \
../Eigen/bench/benchmarkXcwise.cpp \
../Eigen/bench/check_cache_queries.cpp \
../Eigen/bench/eig33.cpp \
../Eigen/bench/geometry.cpp \
../Eigen/bench/product_threshold.cpp \
../Eigen/bench/quat_slerp.cpp \
../Eigen/bench/quatmul.cpp \
../Eigen/bench/sparse_cholesky.cpp \
../Eigen/bench/sparse_dense_product.cpp \
../Eigen/bench/sparse_lu.cpp \
../Eigen/bench/sparse_product.cpp \
../Eigen/bench/sparse_randomsetter.cpp \
../Eigen/bench/sparse_setter.cpp \
../Eigen/bench/sparse_transpose.cpp \
../Eigen/bench/sparse_trisolver.cpp \
../Eigen/bench/spmv.cpp \
../Eigen/bench/vdw_new.cpp 

OBJS += \
./Eigen/bench/basicbenchmark.o \
./Eigen/bench/benchBlasGemm.o \
./Eigen/bench/benchCholesky.o \
./Eigen/bench/benchEigenSolver.o \
./Eigen/bench/benchFFT.o \
./Eigen/bench/benchGeometry.o \
./Eigen/bench/benchVecAdd.o \
./Eigen/bench/bench_gemm.o \
./Eigen/bench/bench_norm.o \
./Eigen/bench/bench_reverse.o \
./Eigen/bench/bench_sum.o \
./Eigen/bench/benchmark.o \
./Eigen/bench/benchmarkSlice.o \
./Eigen/bench/benchmarkX.o \
./Eigen/bench/benchmarkXcwise.o \
./Eigen/bench/check_cache_queries.o \
./Eigen/bench/eig33.o \
./Eigen/bench/geometry.o \
./Eigen/bench/product_threshold.o \
./Eigen/bench/quat_slerp.o \
./Eigen/bench/quatmul.o \
./Eigen/bench/sparse_cholesky.o \
./Eigen/bench/sparse_dense_product.o \
./Eigen/bench/sparse_lu.o \
./Eigen/bench/sparse_product.o \
./Eigen/bench/sparse_randomsetter.o \
./Eigen/bench/sparse_setter.o \
./Eigen/bench/sparse_transpose.o \
./Eigen/bench/sparse_trisolver.o \
./Eigen/bench/spmv.o \
./Eigen/bench/vdw_new.o 

CPP_DEPS += \
./Eigen/bench/basicbenchmark.d \
./Eigen/bench/benchBlasGemm.d \
./Eigen/bench/benchCholesky.d \
./Eigen/bench/benchEigenSolver.d \
./Eigen/bench/benchFFT.d \
./Eigen/bench/benchGeometry.d \
./Eigen/bench/benchVecAdd.d \
./Eigen/bench/bench_gemm.d \
./Eigen/bench/bench_norm.d \
./Eigen/bench/bench_reverse.d \
./Eigen/bench/bench_sum.d \
./Eigen/bench/benchmark.d \
./Eigen/bench/benchmarkSlice.d \
./Eigen/bench/benchmarkX.d \
./Eigen/bench/benchmarkXcwise.d \
./Eigen/bench/check_cache_queries.d \
./Eigen/bench/eig33.d \
./Eigen/bench/geometry.d \
./Eigen/bench/product_threshold.d \
./Eigen/bench/quat_slerp.d \
./Eigen/bench/quatmul.d \
./Eigen/bench/sparse_cholesky.d \
./Eigen/bench/sparse_dense_product.d \
./Eigen/bench/sparse_lu.d \
./Eigen/bench/sparse_product.d \
./Eigen/bench/sparse_randomsetter.d \
./Eigen/bench/sparse_setter.d \
./Eigen/bench/sparse_transpose.d \
./Eigen/bench/sparse_trisolver.d \
./Eigen/bench/spmv.d \
./Eigen/bench/vdw_new.d 


# Each subdirectory must supply rules for building sources it contributes
Eigen/bench/%.o: ../Eigen/bench/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -IEigen -I/home/jason.b/Desktop/Github/NICE/ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


