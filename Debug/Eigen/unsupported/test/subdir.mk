################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Eigen/unsupported/test/BVH.cpp \
../Eigen/unsupported/test/FFT.cpp \
../Eigen/unsupported/test/FFTW.cpp \
../Eigen/unsupported/test/NonLinearOptimization.cpp \
../Eigen/unsupported/test/NumericalDiff.cpp \
../Eigen/unsupported/test/alignedvector3.cpp \
../Eigen/unsupported/test/autodiff.cpp \
../Eigen/unsupported/test/bdcsvd.cpp \
../Eigen/unsupported/test/dgmres.cpp \
../Eigen/unsupported/test/forward_adolc.cpp \
../Eigen/unsupported/test/gmres.cpp \
../Eigen/unsupported/test/jacobisvd.cpp \
../Eigen/unsupported/test/kronecker_product.cpp \
../Eigen/unsupported/test/levenberg_marquardt.cpp \
../Eigen/unsupported/test/matrix_exponential.cpp \
../Eigen/unsupported/test/matrix_function.cpp \
../Eigen/unsupported/test/matrix_power.cpp \
../Eigen/unsupported/test/matrix_square_root.cpp \
../Eigen/unsupported/test/minres.cpp \
../Eigen/unsupported/test/mpreal_support.cpp \
../Eigen/unsupported/test/openglsupport.cpp \
../Eigen/unsupported/test/polynomialsolver.cpp \
../Eigen/unsupported/test/polynomialutils.cpp \
../Eigen/unsupported/test/sparse_extra.cpp \
../Eigen/unsupported/test/splines.cpp 

OBJS += \
./Eigen/unsupported/test/BVH.o \
./Eigen/unsupported/test/FFT.o \
./Eigen/unsupported/test/FFTW.o \
./Eigen/unsupported/test/NonLinearOptimization.o \
./Eigen/unsupported/test/NumericalDiff.o \
./Eigen/unsupported/test/alignedvector3.o \
./Eigen/unsupported/test/autodiff.o \
./Eigen/unsupported/test/bdcsvd.o \
./Eigen/unsupported/test/dgmres.o \
./Eigen/unsupported/test/forward_adolc.o \
./Eigen/unsupported/test/gmres.o \
./Eigen/unsupported/test/jacobisvd.o \
./Eigen/unsupported/test/kronecker_product.o \
./Eigen/unsupported/test/levenberg_marquardt.o \
./Eigen/unsupported/test/matrix_exponential.o \
./Eigen/unsupported/test/matrix_function.o \
./Eigen/unsupported/test/matrix_power.o \
./Eigen/unsupported/test/matrix_square_root.o \
./Eigen/unsupported/test/minres.o \
./Eigen/unsupported/test/mpreal_support.o \
./Eigen/unsupported/test/openglsupport.o \
./Eigen/unsupported/test/polynomialsolver.o \
./Eigen/unsupported/test/polynomialutils.o \
./Eigen/unsupported/test/sparse_extra.o \
./Eigen/unsupported/test/splines.o 

CPP_DEPS += \
./Eigen/unsupported/test/BVH.d \
./Eigen/unsupported/test/FFT.d \
./Eigen/unsupported/test/FFTW.d \
./Eigen/unsupported/test/NonLinearOptimization.d \
./Eigen/unsupported/test/NumericalDiff.d \
./Eigen/unsupported/test/alignedvector3.d \
./Eigen/unsupported/test/autodiff.d \
./Eigen/unsupported/test/bdcsvd.d \
./Eigen/unsupported/test/dgmres.d \
./Eigen/unsupported/test/forward_adolc.d \
./Eigen/unsupported/test/gmres.d \
./Eigen/unsupported/test/jacobisvd.d \
./Eigen/unsupported/test/kronecker_product.d \
./Eigen/unsupported/test/levenberg_marquardt.d \
./Eigen/unsupported/test/matrix_exponential.d \
./Eigen/unsupported/test/matrix_function.d \
./Eigen/unsupported/test/matrix_power.d \
./Eigen/unsupported/test/matrix_square_root.d \
./Eigen/unsupported/test/minres.d \
./Eigen/unsupported/test/mpreal_support.d \
./Eigen/unsupported/test/openglsupport.d \
./Eigen/unsupported/test/polynomialsolver.d \
./Eigen/unsupported/test/polynomialutils.d \
./Eigen/unsupported/test/sparse_extra.d \
./Eigen/unsupported/test/splines.d 


# Each subdirectory must supply rules for building sources it contributes
Eigen/unsupported/test/%.o: ../Eigen/unsupported/test/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/include/ -I/home/jason.b/Desktop/Github/NICE/Eigen -I/usr/include/gtest -I/home/jason.b/Desktop/Github/NICE/cpp/src/ -I/home/jason.b/Desktop/Github/NICE/ -O0 -g3 -Wall -c -fmessage-length=0 -std=c++14 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


