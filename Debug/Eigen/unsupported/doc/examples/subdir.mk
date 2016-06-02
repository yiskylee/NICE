################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Eigen/unsupported/doc/examples/BVH_Example.cpp \
../Eigen/unsupported/doc/examples/FFT.cpp \
../Eigen/unsupported/doc/examples/MatrixExponential.cpp \
../Eigen/unsupported/doc/examples/MatrixFunction.cpp \
../Eigen/unsupported/doc/examples/MatrixLogarithm.cpp \
../Eigen/unsupported/doc/examples/MatrixPower.cpp \
../Eigen/unsupported/doc/examples/MatrixPower_optimal.cpp \
../Eigen/unsupported/doc/examples/MatrixSine.cpp \
../Eigen/unsupported/doc/examples/MatrixSinh.cpp \
../Eigen/unsupported/doc/examples/MatrixSquareRoot.cpp \
../Eigen/unsupported/doc/examples/PolynomialSolver1.cpp \
../Eigen/unsupported/doc/examples/PolynomialUtils1.cpp 

OBJS += \
./Eigen/unsupported/doc/examples/BVH_Example.o \
./Eigen/unsupported/doc/examples/FFT.o \
./Eigen/unsupported/doc/examples/MatrixExponential.o \
./Eigen/unsupported/doc/examples/MatrixFunction.o \
./Eigen/unsupported/doc/examples/MatrixLogarithm.o \
./Eigen/unsupported/doc/examples/MatrixPower.o \
./Eigen/unsupported/doc/examples/MatrixPower_optimal.o \
./Eigen/unsupported/doc/examples/MatrixSine.o \
./Eigen/unsupported/doc/examples/MatrixSinh.o \
./Eigen/unsupported/doc/examples/MatrixSquareRoot.o \
./Eigen/unsupported/doc/examples/PolynomialSolver1.o \
./Eigen/unsupported/doc/examples/PolynomialUtils1.o 

CPP_DEPS += \
./Eigen/unsupported/doc/examples/BVH_Example.d \
./Eigen/unsupported/doc/examples/FFT.d \
./Eigen/unsupported/doc/examples/MatrixExponential.d \
./Eigen/unsupported/doc/examples/MatrixFunction.d \
./Eigen/unsupported/doc/examples/MatrixLogarithm.d \
./Eigen/unsupported/doc/examples/MatrixPower.d \
./Eigen/unsupported/doc/examples/MatrixPower_optimal.d \
./Eigen/unsupported/doc/examples/MatrixSine.d \
./Eigen/unsupported/doc/examples/MatrixSinh.d \
./Eigen/unsupported/doc/examples/MatrixSquareRoot.d \
./Eigen/unsupported/doc/examples/PolynomialSolver1.d \
./Eigen/unsupported/doc/examples/PolynomialUtils1.d 


# Each subdirectory must supply rules for building sources it contributes
Eigen/unsupported/doc/examples/%.o: ../Eigen/unsupported/doc/examples/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/jason.b/Desktop/Github/NICE/Eigen -I/usr/include/ -I/usr/include/gtest -I/home/jason.b/Desktop/Github/NICE/cpp/include -O0 -g3 -Wall -c -fmessage-length=0 -std=c++14 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


