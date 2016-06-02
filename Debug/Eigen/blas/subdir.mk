################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Eigen/blas/complex_double.cpp \
../Eigen/blas/complex_single.cpp \
../Eigen/blas/double.cpp \
../Eigen/blas/single.cpp \
../Eigen/blas/xerbla.cpp 

OBJS += \
./Eigen/blas/complex_double.o \
./Eigen/blas/complex_single.o \
./Eigen/blas/double.o \
./Eigen/blas/single.o \
./Eigen/blas/xerbla.o 

CPP_DEPS += \
./Eigen/blas/complex_double.d \
./Eigen/blas/complex_single.d \
./Eigen/blas/double.d \
./Eigen/blas/single.d \
./Eigen/blas/xerbla.d 


# Each subdirectory must supply rules for building sources it contributes
Eigen/blas/%.o: ../Eigen/blas/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/jason.b/Desktop/Github/NICE/Eigen -I/usr/include/ -I/usr/include/gtest -I/home/jason.b/Desktop/Github/NICE/cpp/include -O0 -g3 -Wall -c -fmessage-length=0 -std=c++14 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


