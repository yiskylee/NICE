################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../cpp/src/core/TransposeTest.cc \
../cpp/src/core/TransposeTestClass.cc \
../cpp/src/core/cpu_operations.cc 

OBJS += \
./cpp/src/core/TransposeTest.o \
./cpp/src/core/TransposeTestClass.o \
./cpp/src/core/cpu_operations.o 

CC_DEPS += \
./cpp/src/core/TransposeTest.d \
./cpp/src/core/TransposeTestClass.d \
./cpp/src/core/cpu_operations.d 


# Each subdirectory must supply rules for building sources it contributes
cpp/src/core/%.o: ../cpp/src/core/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/jason.b/Desktop/Github/NICE/Eigen -I/usr/include/ -I/usr/include/gtest -I/home/jason.b/Desktop/Github/NICE/cpp/include -O0 -g3 -Wall -c -fmessage-length=0 -std=c++14 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


