################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../cpp/src/core/cpu_operations.cc \
../cpp/src/core/util.cc 

OBJS += \
./cpp/src/core/cpu_operations.o \
./cpp/src/core/util.o 

CC_DEPS += \
./cpp/src/core/cpu_operations.d \
./cpp/src/core/util.d 


# Each subdirectory must supply rules for building sources it contributes
cpp/src/core/%.o: ../cpp/src/core/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/include/ -I/home/jason.b/Desktop/Github/NICE/Eigen -I/usr/include/gtest -I/home/jason.b/Desktop/Github/NICE/cpp/src/ -I/home/jason.b/Desktop/Github/NICE/ -O0 -g3 -Wall -c -fmessage-length=0 -std=c++14 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


