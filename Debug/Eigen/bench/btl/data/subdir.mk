################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CXX_SRCS += \
../Eigen/bench/btl/data/mean.cxx \
../Eigen/bench/btl/data/regularize.cxx \
../Eigen/bench/btl/data/smooth.cxx 

OBJS += \
./Eigen/bench/btl/data/mean.o \
./Eigen/bench/btl/data/regularize.o \
./Eigen/bench/btl/data/smooth.o 

CXX_DEPS += \
./Eigen/bench/btl/data/mean.d \
./Eigen/bench/btl/data/regularize.d \
./Eigen/bench/btl/data/smooth.d 


# Each subdirectory must supply rules for building sources it contributes
Eigen/bench/btl/data/%.o: ../Eigen/bench/btl/data/%.cxx
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/include/ -I/home/jason.b/Desktop/Github/NICE/Eigen -I/usr/include/gtest -I/home/jason.b/Desktop/Github/NICE/cpp/src/ -I/home/jason.b/Desktop/Github/NICE/ -O0 -g3 -Wall -c -fmessage-length=0 -std=c++14 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


