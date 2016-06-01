################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Eigen/bench/btl/libs/eigen3/btl_tiny_eigen3.cpp \
../Eigen/bench/btl/libs/eigen3/main_adv.cpp \
../Eigen/bench/btl/libs/eigen3/main_linear.cpp \
../Eigen/bench/btl/libs/eigen3/main_matmat.cpp \
../Eigen/bench/btl/libs/eigen3/main_vecmat.cpp 

OBJS += \
./Eigen/bench/btl/libs/eigen3/btl_tiny_eigen3.o \
./Eigen/bench/btl/libs/eigen3/main_adv.o \
./Eigen/bench/btl/libs/eigen3/main_linear.o \
./Eigen/bench/btl/libs/eigen3/main_matmat.o \
./Eigen/bench/btl/libs/eigen3/main_vecmat.o 

CPP_DEPS += \
./Eigen/bench/btl/libs/eigen3/btl_tiny_eigen3.d \
./Eigen/bench/btl/libs/eigen3/main_adv.d \
./Eigen/bench/btl/libs/eigen3/main_linear.d \
./Eigen/bench/btl/libs/eigen3/main_matmat.d \
./Eigen/bench/btl/libs/eigen3/main_vecmat.d 


# Each subdirectory must supply rules for building sources it contributes
Eigen/bench/btl/libs/eigen3/%.o: ../Eigen/bench/btl/libs/eigen3/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/include/ -I/home/jason.b/Desktop/Github/NICE/Eigen -I/usr/include/gtest -I/home/jason.b/Desktop/Github/NICE/cpp/src/ -I/home/jason.b/Desktop/Github/NICE/ -O0 -g3 -Wall -c -fmessage-length=0 -std=c++14 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


