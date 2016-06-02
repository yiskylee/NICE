################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Eigen/doc/special_examples/Tutorial_sparse_example.cpp \
../Eigen/doc/special_examples/Tutorial_sparse_example_details.cpp 

OBJS += \
./Eigen/doc/special_examples/Tutorial_sparse_example.o \
./Eigen/doc/special_examples/Tutorial_sparse_example_details.o 

CPP_DEPS += \
./Eigen/doc/special_examples/Tutorial_sparse_example.d \
./Eigen/doc/special_examples/Tutorial_sparse_example_details.d 


# Each subdirectory must supply rules for building sources it contributes
Eigen/doc/special_examples/%.o: ../Eigen/doc/special_examples/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/jason.b/Desktop/Github/NICE/Eigen -I/usr/include/ -I/usr/include/gtest -I/home/jason.b/Desktop/Github/NICE/cpp/include -O0 -g3 -Wall -c -fmessage-length=0 -std=c++14 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


