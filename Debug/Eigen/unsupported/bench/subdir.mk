################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Eigen/unsupported/bench/bench_svd.cpp 

OBJS += \
./Eigen/unsupported/bench/bench_svd.o 

CPP_DEPS += \
./Eigen/unsupported/bench/bench_svd.d 


# Each subdirectory must supply rules for building sources it contributes
Eigen/unsupported/bench/%.o: ../Eigen/unsupported/bench/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -IEigen -I/home/jason.b/Desktop/Github/NICE/ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


