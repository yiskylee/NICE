################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Eigen/scripts/eigen_gen_credits.cpp 

OBJS += \
./Eigen/scripts/eigen_gen_credits.o 

CPP_DEPS += \
./Eigen/scripts/eigen_gen_credits.d 


# Each subdirectory must supply rules for building sources it contributes
Eigen/scripts/%.o: ../Eigen/scripts/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -IEigen -I/home/jason.b/Desktop/Github/NICE/ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


