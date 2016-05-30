################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Eigen/bench/spbench/sp_solver.cpp \
../Eigen/bench/spbench/spbenchsolver.cpp \
../Eigen/bench/spbench/test_sparseLU.cpp 

OBJS += \
./Eigen/bench/spbench/sp_solver.o \
./Eigen/bench/spbench/spbenchsolver.o \
./Eigen/bench/spbench/test_sparseLU.o 

CPP_DEPS += \
./Eigen/bench/spbench/sp_solver.d \
./Eigen/bench/spbench/spbenchsolver.d \
./Eigen/bench/spbench/test_sparseLU.d 


# Each subdirectory must supply rules for building sources it contributes
Eigen/bench/spbench/%.o: ../Eigen/bench/spbench/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -IEigen -I/home/jason.b/Desktop/Github/NICE/ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


