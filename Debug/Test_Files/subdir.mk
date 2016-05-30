################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Test_Files/EigenTest.cpp \
../Test_Files/Gtest.cpp 

OBJS += \
./Test_Files/EigenTest.o \
./Test_Files/Gtest.o 

CPP_DEPS += \
./Test_Files/EigenTest.d \
./Test_Files/Gtest.d 


# Each subdirectory must supply rules for building sources it contributes
Test_Files/%.o: ../Test_Files/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -IEigen -I/home/jason.b/Desktop/Github/NICE/ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


