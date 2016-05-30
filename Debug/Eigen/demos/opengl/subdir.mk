################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Eigen/demos/opengl/camera.cpp \
../Eigen/demos/opengl/gpuhelper.cpp \
../Eigen/demos/opengl/icosphere.cpp \
../Eigen/demos/opengl/quaternion_demo.cpp \
../Eigen/demos/opengl/trackball.cpp 

OBJS += \
./Eigen/demos/opengl/camera.o \
./Eigen/demos/opengl/gpuhelper.o \
./Eigen/demos/opengl/icosphere.o \
./Eigen/demos/opengl/quaternion_demo.o \
./Eigen/demos/opengl/trackball.o 

CPP_DEPS += \
./Eigen/demos/opengl/camera.d \
./Eigen/demos/opengl/gpuhelper.d \
./Eigen/demos/opengl/icosphere.d \
./Eigen/demos/opengl/quaternion_demo.d \
./Eigen/demos/opengl/trackball.d 


# Each subdirectory must supply rules for building sources it contributes
Eigen/demos/opengl/%.o: ../Eigen/demos/opengl/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -IEigen -I/home/jason.b/Desktop/Github/NICE/ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


