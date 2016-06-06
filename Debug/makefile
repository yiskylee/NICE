################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include googletest/xcode/Samples/FrameworkSample/subdir.mk
-include googletest/test/subdir.mk
-include googletest/src/subdir.mk
-include googletest/samples/subdir.mk
-include googletest/codegear/subdir.mk
-include cpp/test/cpu_operations_test/subdir.mk
-include cpp/src/util/subdir.mk
-include cpp/src/core/subdir.mk
-include cpp/build/CMakeFiles/nice_lib.dir/src/util/subdir.mk
-include cpp/build/CMakeFiles/nice_lib.dir/src/core/subdir.mk
-include cpp/build/CMakeFiles/2.8.12.2/CompilerIdCXX/subdir.mk
-include cpp/build/CMakeFiles/2.8.12.2/CompilerIdC/subdir.mk
-include Eigen/unsupported/test/subdir.mk
-include Eigen/unsupported/doc/examples/subdir.mk
-include Eigen/unsupported/bench/subdir.mk
-include Eigen/test/eigen2/subdir.mk
-include Eigen/test/subdir.mk
-include Eigen/scripts/subdir.mk
-include Eigen/lapack/subdir.mk
-include Eigen/failtest/subdir.mk
-include Eigen/doc/special_examples/subdir.mk
-include Eigen/doc/snippets/subdir.mk
-include Eigen/doc/examples/subdir.mk
-include Eigen/doc/subdir.mk
-include Eigen/demos/opengl/subdir.mk
-include Eigen/demos/mix_eigen_and_c/subdir.mk
-include Eigen/demos/mandelbrot/subdir.mk
-include Eigen/blas/subdir.mk
-include Eigen/bench/spbench/subdir.mk
-include Eigen/bench/btl/libs/ublas/subdir.mk
-include Eigen/bench/btl/libs/tvmet/subdir.mk
-include Eigen/bench/btl/libs/mtl4/subdir.mk
-include Eigen/bench/btl/libs/gmm/subdir.mk
-include Eigen/bench/btl/libs/eigen3/subdir.mk
-include Eigen/bench/btl/libs/eigen2/subdir.mk
-include Eigen/bench/btl/libs/blitz/subdir.mk
-include Eigen/bench/btl/libs/STL/subdir.mk
-include Eigen/bench/btl/libs/BLAS/subdir.mk
-include Eigen/bench/btl/data/subdir.mk
-include Eigen/bench/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: NICE

# Tool invocations
NICE: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ -L/usr/include/Eigen -o "NICE" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C++_DEPS)$(C_DEPS)$(CC_DEPS)$(CPP_DEPS)$(EXECUTABLES)$(CXX_DEPS)$(C_UPPER_DEPS) NICE
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
