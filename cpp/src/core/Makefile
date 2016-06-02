# Generic Makefile for a program compiled with gtest

# This flag denotes which compiler is used
CC = g++ -std=c++11

# Includes: must include pthread library and the path to the library includes
# Also includes the library libgtest.a using -L(path) -l(name) where -l is the 
# Gtest flag included at the end of the command
INCLUDES = -I/home/jason.b/Desktop/Github/NICE/cpp/include/ -I/home/jason.b/Desktop/Github/NICE/Eigen/ -I/home/jason.b/Desktop/Github/NICE/googletest/include/ -pthread -L/home/jason.b/Desktop/Github/NICE/cpp/include/

# Gtest: for compilation gtest has to appear at the end of the script
GTEST = -lgtest

# Target is called executable
TARGET = executable

# Source refers to the .cpp file you are trying to compile
SRC = cpu_operations.cc TransposeTestClass.cc

# Objects: refers to the object file generated
OBJS = $(SRC:.cc=.o)



$(TARGET): $(OBJS) $(DEPS)
	$(CC) $(INCLUDES) $(OBJS) -o $(TARGET) $(GTEST)

$(OBJS): $(SRC) $(DEPS)
	$(CC) -c $(INCLUDES) $(SRC)

clean:
	rm *.o *.gch $(TARGET)
