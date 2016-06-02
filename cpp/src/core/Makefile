# Generic Makefile for a program compiled with gtest

# This flag denotes which compiler is used
CC = g++ -std=c++14

# Includes: must include pthread library and the path to the library includes
INCLUDES = -I/usr/include -I/home/jason.b/Desktop/Github/NICE/cpp/src/ -I/home/jason.b/Desktop/Github/NICE/Eigen -pthread

# Dependencies: this variable is used to make the Makefile more general
DEPS = cpu_operations.h /usr/lib/libgtest.a

# Gtest: for compilation gtest has to appear at the end of the script
GTEST = -lgtest

# Target is called executable
TARGET = executable

# Source refers to the .cpp file you are trying to compile
SRC = TransposeTestClass.cc cpu_operations.cc

# Objects: refers to the object file generated
OBJS = $(SRC:.cc=.o)



$(TARGET): $(OBJS) $(DEPS)
	$(CC) $(INCLUDES) $(DEPS) $(OBJS) -o $(TARGET) $(GTEST)

$(OBJS): $(SRC) $(DEPS)
	$(CC) -c $(DEPS) $(INCLUDES) $(SRC)

clean:
	rm *.o *.gch $(TARGET)
