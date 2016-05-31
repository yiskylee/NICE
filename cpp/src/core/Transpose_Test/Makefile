# Generic Makefile for a program compiled with gtest

# This flag denotes which compiler is used
CC = g++

# Includes: must include pthread library and the path to the library includes
INCLUDES = -I/usr/include -I/home/jason.b/Desktop/Github/NICE/cpp/src/ -I/home/jason.b/Desktop/Github/NICE/Eigen -pthread

# Dependencies: this variable is used to make the Makefile more general
DEPS = cpu_operations.h /usr/lib/libgtest.a

# Target is called executable
TARGET = executable

# Source refers to the .cpp file you are trying to compile
SRC = TransposeTest.cc cpu_operations.cc

# Objects: refers to the object file generated
OBJS = $(SRC:.cc=.o)



$(TARGET): $(OBJS) $(DEPS)
	$(CC) $(INCLUDES) $(DEPS) $(OBJS) -o $(TARGET)

$(OBJS): $(SRC) $(DEPS)
	$(CC) -c $(DEPS) $(INCLUDES) $(SRC)

clean:
	rm *.o $(TARGET)
