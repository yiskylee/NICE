# Generic makefile for a program compiled with gtest

# This flag denotes which compiler is used
CC = g++

# Includes: must include pthread library and the path to the library includes
INCLUDES = -I/usr/include/ -pthread

# Dependencies: this variable is used to make the Makefile more general
DEPS = libgtest.a

# Target is called executable
TARGET = executable

# Source refers to the .cpp file you are trying to compile
SRC = Gtest.cpp

# Objects: refers to the object file generated
OBJS = $(SRC:.cpp=.o)



$(TARGET): $(OBJS) $(DEPS)
	$(CC) $(INCLUDES) $(OBJS) $(DEPS) -o $(TARGET)

$(OBJS): $(SRC) 
	$(CC) -c $(SRC)

clean:
	rm *.o $(TARGET)
