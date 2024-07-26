# Makefile for building the rotating_cube project

# Compiler
CXX = g++

# Common source files
SOURCES = rotating_cube.cpp

# Common compiler flags
CXXFLAGS = -std=c++11

# Target executable
TARGET = rotating_cube

# Platform-specific settings
ifeq ($(shell uname), Darwin)
    # macOS settings
    HOMEBREW_PREFIX = $(shell brew --prefix)
    CXXFLAGS += -I$(HOMEBREW_PREFIX)/include/
    LDFLAGS = -L$(HOMEBREW_PREFIX)/lib/ -lSDL2 -lGLEW -framework OpenGL
else
    # Linux settings
    LDFLAGS = -lSDL2 -lGLEW -lGL
endif

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SOURCES)
	$(CXX) -o $(TARGET) $(SOURCES) $(CXXFLAGS) $(LDFLAGS)

# Clean target
clean:
	rm -f $(TARGET)

# Run the executable
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

