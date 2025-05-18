TARGET = Convolution_OpenMP
SRC = Convolution_OpenMP.c
CC = gcc
CFLAGS = -Wall -std=c99 -O2 -fopenmp

# Build rule for the target executable
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean target
clean:
	del /f $(TARGET).exe
