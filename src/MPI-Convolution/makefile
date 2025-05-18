# Compiler
CC = gcc

# MPI include and library paths
INCLUDE_PATH = "C:/Program Files (x86)/Microsoft SDKs/MPI/Include"
LIBRARY_PATH = "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64"

# Output executable
TARGET = MPI_Local_Test

# Compiler flags
CFLAGS = -I$(INCLUDE_PATH) -O2
LDFLAGS = -L$(LIBRARY_PATH) -lmsmpi

# Build target
$(TARGET): Convolution_MPI.c
	$(CC) Convolution_MPI.c $(CFLAGS) $(LDFLAGS) -o $(TARGET)

# Clean target
clean:
	del /f $(TARGET).exe
