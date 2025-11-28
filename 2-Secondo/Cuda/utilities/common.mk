ifndef CUDA_HOME
CUDA_HOME:=/usr/local/cuda
endif

ifndef EXERCISE
EXERCISE=cholesky.cu
endif

BUILD_DIR ?= ./build

NVCC=$(CUDA_HOME)/bin/nvcc
CXX=g++
CC=gcc

OPT:=-O2 -g
NVOPT:=-Xcompiler -fopenmp -lineinfo `pkg-config --cflags --libs opencv4`

CXXFLAGS:=$(OPT) -I. -I$(UTIL_DIR) $(EXT_CXXFLAGS) 
LDFLAGS:=-lm -lcudart $(EXT_LDFLAGS)

NVCFLAGS:=$(CXXFLAGS) $(NVOPT)
NVLDFLAGS:=$(LDFLAGS) -lgomp

CXX_SRCS:= utils.c
OBJS := $(CXX_SRCS:%=$(BUILD_DIR)/%.o) $(BUILD_DIR)/polybench.c.o $(EXERCISE:%=$(BUILD_DIR)/%.o)
EXE=$(EXERCISE:.cu=.exe)

$(EXE):	$(OBJS)
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) $(OBJS) -o $@ $(NVLDFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# utils.c ha sintassi C++ (extern "C") quindi usa g++
$(BUILD_DIR)/utils.c.o: utils.c
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# polybench.c è C puro, usa gcc
$(BUILD_DIR)/polybench.c.o: $(UTIL_DIR)/polybench.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CXXFLAGS) -c $< -o $@

all: $(EXE)

.PHONY: run profile clean
run: $(EXE)
	./$(EXE)

profile: $(EXE)
	sudo $(CUDA_HOME)/bin/nvprof --unified-memory-profiling off ./$(EXE)

metrics: $(EXE)
	sudo $(CUDA_HOME)/bin/nvprof --print-gpu-trace --metrics "eligible_warps_per_cycle,achieved_occupancy,sm_efficiency,ipc" ./$(EXE)

clean:
	-rm -fr $(BUILD_DIR) *.exe *.out *~

MKDIR_P ?= mkdir -p

