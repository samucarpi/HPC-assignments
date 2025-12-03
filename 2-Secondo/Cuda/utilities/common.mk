
# ============================================================================
# PARAMETRI DI CONFIGURAZIONE
# ============================================================================
EXERCISE=cholesky.cu


# ============================================================================
# INIZIALIZZAZIONE  
# ============================================================================
# Path CUDA
CUDA_HOME:=/usr/local/cuda

# Compilatori
NVCC:=$(CUDA_HOME)/bin/nvcc   # CUDA -> .cu
CXX:=g++                      # C++ -> .cpp e .c (non puro)
CC:=gcc                       # C -> .c puri

# ============================================================================
# FLAG DI COMPILAZIONE
# ============================================================================
# Flag di ottimizzazione
OPT:=-O2 -g
# Flag per la compilazione C/C++
# $(FLAGS): flag aggiuntivi passati da linea comando
CXXFLAGS:=$(OPT) -I. -I$(UTIL_DIR) $(FLAGS)
# Flag per NVCC (compilatore CUDA)
NVOPT:=-Xcompiler -fopenmp -lineinfo `pkg-config --cflags --libs opencv4`
# Unione flags
NVCFLAGS:=$(CXXFLAGS) $(NVOPT)

# ============================================================================
# FLAG DI LINKING
# ============================================================================
# Flag di linking base
LDFLAGS:=-lm -lcudart $(EXT_LDFLAGS)
# Flag di linking per NVCC
NVLDFLAGS:=$(LDFLAGS) -lgomp

# ============================================================================
# SORGENTI E OGGETTI
# ============================================================================
# Directory per i file oggetto compilati
BUILD_DIR ?= ./build
# Lista oggetti 
OBJS := $(BUILD_DIR)/utils.c.o $(BUILD_DIR)/polybench.c.o $(BUILD_DIR)/$(EXERCISE).o
# Nome dell'eseguibile finale (rimuove .cu e aggiunge .exe)
EXE=$(EXERCISE:.cu=.exe)

# ============================================================================
# REGOLE DI BUILD
# ============================================================================
# Comando per creare directory nel caso non esista
MKDIR_P ?= mkdir -p
# Crea l'eseguibile linkando tutti gli oggetti
$(EXE):	$(OBJS)
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) $(OBJS) -o $@ $(NVLDFLAGS)

# Compila file .cu in .cu.o usando nvcc
$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) -c $< -o $@

# Compila file .cpp in .cpp.o usando g++
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

# Compila file .c locali in .c.o usando g++
$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compila file .c da UTIL_DIR in .c.o usando g++
$(BUILD_DIR)/%.c.o: $(UTIL_DIR)/%.c
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ============================================================================
# TARGETS
# ============================================================================
all: $(EXE)

.PHONY: run profile clean

# Esegue l'eseguibile
run: $(EXE)
	./$(EXE)

# Profila con nvprof (NVIDIA profiler)
profile: $(EXE)
	sudo $(CUDA_HOME)/bin/nvprof --unified-memory-profiling off ./$(EXE)

# Profila con metriche GPU
metrics: $(EXE)
	sudo $(CUDA_HOME)/bin/nvprof --print-gpu-trace --metrics "eligible_warps_per_cycle,achieved_occupancy,sm_efficiency,ipc" ./$(EXE)

# Rimuove file generati dalla compilazione
clean:
	-rm -fr $(BUILD_DIR) *.exe *.out *~
