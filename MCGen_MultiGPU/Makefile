# MCGen_MultiGPU build
#   make            -> MCGen_MultiGPU   (CUDA, multi-GPU + CPU backends)
#   make cpu        -> MCGen_CPU        (no CUDA needed, CPU backend only)
#   make CUDA_ARCH="-gencode arch=compute_89,code=sm_89 -gencode arch=compute_80,code=sm_80"
#
# CUDA_ARCH defaults to -arch=native (builds for the GPUs present in this machine).

CUDA_HOME ?= /usr/local/cuda-12.6
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= -arch=native
CXX       ?= g++
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo

SRC = MCGen_MultiGPU.cpp mc_host.cpp mc_cpu.cpp
HDR = mc_core.h mc_host.h mc_backend.h

.PHONY: gpu cpu all clean
gpu: MCGen_MultiGPU
cpu: MCGen_CPU
all: gpu cpu

mc_gpu.o: mc_gpu.cu $(HDR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -DMC_WITH_CUDA -c mc_gpu.cu -o $@

MCGen_MultiGPU: $(SRC) $(HDR) mc_gpu.o
	$(CXX) $(CXXFLAGS) -DMC_WITH_CUDA $(SRC) mc_gpu.o -o $@ -L$(CUDA_HOME)/lib64 -lcudart -pthread

MCGen_CPU: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@ -pthread

clean:
	rm -f *.o MCGen_MultiGPU MCGen_CPU
