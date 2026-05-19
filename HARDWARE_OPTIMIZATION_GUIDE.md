# Hardware Optimization Guide: AMD Opteron 3280 + NVIDIA GTX 1050

## Target Hardware Specifications

### CPU: AMD Opteron 3280 (Bulldozer bdver1)
- **Microarchitecture**: Bulldozer (bdver1)
- **Cores**: 8 cores (4 modules with 2 integer cores each, shared FPU)
- **Threads**: 8 threads
- **Base Clock**: ~3.2 GHz
- **L1 Cache**: 64KB I-cache + 16KB D-cache per core
- **L2 Cache**: 2MB per module (shared between 2 integer cores)
- **L3 Cache**: 8MB shared
- **Memory**: DDR3-1600 (dual channel supported)

### Instruction Set Support
- **SUPPORTED**: 
  - SSE4.2
  - AVX (Advanced Vector Extensions)
  - AES-NI (Advanced Encryption Standard New Instructions)
  - POPCNT (Population Count)
  - CLMUL (Carry-Less Multiplication)
- **NOT SUPPORTED**:
  - AVX2
  - FMA (Fused Multiply-Add)
  - F16C (Half-Precision Floating-Point Conversion)
  - BMI2 (Bit Manipulation Instruction Set 2)
  - AVX512 family
  - SSE4a, XOP, FMA4 (AMD-specific extensions that Bulldozer lacks)

### GPU: NVIDIA GTX 1050
- **Architecture**: Pascal (GP107)
- **CUDA Cores**: 640
- **Base Clock**: 1354 MHz
- **Boost Clock**: 1455 MHz
- **Memory**: 2GB GDDR5
- **Memory Bus**: 128-bit
- **Memory Bandwidth**: ~112 GB/s
- **Compute Capability**: 6.1
- **TDP**: 75W

## CPU Optimization Strategies

### 1. SIMD Instruction Selection
Focus on instruction sets actually available on bdver1:

**Recommended Compiler Flags:**
```bash
-msse4.2 -mavx -maes -mpopcnt -mno-avx2 -mno-fma -mno-f16c -mno-bmi2 -mno-avx512f -mno-avx512cd -mno-avx512er -mno-avx512pf
```

**Alternative Approach:**
```bash
-march=native  # Let compiler detect bdver1 capabilities
```
*Note: Verify that -march=native doesn't enable unsupported features on your specific compiler version.*

### 2. Bulldozer-Specific Considerations

#### Module Architecture
- Each module has 2 integer cores sharing a single FPU
- FPU-intensive tasks may see contention between sibling cores
- **Optimization Strategy**: 
  - Schedule FPU-heavy workloads to avoid sibling core conflicts
  - Consider using only 1 core per module for FPU-bound tasks (4 threads max)
  - Integer-heavy tasks can utilize all 8 cores

#### Cache Optimization
- L1 D-cache is only 16KB (smaller than typical)
- L2 cache is 2MB per module (shared)
- **Optimization Strategy**:
  - Optimize for 16KB L1 D-cache blocking
  - Consider larger working sets that fit in 2MB L2 per module
  - Prefetching strategies important due to cache hierarchy

#### Memory Bandwidth
- DDR3-1600 dual channel provides ~25.6 GB/s theoretical bandwidth
- **Optimization Strategy**:
  - Optimize memory access patterns for sequential access
  - Consider cache blocking techniques
  - Use non-temporal stores for streaming data when appropriate

### 3. Specific Instruction Optimizations

#### SSE4.2
- Useful for string/text processing (though less relevant for ML)
- CRC32 instructions available
- **Application**: Hash functions, data integrity checks

#### AVX
- 256-bit vector operations
- **Key Applications**:
  - Matrix multiplication (vectorized dot products)
  - Activation functions (ReLU, sigmoid, tanh)
  - Quantization/dequantization operations
  - Memory initialization/copy operations

#### AES-NI
- Hardware-accelerated AES encryption/decryption
- **Application**: While not directly used in standard LLM inference, could be useful for:
  - Encrypted model storage
  - Secure communication protocols
  - Cryptographic hashing if needed

#### POPCNT
- Fast population count (number of set bits)
- **Applications**:
  - Binary neural networks
  - Sparsity computations
  - Certain quantization schemes
  - Hash table implementations

### 4. Code-Level Optimizations

#### Loop Vectorization
- Ensure loops are structured for AVX vectorization (multiples of 8 for float, 4 for double)
- Use `__builtin_assume_aligned` for aligned memory access
- Consider `#pragma ivdep` to ignore vector dependencies when safe

#### Memory Alignment
- Align data structures to 32-byte boundaries for AVX
- Use `alignas(32)` in C++ or `aligned_alloc`/`posix_memalign` in C
- Stack alignment: Ensure functions maintain 32-byte stack alignment

#### Instruction Scheduling
- Bulldozer has relatively long execution pipelines
- **Optimization**: 
  - Interleave independent instructions to hide latency
  - Balance instruction types across execution ports
  - Minimize dependency chains

## GPU Optimization Strategies (GTX 1050, Compute 6.1)

### 1. Architectural Considerations

#### Pascal Architecture (GP107)
- **SM Architecture**: 
  - 128 CUDA cores per SM (4 warp schedulers, 2 dispatch units)
  - GTX 1050: 5 SMs (640 cores / 128 = 5)
- **Memory Subsystem**:
  - 128-bit memory bus
  - 2MB L2 cache
  - 2GB GDDR5 VRAM
- **Compute Capability 6.1 Features**:
  - Unified memory support
  - Page faulting support
  - Improved atomic operations
  - Better performance for half-precision (though no native FP16 arithmetic)

### 2. Optimization Priorities for 2GB VRAM Constraint

#### Memory Management
- **Critical**: Optimize for minimal VRAM usage
- **Strategies**:
  - Model quantization (4-bit, 5-bit) to reduce memory footprint
  - Activation recomputation trade-off (compute vs memory)
  - KV cache optimization (sliding window, quantization)
  - Memory pooling to reduce allocation/free overhead

#### Kernel Launch Optimization
- **Block Size**: 
  - Optimal: multiples of 32 (warp size)
  - Common: 128, 256, 512 threads per block
  - Consider occupancy calculator for specific kernels
- **Grid Size**: 
  - Sufficient blocks to fill all SMs (aim for 10s-100s of blocks)
  - Balance between block count and shared memory usage

### 3. Specific Optimization Techniques

#### Memory Access Patterns
- **Coalesced Access**: 
  - Ensure consecutive threads access consecutive memory addresses
  - Critical for global memory performance
- **Shared Memory Usage**:
  - 64KB configurable shared memory per SM
  - Optimize for bank conflict avoidance
  - Consider using as user-managed L1 cache
- **Constant Memory**:
  - Small, frequently accessed data (64KB total)
  - Model metadata, configuration parameters

#### Instruction Efficiency
- **Warp Specialization**: 
  - Different warps within a block handling different tasks
  - Reduces synchronization overhead
- **Instruction Mix**:
  - Balance math instructions with memory operations
  - Avoid long latency chains
  - Use fast math operations where precision allows (`-use_fast_math`)

#### Precision Considerations
- **FP32 vs FP16**:
  - GTX 1050 has no native FP16 arithmetic (emulated)
  - FP32 often faster than FP16 on Pascal
  - Consider mixed precision: FP16 for storage, FP32 for computation
- **Tensor Cores**: 
  - Not available on Pascal (introduced in Volta)
  - Focus on optimizing CUDA cores instead

### 4. CUDA-Specific Build Flags

```bash
# For PTX compilation targeting compute 6.1
-arch=sm_61

# For binary compilation (if targeting specific GPU)
-code=sm_61

# Optimization flags
-O3 --use_fast_math  # If precision allows
--ptxas-options=-v   # Verbose PTX output for register usage
```

### 5. Memory-Specific Optimizations

#### Pinned Memory
- Use `cudaHostAlloc` or `cudaMallocHost` for CPU↔GPU transfer buffers
- Significantly improves transfer performance
- Critical for batch processing scenarios

#### Asynchronous Operations
- Use CUDA streams for overlap of computation and data transfer
- Consider double/triple buffering pipelines
- Event-based synchronization for fine-grained control

#### Memory Advice
- Use `cudaMemAdvise` for memory residency hints
- `cudaMemAdviseSetPreferredLocation` for GPU residency
- `cudaMemAdviseSetReadMostly` for read-heavy data

## Build System Modifications

### CMake Configuration Updates

#### For CPU Backend (ggml/CMakeLists.txt)
Modify the x86_64 section to reflect bdver1 capabilities:

```cmake
# x86_64 only
message(STATUS "x86 detected")
list(APPEND GGML_CPU_SOURCES
    ggml-cpu/arch/x86/quants.c
    ggml-cpu/arch/x86/repack.cpp
    )

# BDVER1-SPECIFIC OPTIMIZATIONS
if (GGML_NATIVE)
    # Check if native actually gives us bdver1 - otherwise be explicit
    list(APPEND ARCH_FLAGS -march=native)
else ()
    # Explicit bdver1 optimization - ONLY what's supported
    if (GGML_SSE42)
        list(APPEND ARCH_FLAGS -msse4.2)
        list(APPEND ARCH_DEFINITIONS GGML_SSE42)
    endif()
    if (GGML_AVX)
        list(APPEND ARCH_FLAGS -mavx)
        list(APPEND ARCH_DEFINITIONS GGML_AVX)
    endif()
    if (GGML_AES)
        list(APPEND ARCH_FLAGS -maes)
        list(APPEND ARCH_DEFINITIONS GGML_AES)
    endif()
    if (GGML_POPCNT)
        list(APPEND ARCH_FLAGS -mpopcnt)
        list(APPEND ARCH_DEFINITIONS GGML_POPCNT)
    endif()
    # EXPLICITLY DISABLE UNSUPPORTED FEATURES
    # These should ideally not be set, but being explicit helps
    # if (NOT GGML_AVX2)
    #     list(APPEND ARCH_FLAGS -mno-avx2)
    # endif()
    # ... similarly for FMA, F16C, BMI2, AVX512
endif()
```

#### For CUDA Backend
Ensure CUDA backend is configured for compute 6.1:

```cmake
# In ggml/CMakeLists.txt CUDA section
if (GGML_CUDA)
    # ... existing CUDA setup ...
    
    # PTX compilation for compute 6.1
    set(GGML_CUDA_ARCHITECTURES "61")
    
    # Or if using newer CMake CUDA support:
    # set_property(TARGET ggml-cuda PROPERTY CUDA_ARCHITECTURES 61)
endif()
```

## Validation and Benchmarking

### CPU Validation
1. **Correctness**: 
   - Run existing test suite
   - Verify numerical results match reference implementation
2. **Performance**:
   - Benchmark key operations (matmul, quantize, etc.)
   - Compare against baseline (-march=x86-64 without optimizations)
   - Measure scalability with thread count
3. **Hardware Verification**:
   - Use `cpuid` or `/proc/cpuinfo` to confirm instruction set usage
   - Consider using performance counters (perf, likwid) to measure:
     - IPC (Instructions Per Cycle)
     - Cache hit/miss rates
     - Memory bandwidth utilization

### GPU Validation
1. **Correctness**:
   - CUDA-specific tests
   - Numerical equivalence to CPU implementation
2. **Performance**:
   - Kernel timing (cudaEvents)
   - Occupancy calculation
   - Memory bandwidth utilization
   - Compare different block/grid configurations
3. **VRAM Usage**:
   - Monitor actual VRAM consumption
   - Test with various model sizes
   - Validate optimization effectiveness

## Risk Mitigation

### CPU Risks
- **Risk**: Over-optimization leading to incorrect code
  - **Mitigation**: Rigorous testing, incremental optimization
- **Risk**: Assuming bdver1 features that aren't present
  - **Mitigation**: Runtime CPU feature detection as fallback
- **Risk**: Memory bandwidth saturation
  - **Mitigation**: Optimize access patterns, consider NUMA effects

### GPU Risks
- **Risk**: Exceeding 2GB VRAM limit
  - **Mitigation**: Aggressive quantization, memory profiling
- **Risk**: Poor kernel occupancy
  - **Mitigation**: Use occupancy API, experiment with block sizes
- **Risk**: PCIe bottleneck (though less critical with 2GB card)
  - **Mitigation**: Overlap compute and transfer, pinned memory

## Recommended Implementation Approach

### Phase 1: Baseline Establishment
1. Confirm current performance on hardware
2. Establish baseline metrics for key operations
3. Verify all existing optimizations work correctly

### Phase 2: CPU Optimization
1. Implement bdver1-specific SIMD paths
2. Add runtime CPU detection
3. Benchmark and iterate
4. Verify correctness at each step

### Phase 3: GPU Optimization
1. Profile existing CUDA kernels
2. Implement memory optimizations
3. Optimize key kernels (matmul, attention, etc.)
4. Validate VRAM usage improvements

### Phase 4: Integrated Optimization
1. Optimize CPU-GPU data transfer
2. Balance workload between CPU and GPU
3. End-to-end benchmarking
4. Final validation and documentation

## References

1. AMD Bulldozer Microarchitecture Documentation
2. NVIDIA Pascal Architecture Whitepaper
3. AMD64 Architecture Programmer's Manual Volumes 1-5
4. CUDA C Programming Guide
5. Intel® 64 and IA-32 Architectures Optimization Reference Manual
6. "Optimizing Software for x86 Processors" by Agner Fog