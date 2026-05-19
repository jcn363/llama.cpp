# Rust Migration Plan: llama.cpp → llama.rs (Hardware-Optimized for AMD Opteron 3280 + GTX 1050)

## Executive Summary

Convert the ~285K lines of C/C++ llama.cpp codebase to a Rust-only project, optimized specifically for AMD Opteron 3280 (bdver1) CPU and NVIDIA GTX 1050 GPU. This is a multi-phase, multi-month effort that should be done incrementally while maintaining a working product at each phase.

**Current Hardware:**
- CPU: AMD Opteron 3280 (Bulldozer bdver1, 8 cores, 32GB RAM)
  - Supports: SSE4.2, AVX, AES, POPCNT
  - Does NOT support: AVX2, FMA, F16C, BMI2, AVX512
- GPU: NVIDIA GTX 1050 (2GB VRAM, CUDA 12.0, compute capability 6.1, Pascal architecture)
- OS: AMD64 Linux only

**Current scope:**
- 503 C/C++ source files (excluding vendor)
- 189 Python files (conversion scripts)
- Core: ggml tensor library + llama inference engine
- Backends: CPU (optimized for bdver1), CUDA (optimized for compute 6.1)
- Tools: server, CLI, bench, quantize, perplexity, etc.
- Common: arg parsing, chat, sampling, unicode, jinja, PEG parser

## Architecture Overview

[Same as original plan - omitted for brevity]

## Phase 1: Foundation (Weeks 1-4)

### 1.1 Project Setup
- Create workspace Cargo.toml with all crates
- Set up CI/CD (GitHub Actions for Linux AMD64)
- Configure clippy, rustfmt, cargo-deny
- Set up benchmarking infrastructure (criterion)

### 1.2 GGUF Format Library
**Priority: P0** — Everything depends on reading GGUF files.
- Port `gguf-py/gguf/` to Rust
- Implement GGUF reader/writer
- Support all GGUF types and metadata
- Add tests against existing GGUF files

**Dependencies:** None
**Estimated effort:** 1 week
**Key Rust crates:** `memmap2`, `byteorder`, `serde`

### 1.3 Core Types and Tensor Library (ggml-rs skeleton)
- Define `Tensor` struct with dtype, shape, data
- Implement basic tensor operations (add, mul, matmul)
- Set up computation graph infrastructure
- Memory allocator (arena-based like ggml-alloc)

**Dependencies:** gguf crate
**Estimated effort:** 2 weeks
**Key Rust crates:** `ndarray` or custom, `half` (f16/bf16)

## Phase 2: CPU Backend Optimized for AMD Opteron 3280 (Weeks 5-10)

### 2.1 CPU Backend Core
- Port `ggml-cpu/` to Rust
- Implement SIMD operations using `std::simd` (portable_simd)
- **Hardware-specific optimizations for bdver1:**
  - Explicit SSE4.2 and AVX intrinsics using `std::arch::x86_64`
  - AES-NI and POPCNT optimizations where applicable
  - **Explicitly exclude:** AVX2, FMA, F16C, BMI2, AVX512 code paths (not supported)
  - Runtime CPU feature detection to enable/disable optimizations
  - Fallback to scalar operations for unsupported instructions
- Benchmark against C implementation on actual hardware

**Key considerations:**
- Bulldozer architecture has unique characteristics (shared FPUs, etc.)
- Memory bandwidth optimization is critical
- Use `std::simd` for portable SSE4.2/AVX, `std::arch::x86_64` for explicit intrinsics
- Consider cache line optimization (64-byte lines typical)

### 2.2 Quantization Functions
- Port `ggml-quants.c` to Rust
- Support all quantization types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, etc.
- **SIMD-optimized dequantization for bdver1:**
  - Focus on SSE4.2/AVX optimizations
  - Skip AVX2/FMA optimizations (not available)
  - AES-NI optimizations where applicable for certain quant types
- **Estimated effort:** 3 weeks
**Key Rust crates:** `std::simd`, `bytemuck`

### 2.3 Backend Infrastructure
- Backend registry and discovery
- Buffer management
- Graph execution engine
- Memory management (ggml-alloc equivalent)
- **Hardware-aware memory allocation:**
  - Consider NUMA awareness (though Opteron 3280 may not benefit much)
  - Page size optimization for huge pages if beneficial
**Estimated effort:** 2 weeks

## Phase 3: CUDA Backend Optimized for GTX 1050 (Weeks 11-16)

### 3.1 CUDA Bindings
- Use `bindgen` for CUDA runtime API
- Use `cudarc` or custom bindings for cuBLAS
- **Target compute capability 6.1 (Pascal):**
  - Explicitly target sm_61 for PTX compilation
  - No need for newer architecture features (Volta/Turing/Ampere)
  - Keep compatibility with CUDA 12.0

### 3.2 CUDA Operations
- Port `ggml-cuda/` kernels to Rust
- Use `rustacuda` or `cudarc` for kernel launches
- **GTX 1050-specific optimizations:**
  - 2GB VRAM constraint - optimize for memory usage
  - 640 CUDA cores, 768MHz base clock - balance compute vs memory
  - L1 cache configuration (48KB or 16KB configurable)
  - Shared memory per block: 48KB
  - Registers per block: 64K
  - Warp size: 32 threads
- Implement matrix multiplication, attention, etc. optimized for Pascal
- Consider using CUDA math API for intrinsics where beneficial

### 3.3 CUDA Integration
- Backend registration
- Memory transfer (host ↔ device)
- Graph execution with CUDA
- **Memory management considerations:**
  - Pinned memory for faster host-device transfers
  - Asynchronous transfers where possible
  - Stream concurrency for overlap
**Estimated effort:** 6 weeks
**Key Rust crates:** `cudarc`, `rustacuda`, `bindgen`

## Phase 4: llama-rs Core (Weeks 17-26)

[Same as original plan - omitted for brevity]

## Phase 5: Common Utilities (Weeks 27-30)
[Same as original plan - omitted for brevity]

## Phase 6: Tools (Weeks 31-36)
[Same as original plan - omitted for brevity]

## Phase 7: Other Backends (Weeks 37-44)
[Same as original plan - omitted for brevity]

## Phase 8: Testing, Optimization, Documentation (Weeks 45-52)

### 8.1 Testing
- Unit tests for all components
- Integration tests against GGUF models
- Fuzz testing for parsers
- Performance regression tests

### 8.2 Optimization
- Profile and optimize hot paths
- **SIMD tuning for bdver1:**
  - Focus on SSE4.2/AVX optimization opportunities
  - Memory access pattern optimization
  - Instruction scheduling for Bulldozer pipeline
- Memory allocation optimization
- Benchmark against C implementation on actual hardware
- Power efficiency considerations (important for server use cases)

### 8.3 Documentation
- API documentation (rustdoc)
- User guides
- Migration guide from llama.cpp
- Architecture documentation
- **Hardware-specific documentation:**
  - Build instructions for AMD Opteron 3280
  - CUDA setup for GTX 1050
  - Performance tuning guidelines

### 8.4 Packaging
- crates.io publishing
- Binary releases
- Docker images
- **Hardware-specific binaries:**
  - Consider providing bdver1-optimized binaries
  - Generic x86-64 binaries with runtime detection

## Key Technical Decisions (Hardware-Focused)

### Memory Management
- **Arena allocator** (like ggml) for tensor data
- **Rust ownership** for lifetime management
- **Custom allocators** for GPU memory
- Consider huge pages for large models if beneficial

### SIMD Strategy
- `std::simd` (portable_simd) for portable SSE4.2/AVX
- `std::arch::x86_64` for explicit intrinsics when needed (AES, POPCNT, etc.)
- **Feature detection at runtime** for CPU capabilities (critical for bdver1 vs other CPUs)
- Explicitly disable AVX2/FMA/F16C/BMI2/AVX512 code paths

### GPU Strategy
- **CUDA:** `cudarc` for safe CUDA bindings
- Target compute capability 6.1 specifically (no need for broader compatibility)
- Optimize for GTX 1050's 2GB VRAM limitation
- Consider memory pooling to reduce allocation overhead

### Concurrency
- `rayon` for CPU parallelism (tune for 8-core bdver1)
- `tokio` for async I/O (server)
- Thread pools for backend execution
- **Consider Bulldozer module sharing** when scheduling threads

### Error Handling
- `thiserror` for error types
- `anyhow` for application-level errors
- Result types throughout

## Risk Assessment (Updated for Hardware Focus)

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance regression vs C | High | Benchmark at each phase on actual hardware, optimize hot paths for bdver1/GTX 1050 |
| CUDA kernel complexity | High | Use existing PTX shaders initially, port gradually focusing on compute 6.1 |
| Model compatibility | High | Test against all supported GGUF models on actual hardware |
| Hardware-specific optimization complexity | Medium | Start with portable SIMD, add explicit intrinsics gradually |
| Scope creep | Medium | Strict phase boundaries, MVP first (CPU + CUDA for target hardware) |
| Team capacity | Medium | Focus on CPU + CUDA first, defer other backends |
| VRAM limitations (2GB GTX 1050) | Medium | Optimize memory usage, support model offloading if needed |

## Success Criteria

1. **Functional parity:** All llama.cpp features work in Rust
2. **Performance parity:** Within 10% of C implementation on **actual AMD Opteron 3280 + GTX 1050 hardware**
3. **Model compatibility:** All GGUF models load and run correctly
4. **API compatibility:** Drop-in replacement for llama.cpp C API (via FFI if needed)
5. **Build simplicity:** `cargo build --release` works out of the box
6. **Hardware optimization:** Binary is explicitly tuned for bdver1 and compute 6.1

## Immediate Next Steps

1. Create workspace structure with `cargo init`
2. Implement GGUF reader/writer (Phase 1.2)
3. Set up CI/CD pipeline (testing on actual hardware if possible)
4. Begin CPU backend skeleton with bdver1-specific considerations (Phase 2)
5. Set up CUDA development environment for GTX 1050 (Phase 3 preparation)

## Hardware-Specific Build Recommendations

For development and deployment:
- **CPU flags:** `-march=native` or `-msse4.2 -mavx -maes -mpopcnt` (explicitly avoid `-mavx2`, `-mfma`, etc.)
- **Rust flags:** `-C target-cpu=native` or `-C target-cpu=bdver1` (if supported)
- **CUDA flags:** `-arch=sm_61` for PTX compilation
- Consider LTO (Link Time Optimization) for final builds