# Rust Migration Plan: llama.cpp → llama.rs

## Executive Summary

Convert the ~285K lines of C/C++ llama.cpp codebase to a Rust-only project. This is a multi-phase, multi-month effort that should be done incrementally while maintaining a working product at each phase.

**Current scope:**
- 503 C/C++ source files (excluding vendor)
- 189 Python files (conversion scripts)
- Core: ggml tensor library + llama inference engine
- Backends: CPU, CUDA, Vulkan, HIP, RPC, BLAS, OpenCL
- Tools: server, CLI, bench, quantize, perplexity, etc.
- Common: arg parsing, chat, sampling, unicode, jinja, PEG parser

## Architecture Overview

```
llama.rs/
├── Cargo.toml
├── crates/
│   ├── ggml/              # Core tensor library (replaces ggml/)
│   │   ├── src/
│   │   │   ├── lib.rs     # Public API
│   │   │   ├── tensor.rs  # Tensor types and operations
│   │   │   ├── graph.rs   # Computation graphs
│   │   │   ├── alloc.rs   # Memory allocation
│   │   │   ├── quant.rs   # Quantization types
│   │   │   ├── opt.rs     # Optimization (Adam, etc.)
│   │   │   └── backends/
│   │   │       ├── mod.rs
│   │   │       ├── cpu/   # CPU backend (SIMD)
│   │   │       ├── cuda/  # CUDA backend (bindgen + cuBLAS)
│   │   │       └── ...    # Other backends
│   │   └── Cargo.toml
│   ├── llama/             # Model inference (replaces src/)
│   │   ├── src/
│   │   │   ├── lib.rs     # Public API
│   │   │   ├── model.rs   # Model loading and metadata
│   │   │   ├── context.rs # Inference context
│   │   │   ├── memory.rs  # KV cache management
│   │   │   ├── batch.rs   # Batch processing
│   │   │   ├── sampler.rs # Token sampling
│   │   │   ├── grammar.rs # Grammar-constrained decoding
│   │   │   ├── vocab.rs   # Tokenizer (BPE, SPM, etc.)
│   │   │   ├── quant.rs   # Model quantization
│   │   │   ├── adapter.rs # LoRA, adapters
│   │   │   ├── loader.rs  # GGUF model loading
│   │   │   ├── saver.rs   # Model saving
│   │   │   └── models/    # Model-specific implementations
│   │   └── Cargo.toml
│   ├── common/            # Shared utilities (replaces common/)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── arg.rs     # CLI argument parsing (clap)
│   │   │   ├── chat.rs    # Chat handling
│   │   │   ├── sampling.rs# Sampling utilities
│   │   │   ├── unicode.rs # Unicode handling
│   │   │   ├── download.rs# Model downloading
│   │   │   ├── log.rs     # Logging
│   │   │   ├── jinja/     # Jinja template engine
│   │   │   └── peg/       # PEG grammar parser
│   │   └── Cargo.toml
│   └── tools/             # CLI tools (replaces tools/)
│       ├── src/
│       │   ├── server/    # HTTP server (axum)
│       │   ├── cli/       # CLI interface
│       │   ├── bench/     # Benchmarking
│       │   ├── perplexity/# Perplexity calculation
│       │   └── quantize/  # Model quantization
│       └── Cargo.toml
├── gguf/                  # GGUF format library (replaces gguf-py)
│   ├── src/
│   │   └── lib.rs
│   └── Cargo.toml
└── conversion/            # Model conversion (Python → Rust)
    ├── src/
    │   └── lib.rs
    └── Cargo.toml
```

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

## Phase 2: CPU Backend (Weeks 5-10)

### 2.1 CPU Backend Core
- Port `ggml-cpu/` to Rust
- Implement SIMD operations using `std::simd` (portable_simd)
- Support for current hardware: SSE4.2, AVX, AES, POPCNT
- No AVX2/FMA (AMD Opteron 3280 bdver1 limitation)

**Key considerations:**
- Use `std::arch::x86_64` for explicit SIMD intrinsics
- Fallback to scalar operations for unsupported instructions
- Benchmark against C implementation

### 2.2 Quantization Functions
- Port `ggml-quants.c` to Rust
- Support all quantization types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, etc.
- SIMD-optimized dequantization

**Estimated effort:** 3 weeks
**Key Rust crates:** `std::simd`, `bytemuck`

### 2.3 Backend Infrastructure
- Backend registry and discovery
- Buffer management
- Graph execution engine
- Memory management (ggml-alloc equivalent)

**Estimated effort:** 2 weeks

## Phase 3: CUDA Backend (Weeks 11-16)

### 3.1 CUDA Bindings
- Use `bindgen` for CUDA runtime API
- Use `cudarc` or custom bindings for cuBLAS
- Support compute capability 6.1+ (GTX 1050)

### 3.2 CUDA Operations
- Port `ggml-cuda/` kernels to Rust
- Use `rustacuda` or `cudarc` for kernel launches
- Implement matrix multiplication, attention, etc.

### 3.3 CUDA Integration
- Backend registration
- Memory transfer (host ↔ device)
- Graph execution with CUDA

**Estimated effort:** 6 weeks
**Key Rust crates:** `cudarc`, `rustacuda`, `bindgen`

## Phase 4: llama-rs Core (Weeks 17-26)

### 4.1 Model Loading
- Port `llama-model.cpp` and `llama-model-loader.cpp`
- GGUF model parsing
- Model architecture detection
- Weight loading and quantization

### 4.2 Inference Context
- Port `llama-context.cpp`
- Batch processing
- Graph building for forward pass
- Output processing

### 4.3 KV Cache / Memory
- Port `llama-memory.cpp` and variants
- KV cache management
- Hybrid/recurrent memory support
- ISWA (Infinite Sequence Window Attention)

### 4.4 Tokenizer / Vocab
- Port `llama-vocab.cpp`
- BPE tokenizer
- SPM (SentencePiece) tokenizer
- Unicode handling

### 4.5 Sampling
- Port `llama-sampler.cpp`
- Temperature, top-k, top-p, min-p
- Mirostat, typical sampling
- Grammar-constrained decoding

### 4.6 Model Implementations
- Port `src/models/` — model-specific forward passes
- Llama, Mistral, Gemma, Qwen, etc.
- Architecture-specific layers (RoPE, MoE, etc.)

**Estimated effort:** 10 weeks
**Dependencies:** ggml-rs (Phases 2-3)

## Phase 5: Common Utilities (Weeks 27-30)

### 5.1 CLI Argument Parsing
- Replace `common/arg.cpp` with `clap`
- Define all CLI arguments

### 5.2 Chat Handling
- Port `common/chat.cpp`
- Chat templates
- Message formatting

### 5.3 Jinja Template Engine
- Port `common/jinja/` to Rust
- Template parsing and execution
- Used for chat templates

### 5.4 PEG Parser
- Port `common/peg-parser.cpp`
- Or use existing Rust PEG parser (`pest`)
- Grammar-constrained decoding

### 5.5 Unicode Handling
- Port `common/unicode.cpp`
- Or use `unicode-segmentation`, `unicode-normalization`

**Estimated effort:** 4 weeks
**Key Rust crates:** `clap`, `pest`, `unicode-segmentation`

## Phase 6: Tools (Weeks 31-36)

### 6.1 HTTP Server
- Replace `tools/server/` with `axum`
- OpenAI-compatible API
- Streaming responses
- Model management endpoints

### 6.2 CLI Tools
- `llama-cli` → interactive chat
- `llama-bench` → benchmarking
- `llama-perplexity` → perplexity calculation
- `llama-quantize` → model quantization
- `llama-tokenize` → tokenization
- `llama-imatrix` → importance matrix

### 6.3 Model Conversion
- Port Python `convert_hf_to_gguf.py` to Rust
- Support all model architectures in `conversion/`
- Or keep Python scripts as external tooling

**Estimated effort:** 6 weeks
**Key Rust crates:** `axum`, `tokio`, `serde_json`, `tower`

## Phase 7: Other Backends (Weeks 37-44)

### 7.1 Vulkan Backend
- Port `ggml-vulkan/`
- Use `ash` (Vulkan bindings)
- Shader compilation

### 7.2 HIP/ROCm Backend
- Port `ggml-hip/`
- AMD GPU support

### 7.3 RPC Backend
- Port `ggml-rpc/`
- Remote inference

### 7.4 BLAS Backend
- Port `ggml-blas/`
- Use `blas-src` + `openblas-src`

**Estimated effort:** 8 weeks

## Phase 8: Testing, Optimization, Documentation (Weeks 45-52)

### 8.1 Testing
- Unit tests for all components
- Integration tests against GGUF models
- Fuzz testing for parsers
- Performance regression tests

### 8.2 Optimization
- Profile and optimize hot paths
- SIMD tuning for target hardware
- Memory allocation optimization
- Benchmark against C implementation

### 8.3 Documentation
- API documentation (rustdoc)
- User guides
- Migration guide from llama.cpp
- Architecture documentation

### 8.4 Packaging
- crates.io publishing
- Binary releases
- Docker images

**Estimated effort:** 8 weeks

## Key Technical Decisions

### Memory Management
- **Arena allocator** (like ggml) for tensor data
- **Rust ownership** for lifetime management
- **Custom allocators** for GPU memory

### SIMD Strategy
- `std::simd` (portable_simd) for portable SIMD
- `std::arch::x86_64` for explicit intrinsics when needed
- Feature detection at runtime for CPU capabilities

### GPU Strategy
- **CUDA:** `cudarc` for safe CUDA bindings
- **Vulkan:** `ash` for Vulkan bindings
- **HIP:** `hip-runtime-sys` + custom bindings

### Concurrency
- `rayon` for CPU parallelism
- `tokio` for async I/O (server)
- Thread pools for backend execution

### Error Handling
- `thiserror` for error types
- `anyhow` for application-level errors
- Result types throughout

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance regression vs C | High | Benchmark at each phase, optimize hot paths |
| CUDA kernel complexity | High | Use existing PTX shaders initially, port gradually |
| Model compatibility | High | Test against all supported GGUF models |
| Scope creep | Medium | Strict phase boundaries, MVP first |
| Team capacity | Medium | Focus on CPU + CUDA first, defer other backends |

## Success Criteria

1. **Functional parity:** All llama.cpp features work in Rust
2. **Performance parity:** Within 10% of C implementation on same hardware
3. **Model compatibility:** All GGUF models load and run correctly
4. **API compatibility:** Drop-in replacement for llama.cpp C API (via FFI if needed)
5. **Build simplicity:** `cargo build --release` works out of the box

## Immediate Next Steps

1. Create workspace structure with `cargo init`
2. Implement GGUF reader/writer (Phase 1.2)
3. Set up CI/CD pipeline
4. Begin CPU backend skeleton (Phase 2)
