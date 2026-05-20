# llama.rs — LLaMA inference in Rust

A Rust port of [llama.cpp](https://github.com/ggml-org/llama.cpp), optimized for **AMD Opteron 3280** (bdver1) + **NVIDIA GTX 1050** (2GB VRAM).

## Architecture

```
llama-rs/
├── crates/
│   ├── gguf/          # GGUF v3 parser (13 value types, 42 tensor types)
│   ├── ggml/          # Core tensor library and computation graphs
│   ├── ggml-cpu/      # CPU backend: AVX + SSE4.2 SIMD matmul
│   ├── ggml-cuda/     # CUDA backend: cuBLAS matmul (feature-gated)
│   ├── llama/         # Inference engine: transformer forward pass
│   ├── common/        # Shared utilities (sampling config, etc.)
│   ├── llama-cli/     # CLI binary for interactive generation
│   └── llama-server/  # HTTP server with /completion endpoint
├── .github/workflows/ # CI: build, test, clippy, release
└── Cargo.toml         # Workspace definition
```

## Hardware Target

| Component | Specs |
|-----------|-------|
| **CPU** | AMD Opteron 3280 (Bulldozer bdver1) — 8 cores, 32GB RAM |
| **SIMD** | SSE4.2, AVX (NO FMA, AVX2, AVX512) |
| **GPU** | NVIDIA GTX 1050 — 640 CUDA cores, 2GB VRAM, Compute 6.1 |

## Performance

| Operation | Size | Single Thread | Parallel (8 cores) | Speedup |
|-----------|------|---------------|-------------------|---------|
| Matmul | 64×64 | 86µs | 429µs | 0.2x (overhead) |
| Matmul | 128×128 | 637µs | 562µs | 1.1x |
| Matmul | 256×256 | 4.2ms | 1.7ms | 2.5x |
| Matmul | 512×512 | 36.5ms | 11.4ms | **3.2x** |
| Dot product | 4096 | 1.1µs | - | - |

## Quick Start

```bash
# Build
cargo build --release --workspace

# Run CLI
./target/release/llama-cli -m model.gguf -p "Hello, world!" -n 128

# Run server
./target/release/llama-server -m model.gguf --host 0.0.0.0 --port 8080

# Test
cargo test --workspace

# Benchmark
cargo bench -p ggml-cpu --bench cpu_bench
```

## Features

- **GGUF v3 parser**: Full support for 13 metadata types, 42 tensor types, memory-mapped I/O
- **SIMD matmul**: AVX 8-wide (32 floats/iter) → SSE4.2 4-wide (16 floats/iter) → scalar fallback
- **CUDA backend**: cuBLAS matmul, VRAM tracking, feature-gated (disabled by default)
- **Inference engine**: RMSNorm, RoPE, multi-head attention with GQA, SwiGLU FFN, KV cache
- **Sampling**: Greedy, temperature, top-k, top-p (nucleus)
- **CLI**: Interactive mode, single prompt, streaming token output
- **Server**: POST `/completion`, GET `/health`, JSON API

## Build Configuration

```toml
# .cargo/config.toml
[build]
rustflags = ["-C", "target-cpu=bdver1"]
```

CUDA is disabled by default. Enable with:
```bash
cargo build --release -p ggml-cuda --features cuda
```

## Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1.1 | ✅ | Workspace setup (8 crates) |
| 1.2 | ✅ | GGUF v3 parser |
| 2 | ✅ | SIMD matmul (AVX + SSE4.2) |
| 3 | ✅ | CUDA backend (cuBLAS) |
| 4 | ✅ | Inference engine (transformer) |
| 5 | ✅ | CLI and server binaries |
| 6 | ✅ | CI/CD pipeline |
| 7 | ✅ | Benchmarks |

**53 tests pass** across all crates.

## License

MIT (same as llama.cpp)
