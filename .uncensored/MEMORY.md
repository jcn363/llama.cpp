# MEMORY.md — Long-Term Knowledge

## Project Context
- **Project:** llama.cpp
- **Location:** /home/user/Desktop/llama.cpp
- **Initialized:** 2026-05-19

## Key Facts
- Hardware: AMD Opteron 3280 (bdver1), 8 cores, 32GB RAM
- GPU: NVIDIA GTX 1050, 2GB VRAM, CUDA 12.0, compute 6.1
- CPU supports: SSE4.2, AVX, AES, POPCNT
- CPU does NOT support: AVX2, FMA, F16C, BMI2, AVX512
- CUDA backend enabled for GPU acceleration

## Decisions
- Disabled AVX2/FMA/F16C/BMI2 in CMake defaults (not supported by bdver1)
- Enabled CUDA for GTX 1050 (compute capability 6.1)
- Kept -march=native for automatic CPU feature detection

## Learnings
- AMD Opteron 3280 is Bulldozer architecture (bdver1)
- Bulldozer has AVX but not AVX2 or FMA
- GTX 1050 has compute capability 6.1 (Pascal architecture)
