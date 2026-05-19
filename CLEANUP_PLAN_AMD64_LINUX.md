# Cleanup Plan: AMD64 Linux Only Support

## Objective
Remove all non-AMD64 Linux platform support from the codebase to simplify maintenance and optimize for current hardware (AMD Opteron 3280 + NVIDIA GTX 1050).

## Current Status
Based on analysis, most non-AMD64 Linux support has already been removed, but some residual platform-specific code and documentation remains.

## Files to Remove/Modify

### 1. Documentation Files (Safe to Remove)
These are documentation files that can be safely removed as they don't affect compilation:
- `./docs/android.md`
- `./docs/android/` (directory with imported-into-android-studio.jpg)
- `./docs/backend/snapdragon/` (directory)
- `./docs/backend/snapdragon/windows.md`
- `./docs/build-riscv64-spacemit.md`
- `./docs/build-s390x.md`

### 2. Vendor/Specific Headers
- `./ggml/src/ggml-cuda/vendors/musa.h` - MUSA/Huawei GPU vendor header
  - **Action**: Remove if not used by CUDA backend for GTX 1050
  - **Verification**: Check if this is actually used in CUDA code paths

### 3. Source Code Cleanup Needed

#### Conditional Compilation to Review/Remove:

**In ggml/src/ggml-cpu/ggml-cpu.c:**
- Lines 49, 2746, 2807: `#ifdef GGML_USE_CPU_RISCV64_SPACEMIT` - RISC-V Spacemit support
- Lines 386, 388: `#elif defined(__riscv)` and `#ifdef __riscv_zihintpause` - RISC-V detection
- Lines 596: `#endif // __ARM_ARCH` - ARM architecture detection
- Lines 3138, 3172, 3192, 3242, 3262: Various RISC-V vector extensions checks
- Lines 3361, 3370, 3371: RISC-V vector intrinsic checks
- Lines 3540, 3541, 3544, 3545: ARM/RISC-V initialization

**In ggml/src/ggml-cpu/ops.cpp:**
- Lines 7141, 8255, 8489, 8848, 9012, 9439, 9469, 9533, 10047, 10055, 10067, 10259, 10267, 10279, 10724: ARM SVE and RISC-V vector checks

**In ggml/src/ggml-quants.c:**
- Lines 595, 744, 938: `#ifdef HAVE_BUGGY_APPLE_LINKER` - Apple linker workaround
- Lines 5288, 5329: `#elif defined(__ARM_NEON)` - ARM NEON support

**In ggml/src/ggml.c:**
- Lines 213, 264, 401, 451, 472: Windows/MingW detection (`_MSC_VER`, `__MINGW32__`, `_WIN32`)

**In tools/tokenize/tokenize.cpp:**
- Line 15: `#include <windows.h>` - Windows header

**In tools/server/server.cpp:**
- Line 21: `#include <windows.h>` - Windows header
- Line 312: `#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))` - Unix/Apple detection

**In common/common.cpp:**
- Lines 71, 125: `#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)` - Android exclusion
- Line 111, 125: Comments about efficiency cores harming lockstep threading

**In vendor/cpp-httplib/httplib.cpp:**
- Lines 2213, 2232, 2347, 5223, 5533, 8619, 8644, 11951, 11952, 12109, 12115, 12155, 12156, 12158, 12268, 12315, 12652, 13874, 13878, 15021, 15027: Windows/Android/Apple conditionals

### 4. CMake Files Review
Check for any remaining platform-specific CMake configurations:
- `./ggml/src/ggml-cpu/cmake/FindSMTIME.cmake`: Contains RISC-V Spacemit specific code (lines 3-31)

## Implementation Approach

### Phase 1: Safe Removals (Documentation and Unused Files)
1. Remove all platform-specific documentation files listed above
2. Remove `./ggml/src/ggml-cuda/vendors/musa.h` if verified unused
3. Remove RISC-V Spacemit CMake file if not used

### Phase 2: Source Code Simplification
For each platform-specific conditional block:
1. Determine if the code path is actually used for our target hardware (AMD64 Linux)
2. If not used, remove the entire conditional block and keep only the Linux/AMD64 path
3. If used but contains platform-specific optimizations we don't need, simplify to generic/fallback path
4. Update comments to reflect AMD64 Linux only focus

### Phase 3: Build System Cleanup
1. Review CMakeLists.txt files for any remaining platform-specific configurations
2. Ensure build system defaults to AMD64 Linux optimizations
3. Remove any platform-specific CMake toolchain files if they still exist

### Phase 4: Verification
1. Ensure code still compiles and runs correctly on AMD64 Linux
2. Verify CUDA backend still works for GTX 1050
3. Run basic tests to ensure functionality is preserved

## Hardware-Specific Considerations for Optimization

### CPU: AMD Opteron 3280 (bdver1)
- Supports: SSE4.2, AVX, AES, POPCNT
- Does NOT support: AVX2, FMA, F16C, BMI2, AVX512
- Should keep `-march=native` or explicit `-msse4.2 -mavx -maes -mpopcnt` flags
- Remove AVX2/FMA/F16C/BMI2/AVX512 specific code paths

### GPU: NVIDIA GTX 1050
- Compute Capability: 6.1 (Pascal architecture)
- CUDA: 12.0 compatible
- Should keep CUDA backend enabled
- No need for newer architecture features (Volcano, Ampere, etc.)

## Risk Assessment
- **Low Risk**: Removing documentation and unused vendor headers
- **Medium Risk**: Removing platform-specific conditionals - need to ensure we don't break legitimate code paths
- **Mitigation**: Keep fallback/generic paths, test thoroughly

## Estimated Effort
- Phase 1: 1-2 hours
- Phase 2: 4-6 hours (careful review of conditionals)
- Phase 3: 1-2 hours
- Phase 4: 2-3 hours (testing and verification)
- **Total**: 8-13 hours

## Success Criteria
1. Codebase compiles successfully on AMD64 Linux
2. CUDA backend works for GTX 1050
3. All platform-specific conditionals for non-AMD64/Linux systems are removed or simplified
4. Documentation reflects AMD64 Linux only focus
5. Build system is simplified and optimized for target hardware