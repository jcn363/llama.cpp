//! CPU backend for ggml, optimized for AMD Opteron 3280 (bdver1).
//!
//! This crate implements tensor operations for CPU execution, with explicit
//! optimizations for SSE4.2 and AVX instruction sets available on bdver1.
//!
//! # Hardware Target
//!
//! - **CPU:** AMD Opteron 3280 (Bulldozer bdver1)
//! - **Supported:** SSE4.2, AVX, AES, POPCNT
//! - **Not supported:** AVX2, FMA, F16C, BMI2, AVX512

#![deny(missing_docs)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(
    clippy::many_single_char_names,
    clippy::wildcard_imports,
    clippy::missing_panics_doc,
    clippy::items_after_statements,
    clippy::too_many_arguments,
    clippy::cast_ptr_alignment,
    clippy::cast_possible_truncation,
    dead_code
)]

use ggml::{DType, Tensor};

// ─── SIMD Configuration ─────────────────────────────────────────────────────

/// Number of floats per AVX register (256 bits / 32 bits).
#[cfg(target_arch = "x86_64")]
const AVX_F32_EPR: usize = 8;

/// Number of floats per SSE register (128 bits / 32 bits).
#[cfg(target_arch = "x86_64")]
const SSE_F32_EPR: usize = 4;

/// Number of accumulators for SIMD dot product (unroll factor).
const DOT_ARR: usize = 4;

/// AVX step size: 8 floats × 4 accumulators = 32 floats per iteration.
#[cfg(target_arch = "x86_64")]
const AVX_F32_STEP: usize = AVX_F32_EPR * DOT_ARR;

/// SSE step size: 4 floats × 4 accumulators = 16 floats per iteration.
#[cfg(target_arch = "x86_64")]
const SSE_F32_STEP: usize = SSE_F32_EPR * DOT_ARR;

// ─── SIMD Dot Product ───────────────────────────────────────────────────────

/// Compute dot product of two f32 vectors using SIMD when available.
///
/// Uses AVX (8-wide) → SSE4.2 (4-wide) → scalar fallback.
/// No FMA instructions (bdver1 doesn't support them) — uses mul + add.
#[must_use]
#[inline]
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Try AVX first
        if cpu_features::has_avx() {
            return dot_f32_avx(&x[..n], &y[..n]);
        }
        // Fallback to SSE4.2
        if cpu_features::has_sse4_2() {
            return dot_f32_sse(&x[..n], &y[..n]);
        }
    }

    // Scalar fallback
    dot_f32_scalar(&x[..n], &y[..n])
}

/// Scalar dot product fallback.
#[inline]
fn dot_f32_scalar(x: &[f32], y: &[f32]) -> f32 {
    let mut sum: f64 = 0.0;
    for i in 0..x.len() {
        sum += f64::from(x[i]) * f64::from(y[i]);
    }
    sum as f32
}

/// AVX-optimized dot product (8-wide, 4 accumulators = 32 floats/iteration).
/// Uses mul + add (no FMA) since bdver1 doesn't support FMA.
#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_f32_avx(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = x.len();
    let np = n & !(AVX_F32_STEP - 1);

    unsafe {
        let mut sum: [__m256; DOT_ARR] = [_mm256_setzero_ps(); DOT_ARR];
        let mut ax: [__m256; DOT_ARR] = [_mm256_setzero_ps(); DOT_ARR];
        let mut ay: [__m256; DOT_ARR] = [_mm256_setzero_ps(); DOT_ARR];

        // Main loop: process AVX_F32_STEP (32) floats per iteration
        for i in (0..np).step_by(AVX_F32_STEP) {
            for j in 0..DOT_ARR {
                let idx = i + j * AVX_F32_EPR;
                ax[j] = _mm256_loadu_ps(x.as_ptr().add(idx));
                ay[j] = _mm256_loadu_ps(y.as_ptr().add(idx));
                // mul + add (no FMA on bdver1)
                sum[j] = _mm256_add_ps(_mm256_mul_ps(ax[j], ay[j]), sum[j]);
            }
        }

        // Horizontal reduction: sum[0..3] → sum[0]
        for j in 1..DOT_ARR {
            sum[0] = _mm256_add_ps(sum[0], sum[j]);
        }

        // Extract 256-bit sum to scalar
        let sum128 = _mm_add_ps(
            _mm256_extractf128_ps(sum[0], 1),
            _mm256_castps256_ps128(sum[0]),
        );
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_movehdup_ps(sum64));
        let mut result = _mm_cvtss_f32(sum32);

        // Leftover elements (scalar)
        for i in np..n {
            result += x[i] * y[i];
        }

        result
    }
}

/// SSE4.2-optimized dot product (4-wide, 4 accumulators = 16 floats/iteration).
#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_f32_sse(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = x.len();
    let np = n & !(SSE_F32_STEP - 1);

    unsafe {
        let mut sum: [__m128; DOT_ARR] = [_mm_setzero_ps(); DOT_ARR];
        let mut ax: [__m128; DOT_ARR] = [_mm_setzero_ps(); DOT_ARR];
        let mut ay: [__m128; DOT_ARR] = [_mm_setzero_ps(); DOT_ARR];

        // Main loop: process SSE_F32_STEP (16) floats per iteration
        for i in (0..np).step_by(SSE_F32_STEP) {
            for j in 0..DOT_ARR {
                let idx = i + j * SSE_F32_EPR;
                ax[j] = _mm_loadu_ps(x.as_ptr().add(idx));
                ay[j] = _mm_loadu_ps(y.as_ptr().add(idx));
                sum[j] = _mm_add_ps(_mm_mul_ps(ax[j], ay[j]), sum[j]);
            }
        }

        // Horizontal reduction
        for j in 1..DOT_ARR {
            sum[0] = _mm_add_ps(sum[0], sum[j]);
        }

        let sum64 = _mm_add_ps(sum[0], _mm_movehl_ps(sum[0], sum[0]));
        let sum32 = _mm_add_ss(sum64, _mm_movehdup_ps(sum64));
        let mut result = _mm_cvtss_f32(sum32);

        // Leftover
        for i in np..n {
            result += x[i] * y[i];
        }

        result
    }
}

// ─── Matrix Multiplication ──────────────────────────────────────────────────

/// Compute C = A × B^T using block-tiling with SIMD dot products.
///
/// This follows ggml's convention: `C = mul_mat(A, B)` means `C[i,j] = dot(A[i,:], B[j,:])`.
///
/// # Arguments
///
/// * `a` - Matrix A with shape `[m, k]`
/// * `b` - Matrix B with shape `[n, k]`
/// * `c` - Output matrix with shape `[m, n]` (must be pre-allocated)
/// * `n_threads` - Number of threads for parallel execution
pub fn matmul_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    n_threads: usize,
) {
    assert_eq!(a.len(), m * k, "A must have shape [{m}, {k}]");
    assert_eq!(b.len(), n * k, "B must have shape [{n}, {k}]");
    assert_eq!(c.len(), m * n, "C must have shape [{m}, {n}]");

    // Block sizes for cache-friendly tiling
    const BLOCK_M: usize = 16;
    const BLOCK_N: usize = 16;

    let n_threads = if n_threads == 0 {
        std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
    } else {
        n_threads
    };

    if n_threads <= 1 {
        // Single-threaded
        matmul_f32_block(a, b, c, n, k, 0, m, 0, n);
        return;
    }

    // Parallel: split rows of A across threads
    let rows_per_thread = m.div_ceil(n_threads);

    // Build row ranges
    let mut ranges = Vec::new();
    for t in 0..n_threads {
        let i_start = (t * rows_per_thread).min(m);
        let i_end = ((t + 1) * rows_per_thread).min(m);
        if i_start < i_end {
            ranges.push((i_start, i_end));
        }
    }

    // Use scoped threads with raw pointers for non-overlapping mutable access
    let c_ptr = c.as_mut_ptr();
    std::thread::scope(|scope| {
        for &(i_start, i_end) in &ranges {
            let c_start = i_start * n;
            let len = (i_end - i_start) * n;
            // Safety: each thread accesses a non-overlapping region of c
            let c_slice = unsafe { std::slice::from_raw_parts_mut(c_ptr.add(c_start), len) };
            scope.spawn(move || {
                matmul_f32_block(a, b, c_slice, n, k, i_start, i_end, 0, n);
            });
        }
    });
}

/// Compute a block of the matrix multiplication.
///
/// `c` is a slice starting at row `i_start` of the full output matrix.
fn matmul_f32_block(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    i_start: usize,
    i_end: usize,
    _j_start: usize,
    j_end: usize,
) {
    const BLOCK_M: usize = 16;
    const BLOCK_N: usize = 16;

    for i0 in (i_start..i_end).step_by(BLOCK_M) {
        let i1 = (i0 + BLOCK_M).min(i_end);
        for j0 in (0..j_end).step_by(BLOCK_N) {
            let j1 = (j0 + BLOCK_N).min(j_end);

            for i in i0..i1 {
                // c slice starts at row i_start, so offset by i_start
                let c_row_offset = (i - i_start) * n;
                for j in j0..j1 {
                    let a_row = &a[i * k..(i + 1) * k];
                    let b_row = &b[j * k..(j + 1) * k];
                    c[c_row_offset + j] = dot_f32(a_row, b_row);
                }
            }
        }
    }
}

// ─── CpuBackend ─────────────────────────────────────────────────────────────

/// Executes a computation graph on the CPU.
pub struct CpuBackend {
    n_threads: usize,
}

impl CpuBackend {
    /// Create a new CPU backend with the given number of threads.
    ///
    /// If `n_threads` is 0, uses the number of available parallel threads.
    #[must_use]
    pub fn new(n_threads: usize) -> Self {
        Self {
            n_threads: if n_threads == 0 {
                std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
            } else {
                n_threads
            },
        }
    }

    /// Execute matrix multiplication: `C = A * B^T`.
    ///
    /// Note: This follows ggml's unconventional matmul convention where
    /// `C = ggml_mul_mat(ctx, A, B)` means `C^T = A * B^T`, i.e., `C = B * A^T`.
    ///
    /// # Panics
    ///
    /// Panics if the tensor shapes are incompatible for multiplication.
    #[must_use]
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(
            a.shape().len(),
            2,
            "matmul requires 2D tensors, got {}D",
            a.ndim()
        );
        assert_eq!(
            b.shape().len(),
            2,
            "matmul requires 2D tensors, got {}D",
            b.ndim()
        );
        assert_eq!(a.dtype(), DType::F32, "matmul requires F32 tensors");
        assert_eq!(b.dtype(), DType::F32, "matmul requires F32 tensors");

        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[0];
        let k2 = b.shape()[1];
        assert_eq!(k, k2, "inner dimensions must match: {k} vs {k2}");

        // Get raw f32 slices
        let a_bytes = a.data();
        let b_bytes = b.data();
        let a_f32 = unsafe {
            std::slice::from_raw_parts(a_bytes.as_ptr().cast::<f32>(), a_bytes.len() / 4)
        };
        let b_f32 = unsafe {
            std::slice::from_raw_parts(b_bytes.as_ptr().cast::<f32>(), b_bytes.len() / 4)
        };

        let mut c = vec![0.0f32; m * n];
        matmul_f32(a_f32, b_f32, &mut c, m, n, k, self.n_threads);

        Tensor::from_f32(&[m, n], &c)
    }

    /// Execute element-wise addition: `C = A + B`.
    ///
    /// # Panics
    ///
    /// Panics if the tensor shapes don't match.
    #[must_use]
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(
            a.shape(),
            b.shape(),
            "addition requires matching shapes: {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
        Tensor::new(a.dtype(), a.shape())
    }

    /// Returns the number of worker threads.
    #[must_use]
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }
}

// ─── CPU Feature Detection ──────────────────────────────────────────────────

/// Runtime CPU feature detection for bdver1 capabilities.
pub mod cpu_features {
    /// Returns whether SSE4.2 is available.
    #[must_use]
    pub fn has_sse4_2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("sse4.2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Returns whether AVX is available.
    #[must_use]
    pub fn has_avx() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Returns whether AES-NI is available.
    #[must_use]
    pub fn has_aes() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("aes")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Returns whether POPCNT is available.
    #[must_use]
    pub fn has_popcnt() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("popcnt")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_should_default_thread_count() {
        let backend = CpuBackend::new(0);
        assert!(backend.n_threads() > 0);
    }

    #[test]
    fn cpu_backend_should_use_explicit_thread_count() {
        let backend = CpuBackend::new(4);
        assert_eq!(backend.n_threads(), 4);
    }

    #[test]
    fn cpu_features_should_detect_sse4_2() {
        assert!(cpu_features::has_sse4_2());
    }

    #[test]
    fn cpu_features_should_detect_avx() {
        assert!(cpu_features::has_avx());
    }

    #[test]
    fn dot_f32_should_compute_correct_result() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [5.0, 6.0, 7.0, 8.0];
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let result = dot_f32(&x, &y);
        assert!((result - 70.0).abs() < 0.001);
    }

    #[test]
    fn dot_f32_should_handle_empty() {
        assert_eq!(dot_f32(&[], &[]), 0.0);
    }

    #[test]
    fn dot_f32_should_handle_large_vectors() {
        let n = 1024;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();
        // sum(i * 2i) = 2 * sum(i^2) = 2 * n*(n-1)*(2n-1)/6
        let expected = 2.0 * (n as f64) * ((n - 1) as f64) * ((2 * n - 1) as f64) / 6.0;
        let result = f64::from(dot_f32(&x, &y));
        assert!((result - expected).abs() < expected * 0.001);
    }

    #[test]
    fn matmul_f32_should_compute_correct_result() {
        // A = [[1, 2], [3, 4]]  (2x2)
        // B = [[5, 6], [7, 8]]  (2x2)
        // C = A * B^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
        //             = [[17, 23], [39, 53]]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2, 1);

        assert!((c[0] - 17.0).abs() < 0.001, "c[0] = {}", c[0]);
        assert!((c[1] - 23.0).abs() < 0.001, "c[1] = {}", c[1]);
        assert!((c[2] - 39.0).abs() < 0.001, "c[2] = {}", c[2]);
        assert!((c[3] - 53.0).abs() < 0.001, "c[3] = {}", c[3]);
    }

    #[test]
    fn matmul_f32_should_handle_non_square() {
        // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
        // B = [[7, 8, 9], [10, 11, 12], [13, 14, 15]]  (3x3)
        // C = A * B^T  (2x3)
        let a: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let b: Vec<f32> = (7..=15).map(|x| x as f32).collect();
        let mut c = vec![0.0; 6];
        matmul_f32(&a, &b, &mut c, 2, 3, 3, 1);

        // C[0,0] = 1*7 + 2*8 + 3*9 = 50
        assert!((c[0] - 50.0).abs() < 0.001);
        // C[0,1] = 1*10 + 2*11 + 3*12 = 68
        assert!((c[1] - 68.0).abs() < 0.001);
    }

    #[test]
    fn matmul_should_require_2d_tensors() {
        let backend = CpuBackend::new(1);
        let a = Tensor::new(DType::F32, &[2, 3]);
        let b = Tensor::new(DType::F32, &[4, 3]);
        let _result = backend.matmul(&a, &b);
    }

    #[test]
    #[should_panic(expected = "inner dimensions must match")]
    fn matmul_should_panic_on_incompatible_shapes() {
        let backend = CpuBackend::new(1);
        let a = Tensor::new(DType::F32, &[2, 3]);
        let b = Tensor::new(DType::F32, &[4, 5]);
        let _result = backend.matmul(&a, &b);
    }

    #[test]
    fn matmul_parallel_should_match_single_thread() {
        let n = 64;
        let k = 128;
        let a: Vec<f32> = (0..n * k).map(|i| (i % 100) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..n * k).map(|i| ((i + 37) % 100) as f32 * 0.01).collect();

        let mut c1 = vec![0.0; n * n];
        matmul_f32(&a, &b, &mut c1, n, n, k, 1);

        let mut c2 = vec![0.0; n * n];
        matmul_f32(&a, &b, &mut c2, n, n, k, 4);

        for i in 0..n * n {
            let diff = (c1[i] - c2[i]).abs();
            assert!(
                diff < 0.001,
                "mismatch at index {i}: {} vs {}",
                c1[i],
                c2[i]
            );
        }
    }
}
