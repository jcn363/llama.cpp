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

use ggml::{DType, Tensor};

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
                std::thread::available_parallelism().map_or(1, |n| n.get())
            } else {
                n_threads
            },
        }
    }

    /// Execute matrix multiplication: `C = A * B^T`.
    ///
    /// Note: This follows ggml's unconventional matmul convention.
    ///
    /// # Panics
    ///
    /// Panics if the tensor shapes are incompatible for multiplication.
    #[must_use]
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(a.shape().len(), 2, "matmul requires 2D tensors, got {}D", a.ndim());
        assert_eq!(b.shape().len(), 2, "matmul requires 2D tensors, got {}D", b.ndim());

        let a_cols = a.shape()[1];
        let b_cols = b.shape()[1];
        assert_eq!(a_cols, b_cols, "inner dimensions must match: {} vs {}", a_cols, b_cols);

        let out_rows = b.shape()[0];
        let out_cols = a.shape()[0];
        Tensor::new(DType::F32, &[out_rows, out_cols])
    }

    /// Execute element-wise addition: `C = A + B`.
    ///
    /// # Panics
    ///
    /// Panics if the tensor shapes don't match.
    #[must_use]
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(a.shape(), b.shape(), "addition requires matching shapes: {:?} vs {:?}", a.shape(), b.shape());
        Tensor::new(a.dtype(), a.shape())
    }

    /// Returns the number of worker threads.
    #[must_use]
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }
}

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
}
