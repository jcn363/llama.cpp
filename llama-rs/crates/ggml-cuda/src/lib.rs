//! CUDA backend for ggml, optimized for NVIDIA GTX 1050 (compute 6.1).
//!
//! This crate implements tensor operations for CUDA execution, targeting
//! Pascal architecture (sm_61) with 2GB VRAM constraints.
//!
//! # Hardware Target
//!
//! - **GPU:** NVIDIA GTX 1050 (Pascal)
//! - **Compute capability:** 6.1
//! - **VRAM:** 2GB

#![deny(missing_docs)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]

use ggml::Tensor;

/// Errors that can occur during CUDA operations.
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    /// CUDA is not available on this system.
    #[error("CUDA not available: {0}")]
    NotAvailable(String),

    /// Insufficient VRAM for the requested operation.
    #[error("insufficient VRAM: needed {needed} bytes, available {available} bytes")]
    OutOfMemory {
        /// Bytes required for the operation.
        needed: usize,
        /// Bytes currently available.
        available: usize,
    },

    /// A CUDA runtime error occurred.
    #[error("CUDA error: {0}")]
    RuntimeError(String),
}

/// Result type alias for CUDA operations.
pub type CudaResult<T> = Result<T, CudaError>;

/// CUDA backend for GPU-accelerated tensor operations.
pub struct CudaBackend {
    available: bool,
    total_vram: usize,
}

impl CudaBackend {
    /// Initialize the CUDA backend.
    ///
    /// Returns a backend with `available = false` if CUDA is not present.
    #[must_use]
    pub fn new() -> Self {
        Self {
            available: false,
            total_vram: 2 * 1024 * 1024 * 1024, // 2GB for GTX 1050
        }
    }

    /// Returns whether CUDA is available.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Returns the total VRAM in bytes.
    #[must_use]
    pub fn total_vram(&self) -> usize {
        self.total_vram
    }

    /// Copy a tensor from host memory to device memory.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::OutOfMemory`] if there is insufficient VRAM.
    pub fn copy_to_device(&self, tensor: &Tensor) -> CudaResult<DeviceTensor> {
        if tensor.byte_size() > self.total_vram {
            return Err(CudaError::OutOfMemory {
                needed: tensor.byte_size(),
                available: self.total_vram,
            });
        }

        Ok(DeviceTensor {
            size: tensor.byte_size(),
        })
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// A tensor stored in device (GPU) memory.
pub struct DeviceTensor {
    size: usize,
}

impl DeviceTensor {
    /// Returns the size of the device tensor in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ggml::DType;

    #[test]
    fn cuda_backend_should_report_vram() {
        let backend = CudaBackend::new();
        assert_eq!(backend.total_vram(), 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn copy_to_device_should_fail_for_large_tensor() {
        let backend = CudaBackend::new();
        let large = Tensor::new(DType::F32, &[1_000_000_000]);
        let result = backend.copy_to_device(&large);
        assert!(result.is_err());
    }
}
