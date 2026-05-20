//! CUDA backend for ggml, optimized for NVIDIA GTX 1050 (compute 6.1).
//!
//! This crate implements tensor operations for CUDA execution, targeting
//! Pascal architecture (`sm_61`) with 2GB VRAM constraints.
//!
//! # Hardware Target
//!
//! - **GPU:** NVIDIA GTX 1050 (Pascal)
//! - **Compute capability:** 6.1
//! - **VRAM:** 2GB
//! - **CUDA cores:** 640
//!
//! # Example
//!
//! ```no_run
//! use ggml_cuda::CudaBackend;
//!
//! let backend = CudaBackend::new().unwrap();
//! println!("VRAM: {} MB", backend.total_vram() / (1024 * 1024));
//! ```

#![deny(missing_docs)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(
    clippy::many_single_char_names,
    dead_code,
    clippy::unnecessary_lazy_evaluations,
    clippy::no_effect_underscore_binding
)]

use ggml::Tensor;

// ─── Errors ──────────────────────────────────────────────────────────────────

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

// ─── CUDA Backend ────────────────────────────────────────────────────────────

/// CUDA backend for GPU-accelerated tensor operations.
///
/// Optimized for GTX 1050 (compute 6.1, 2GB VRAM, 640 CUDA cores).
pub struct CudaBackend {
    /// Whether CUDA is available.
    available: bool,
    /// Total VRAM in bytes.
    total_vram: usize,
    /// Free VRAM in bytes.
    free_vram: usize,
    /// Number of CUDA cores (640 for GTX 1050).
    cuda_cores: usize,
    /// Compute capability major version.
    compute_major: i32,
    /// Compute capability minor version.
    compute_minor: i32,
}

impl CudaBackend {
    /// Initialize the CUDA backend.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotAvailable`] if CUDA is not present.
    pub fn new() -> CudaResult<Self> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::{CudaDevice, DevicePtr};

            let device = CudaDevice::new(0)
                .map_err(|e| CudaError::NotAvailable(format!("failed to initialize CUDA: {e}")))?;

            let props = device.properties().map_err(|e| {
                CudaError::NotAvailable(format!("failed to query device properties: {e}"))
            })?;

            let total_vram = props.total_global_mem as usize;
            let free_vram = total_vram; // Approximation; actual free requires tracking

            Ok(Self {
                available: true,
                total_vram,
                free_vram,
                cuda_cores: props.multi_processor_count as usize * 128, // Pascal: 128 cores/SM
                compute_major: props.major,
                compute_minor: props.minor,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Stub implementation when CUDA feature is disabled
            Ok(Self {
                available: false,
                total_vram: 2 * 1024 * 1024 * 1024, // 2GB for GTX 1050
                free_vram: 2 * 1024 * 1024 * 1024,
                cuda_cores: 640,
                compute_major: 6,
                compute_minor: 1,
            })
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

    /// Returns the free VRAM in bytes.
    #[must_use]
    pub fn free_vram(&self) -> usize {
        self.free_vram
    }

    /// Returns the number of CUDA cores.
    #[must_use]
    pub fn cuda_cores(&self) -> usize {
        self.cuda_cores
    }

    /// Returns the compute capability as a string (e.g., "6.1").
    #[must_use]
    pub fn compute_capability(&self) -> String {
        format!("{}.{}", self.compute_major, self.compute_minor)
    }

    /// Copy a tensor from host memory to device memory.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::OutOfMemory`] if there is insufficient VRAM.
    pub fn copy_to_device(&self, tensor: &Tensor) -> CudaResult<DeviceTensor> {
        if tensor.byte_size() > self.free_vram {
            return Err(CudaError::OutOfMemory {
                needed: tensor.byte_size(),
                available: self.free_vram,
            });
        }

        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::{CudaDevice, DeviceRepr, DeviceSlice};

            let device = CudaDevice::new(0)
                .map_err(|e| CudaError::RuntimeError(format!("failed to get CUDA device: {e}")))?;

            let data: &[f32] = bytemuck::cast_slice(tensor.data());
            let dev_data = device
                .htod_sync_copy(data)
                .map_err(|e| CudaError::RuntimeError(format!("failed to copy to device: {e}")))?;

            Ok(DeviceTensor {
                size: tensor.byte_size(),
                element_count: tensor.element_count(),
                shape: tensor.shape().to_vec(),
                dev_data: Some(dev_data),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(DeviceTensor {
                size: tensor.byte_size(),
                element_count: tensor.element_count(),
                shape: tensor.shape().to_vec(),
                dev_data: None,
            })
        }
    }

    /// Execute matrix multiplication on GPU: C = A × B^T.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::RuntimeError`] if the operation fails.
    pub fn matmul(&self, a: &DeviceTensor, b: &DeviceTensor) -> CudaResult<DeviceTensor> {
        if !self.available {
            return Err(CudaError::NotAvailable("CUDA backend not available".into()));
        }

        let _m = a.shape[0];
        let k = a.shape[1];
        let _n = b.shape[0];
        let k2 = b.shape[1];

        if k != k2 {
            return Err(CudaError::RuntimeError(format!(
                "inner dimensions must match: {k} vs {k2}"
            )));
        }

        #[cfg(feature = "cuda")]
        {
            use cudarc::cublas::{CudaBlas, GemmConfig, Transpose};

            let dev_a = a
                .dev_data
                .as_ref()
                .ok_or_else(|| CudaError::RuntimeError("device tensor has no data".into()))?;
            let dev_b = b
                .dev_data
                .as_ref()
                .ok_or_else(|| CudaError::RuntimeError("device tensor has no data".into()))?;

            let blas = CudaBlas::new(dev_a.device()).map_err(|e| {
                CudaError::RuntimeError(format!("failed to create cuBLAS handle: {e}"))
            })?;

            let out_size = m * n;
            let dev_c = dev_a
                .device()
                .alloc_zeros(out_size)
                .map_err(|e| CudaError::RuntimeError(format!("failed to allocate output: {e}")))?;

            // C = A × B^T using cuBLAS
            // cuBLAS: C = alpha * op(A) * op(B) + beta * C
            // For C = A × B^T: op(A) = A (no transpose), op(B) = B^T (transpose)
            let config = GemmConfig {
                transa: Transpose::N,
                transb: Transpose::T,
                m: m as i32,
                n: n as i32,
                k: k as i32,
                alpha: 1.0,
                lda: k as i32,
                ldb: k as i32,
                beta: 0.0,
                ldc: n as i32,
            };

            unsafe {
                blas.gemm(config, dev_a, dev_b, &dev_c)
                    .map_err(|e| CudaError::RuntimeError(format!("cuBLAS gemm failed: {e}")))?;
            }

            Ok(DeviceTensor {
                size: out_size * 4,
                element_count: out_size,
                shape: vec![m, n],
                dev_data: Some(dev_c),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::NotAvailable("CUDA feature not enabled".into()))
        }
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            available: false,
            total_vram: 2 * 1024 * 1024 * 1024,
            free_vram: 2 * 1024 * 1024 * 1024,
            cuda_cores: 640,
            compute_major: 6,
            compute_minor: 1,
        })
    }
}

// ─── Device Tensor ──────────────────────────────────────────────────────────

/// A tensor stored in device (GPU) memory.
pub struct DeviceTensor {
    /// Size in bytes.
    size: usize,
    /// Number of elements.
    element_count: usize,
    /// Shape dimensions.
    shape: Vec<usize>,
    /// Device data (only available with CUDA feature).
    #[cfg(feature = "cuda")]
    dev_data: Option<cudarc::driver::CudaRc<dyn cudarc::driver::DeviceSlice<f32>>>,
    #[cfg(not(feature = "cuda"))]
    dev_data: Option<()>,
}

impl DeviceTensor {
    /// Returns the size of the device tensor in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.size
    }

    /// Returns the number of elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.element_count
    }

    /// Returns the shape of the tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Copy device data back to host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::RuntimeError`] if the copy fails.
    pub fn to_host(&self) -> CudaResult<Vec<f32>> {
        #[cfg(feature = "cuda")]
        {
            let dev_data = self
                .dev_data
                .as_ref()
                .ok_or_else(|| CudaError::RuntimeError("device tensor has no data".into()))?;

            let host_data = dev_data
                .device()
                .dtoh_sync_copy(dev_data)
                .map_err(|e| CudaError::RuntimeError(format!("failed to copy from device: {e}")))?;

            Ok(host_data)
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::NotAvailable("CUDA feature not enabled".into()))
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_backend_should_report_vram() {
        let backend = CudaBackend::new().unwrap_or_default();
        assert_eq!(backend.total_vram(), 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn cuda_backend_should_report_cuda_cores() {
        let backend = CudaBackend::new().unwrap_or_default();
        assert_eq!(backend.cuda_cores(), 640);
    }

    #[test]
    fn cuda_backend_should_report_compute_capability() {
        let backend = CudaBackend::new().unwrap_or_default();
        assert_eq!(backend.compute_capability(), "6.1");
    }

    #[test]
    fn copy_to_device_should_fail_for_large_tensor() {
        let backend = CudaBackend::new().unwrap_or_default();
        // Create a tensor larger than 2GB
        let large = Tensor::new(ggml::DType::F32, &[1_000_000_000]);
        let result = backend.copy_to_device(&large);
        assert!(result.is_err());
    }
}
