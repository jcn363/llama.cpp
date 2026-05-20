//! GGUF (GGML Universal File) format reader and writer.
//!
//! This crate provides safe Rust bindings for reading and writing GGUF files,
//! the binary format used by llama.cpp for storing model weights and metadata.
//!
//! # Example
//!
//! ```no_run
//! use gguf::GgufReader;
//!
//! let reader = GgufReader::from_file("model.gguf").unwrap();
//! println!("Tensor count: {}", reader.tensor_count());
//! ```

#![deny(missing_docs)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]

use std::path::Path;

use thiserror::Error;

/// Errors that can occur when reading or writing GGUF files.
#[derive(Debug, Error)]
pub enum GgufError {
    /// The file could not be opened or read.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// The file does not have a valid GGUF magic number.
    #[error("invalid GGUF magic number")]
    InvalidMagic,

    /// The GGUF version is not supported.
    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    /// A value could not be decoded.
    #[error("decode error: {0}")]
    DecodeError(String),

    /// The file is truncated or incomplete.
    #[error("unexpected end of file")]
    UnexpectedEof,
}

/// Result type alias for GGUF operations.
pub type GgufResult<T> = Result<T, GgufError>;

/// GGUF magic number: "GGUF" in little-endian.
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Supported GGUF version.
pub const GGUF_VERSION: u32 = 3;

/// A GGUF file reader that memory-maps the file for efficient access.
pub struct GgufReader {
    /// Memory-mapped file data.
    _data: memmap2::Mmap,
    /// Number of tensors in the file.
    tensor_count: u64,
    /// Number of metadata key-value pairs.
    metadata_count: u64,
}

impl GgufReader {
    /// Open a GGUF file from the given path.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] if the file cannot be opened or is not a valid GGUF file.
    pub fn from_file(path: impl AsRef<Path>) -> GgufResult<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(GgufError::UnexpectedEof);
        }

        let magic = u32::from_le_bytes(mmap[0..4].try_into().map_err(|_| GgufError::DecodeError("invalid magic bytes".into()))?);
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic);
        }

        let version = u32::from_le_bytes(mmap[4..8].try_into().map_err(|_| GgufError::DecodeError("invalid version bytes".into()))?);
        if version != GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }

        // TODO(#1): Parse full metadata header
        let tensor_count = 0u64;
        let metadata_count = 0u64;

        Ok(Self {
            _data: mmap,
            tensor_count,
            metadata_count,
        })
    }

    /// Returns the number of tensors in the GGUF file.
    #[must_use]
    pub fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    /// Returns the number of metadata key-value pairs.
    #[must_use]
    pub fn metadata_count(&self) -> u64 {
        self.metadata_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_magic_constant_should_be_correct() {
        assert_eq!(GGUF_MAGIC, 0x46554747);
        assert_eq!(core::str::from_utf8(&GGUF_MAGIC.to_le_bytes()).unwrap(), "GGUF");
    }

    #[test]
    fn gguf_version_should_be_three() {
        assert_eq!(GGUF_VERSION, 3);
    }

    #[test]
    fn from_file_should_return_error_for_missing_file() {
        let result = GgufReader::from_file("/nonexistent/path/file.gguf");
        assert!(result.is_err());
    }
}
