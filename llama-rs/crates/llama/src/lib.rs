//! LLaMA inference engine.
//!
//! This crate provides the high-level API for loading and running GGUF models.

#![deny(missing_docs)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]

use thiserror::Error;

/// Errors that can occur during model loading or inference.
#[derive(Debug, Error)]
pub enum LlamaError {
    /// The model file could not be loaded.
    #[error("failed to load model: {0}")]
    LoadError(String),

    /// The model configuration is invalid.
    #[error("invalid model config: {0}")]
    InvalidConfig(String),

    /// An error occurred during inference.
    #[error("inference error: {0}")]
    InferenceError(String),
}

/// Result type alias for llama operations.
pub type LlamaResult<T> = Result<T, LlamaError>;

/// A loaded language model.
pub struct Model {
    path: String,
    parameter_count: u64,
}

impl Model {
    /// Load a model from a GGUF file.
    ///
    /// # Errors
    ///
    /// Returns [`LlamaError::LoadError`] if the file cannot be read or parsed.
    pub fn from_file(path: impl Into<String>) -> LlamaResult<Self> {
        let path = path.into();
        // TODO(#4): Implement full model loading from GGUF
        Ok(Self {
            path,
            parameter_count: 0,
        })
    }

    /// Returns the number of parameters in the model.
    #[must_use]
    pub fn parameter_count(&self) -> u64 {
        self.parameter_count
    }

    /// Returns the path to the model file.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }
}

/// Configuration for model inference.
pub struct ModelConfig {
    /// Number of CPU threads to use.
    pub n_threads: usize,
    /// Whether to use CUDA acceleration.
    pub use_cuda: bool,
    /// Context size (number of tokens).
    pub n_ctx: usize,
    /// Batch size for prompt processing.
    pub n_batch: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            use_cuda: false,
            n_ctx: 512,
            n_batch: 512,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_config_should_default_reasonable_values() {
        let config = ModelConfig::default();
        assert!(config.n_threads > 0);
        assert!(config.n_ctx > 0);
        assert!(config.n_batch > 0);
    }
}
