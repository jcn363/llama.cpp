//! Common utilities for llama inference.
//!
//! This crate provides shared functionality: argument parsing, chat templates,
//! sampling strategies, unicode handling, and Jinja template rendering.

#![deny(missing_docs)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]

/// Command-line argument parsing.
pub mod args {
    use clap::Parser;

    /// Common arguments for llama inference tools.
    #[derive(Parser, Debug)]
    pub struct CommonArgs {
        /// Path to the model file (GGUF format).
        #[arg(short, long)]
        pub model: String,

        /// Number of threads to use for computation.
        #[arg(short = 't', long, default_value_t = 0)]
        pub threads: usize,

        /// Context size (number of tokens).
        #[arg(short = 'c', long, default_value_t = 512)]
        pub ctx_size: usize,

        /// Batch size for prompt processing.
        #[arg(long, default_value_t = 512)]
        pub batch_size: usize,

        /// Use CUDA acceleration if available.
        #[arg(long, default_value_t = false)]
        pub use_cuda: bool,
    }
}

/// Sampling strategies for text generation.
pub mod sampling {
    /// Configuration for token sampling.
    #[derive(Debug, Clone)]
    pub struct SamplingConfig {
        /// Temperature for sampling (0.0 = greedy).
        pub temperature: f32,
        /// Top-k sampling (0 = disabled).
        pub top_k: usize,
        /// Top-p sampling (0.0 = disabled).
        pub top_p: f32,
        /// Repeat penalty (1.0 = no penalty).
        pub repeat_penalty: f32,
    }

    impl Default for SamplingConfig {
        fn default() -> Self {
            Self {
                temperature: 0.8,
                top_k: 40,
                top_p: 0.95,
                repeat_penalty: 1.1,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sampling::SamplingConfig;

    #[test]
    fn sampling_config_should_default_reasonable_values() {
        let config = SamplingConfig::default();
        assert!(config.temperature > 0.0);
        assert!(config.top_k > 0);
        assert!(config.top_p > 0.0 && config.top_p <= 1.0);
        assert!(config.repeat_penalty >= 1.0);
    }
}
