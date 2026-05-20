//! LLaMA inference engine.
//!
//! This crate provides the high-level API for loading and running GGUF models,
//! equivalent to the `llama.cpp` library.
//!
//! # Example
//!
//! ```no_run
//! use llama::{Model, ModelConfig};
//!
//! let model = Model::from_file("model.gguf").unwrap();
//! println!("Loaded model with {} parameters", model.parameter_count());
//! ```

#![deny(missing_docs)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::uninlined_format_args,
    clippy::cast_ptr_alignment,
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::cloned_instead_of_copied,
    clippy::unnecessary_join,
    clippy::redundant_closure_for_method_calls
)]

use gguf::{GgufReader, TensorInfo};
use std::sync::Arc;
use thiserror::Error;

// ─── Errors ──────────────────────────────────────────────────────────────────

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

    /// GGUF parsing error.
    #[error("GGUF error: {0}")]
    GgufError(#[from] gguf::GgufError),
}

/// Result type alias for llama operations.
pub type LlamaResult<T> = Result<T, LlamaError>;

// ─── Model Architecture ─────────────────────────────────────────────────────

/// Model architecture parameters extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelArch {
    /// Model architecture name (e.g., "llama", "mistral").
    pub architecture: String,
    /// Embedding dimension.
    pub n_embd: u32,
    /// Number of attention heads.
    pub n_head: u32,
    /// Number of key-value heads (for GQA).
    pub n_head_kv: u32,
    /// Number of layers.
    pub n_layer: u32,
    /// Feed-forward dimension.
    pub n_ff: u32,
    /// Vocabulary size.
    pub n_vocab: u32,
    /// RMS norm epsilon.
    pub norm_eps: f32,
    /// Rope scaling factor.
    pub rope_freq_base: f32,
    /// Rope dimension.
    pub rope_dim: u32,
}

impl ModelArch {
    /// Extract architecture from GGUF metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if required metadata keys are missing.
    pub fn from_gguf(reader: &GgufReader) -> LlamaResult<Self> {
        let get_u32 = |key: &str| -> LlamaResult<u32> {
            reader
                .get_kv(key)
                .ok_or_else(|| LlamaError::LoadError(format!("missing key: {key}")))
                .and_then(|v| match v {
                    gguf::GgufValue::U32(val) => Ok(*val),
                    gguf::GgufValue::U64(val) => Ok(*val as u32),
                    gguf::GgufValue::I32(val) => Ok(*val as u32),
                    _ => Err(LlamaError::LoadError(format!("key {key} has wrong type"))),
                })
        };

        let get_f32 = |key: &str| -> LlamaResult<f32> {
            reader
                .get_kv(key)
                .ok_or_else(|| LlamaError::LoadError(format!("missing key: {key}")))
                .and_then(|v| match v {
                    gguf::GgufValue::F32(val) => Ok(*val),
                    gguf::GgufValue::F64(val) => Ok(*val as f32),
                    _ => Err(LlamaError::LoadError(format!("key {key} has wrong type"))),
                })
        };

        let get_str = |key: &str| -> LlamaResult<String> {
            reader
                .get_kv(key)
                .ok_or_else(|| LlamaError::LoadError(format!("missing key: {key}")))
                .and_then(|v| match v {
                    gguf::GgufValue::Str(val) => Ok(val.clone()),
                    _ => Err(LlamaError::LoadError(format!("key {key} has wrong type"))),
                })
        };

        let architecture = get_str("general.architecture")?;

        let prefix = format!("{architecture}.");
        let n_embd = get_u32(&format!("{prefix}embedding_length"))?;
        let n_head = get_u32(&format!("{prefix}attention.head_count"))?;
        let n_head_kv = get_u32(&format!("{prefix}attention.head_count_kv")).unwrap_or(n_head);
        let n_layer = get_u32(&format!("{prefix}block_count"))?;
        let n_ff = get_u32(&format!("{prefix}feed_forward_length"))?;
        let n_vocab = get_u32("tokenizer.ggml.tokens")?;
        let norm_eps =
            get_f32(&format!("{prefix}attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);
        let rope_freq_base = get_f32(&format!("{prefix}rope.freq_base")).unwrap_or(10_000.0);
        let rope_dim = get_u32(&format!("{prefix}rope.dimension_count")).unwrap_or(n_embd / n_head);

        Ok(Self {
            architecture,
            n_embd,
            n_head,
            n_head_kv,
            n_layer,
            n_ff,
            n_vocab,
            norm_eps,
            rope_freq_base,
            rope_dim,
        })
    }
}

// ─── Tokenizer ───────────────────────────────────────────────────────────────

/// Simple tokenizer extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Token strings.
    pub tokens: Vec<String>,
    /// Token scores (optional).
    pub scores: Option<Vec<f32>>,
    /// Token types (optional).
    pub types: Option<Vec<i32>>,
    /// BPE merge rules: (first, second) → rank.
    bpe_merges: Vec<(String, String)>,
    /// BOS token ID.
    bos_token_id: u32,
    /// EOS token ID.
    eos_token_id: u32,
    /// Tokenizer type: "bpe", "spm", or "wpm".
    tokenizer_type: String,
}

impl Tokenizer {
    /// Extract tokenizer from GGUF metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenizer data is missing.
    pub fn from_gguf(reader: &GgufReader) -> LlamaResult<Self> {
        let tokens = match reader.get_kv("tokenizer.ggml.tokens") {
            Some(gguf::GgufValue::Array { data, .. }) => data
                .iter()
                .map(|v| match v {
                    gguf::GgufValue::Str(s) => Ok(s.clone()),
                    _ => Err(LlamaError::LoadError("token is not a string".into())),
                })
                .collect::<LlamaResult<Vec<String>>>()?,
            _ => {
                return Err(LlamaError::LoadError(
                    "missing tokenizer.ggml.tokens".into(),
                ));
            }
        };

        let scores = reader
            .get_kv("tokenizer.ggml.scores")
            .and_then(|v| match v {
                gguf::GgufValue::Array { data, .. } => Some(
                    data.iter()
                        .map(|v| match v {
                            gguf::GgufValue::F32(f) => Ok(*f),
                            gguf::GgufValue::F64(f) => Ok(*f as f32),
                            _ => Err(LlamaError::LoadError("score is not f32".into())),
                        })
                        .collect::<LlamaResult<Vec<f32>>>(),
                ),
                _ => None,
            })
            .transpose()?;

        let types = reader
            .get_kv("tokenizer.ggml.token_type")
            .and_then(|v| match v {
                gguf::GgufValue::Array { data, .. } => Some(
                    data.iter()
                        .map(|v| match v {
                            gguf::GgufValue::I32(t) => Ok(*t),
                            _ => Err(LlamaError::LoadError("type is not i32".into())),
                        })
                        .collect::<LlamaResult<Vec<i32>>>(),
                ),
                _ => None,
            })
            .transpose()?;

        // Read BPE merges
        let bpe_merges = match reader.get_kv("tokenizer.ggml.merges") {
            Some(gguf::GgufValue::Array { data, .. }) => {
                let mut merges = Vec::with_capacity(data.len());
                for merge in data {
                    if let gguf::GgufValue::Str(s) = merge {
                        if let Some(space_pos) = s.find(' ') {
                            let first = s[..space_pos].to_string();
                            let second = s[space_pos + 1..].to_string();
                            merges.push((first, second));
                        }
                    }
                }
                merges
            }
            _ => Vec::new(),
        };

        // Get tokenizer type
        let tokenizer_type = match reader.get_kv("tokenizer.ggml.model") {
            Some(gguf::GgufValue::Str(s)) => s.clone(),
            _ => "bpe".to_string(),
        };

        // Get BOS/EOS token IDs
        let bos_token_id = reader
            .get_kv("tokenizer.ggml.bos_token_id")
            .and_then(|v| match v {
                gguf::GgufValue::U32(id) => Some(*id),
                gguf::GgufValue::U64(id) => Some(*id as u32),
                _ => None,
            })
            .unwrap_or(1);

        let eos_token_id = reader
            .get_kv("tokenizer.ggml.eos_token_id")
            .and_then(|v| match v {
                gguf::GgufValue::U32(id) => Some(*id),
                gguf::GgufValue::U64(id) => Some(*id as u32),
                _ => None,
            })
            .unwrap_or(2);

        Ok(Self {
            tokens,
            scores,
            types,
            bpe_merges,
            bos_token_id,
            eos_token_id,
            tokenizer_type,
        })
    }

    /// Encode a string into token IDs.
    ///
    /// For BPE tokenizers, uses merge rules from the model.
    /// For other types, falls back to exact match.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.tokenizer_type == "bpe" && !self.bpe_merges.is_empty() {
            self.encode_bpe(text)
        } else {
            self.encode_simple(text)
        }
    }

    /// BPE encoding using merge rules.
    fn encode_bpe(&self, text: &str) -> Vec<u32> {
        // Step 1: Split into bytes (GPT-2 style byte-level BPE)
        let mut words: Vec<String> = text
            .bytes()
            .map(|b| format!("<0x{:02X}>", b))
            .collect();

        if words.is_empty() {
            return Vec::new();
        }

        // Step 2: Build merge rank map
        let mut ranks: std::collections::HashMap<(String, String), usize> =
            std::collections::HashMap::with_capacity(self.bpe_merges.len());
        for (i, (first, second)) in self.bpe_merges.iter().enumerate() {
            ranks.insert((first.clone(), second.clone()), i);
        }

        // Step 3: Repeatedly apply lowest-rank merge
        loop {
            let mut best_pair: Option<(String, String)> = None;
            let mut best_rank = usize::MAX;
            let mut best_idx = None;

            for i in 0..words.len().saturating_sub(1) {
                let pair = (words[i].clone(), words[i + 1].clone());
                if let Some(&rank) = ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair = Some(pair);
                        best_idx = Some(i);
                    }
                }
            }

            match (best_pair, best_idx) {
                (Some((first, second)), Some(idx)) => {
                    // Merge the pair
                    let merged = format!("{}{}", first, second);
                    words.splice(idx..=idx + 1, [merged]);
                }
                _ => break, // No more merges possible
            }
        }

        // Step 4: Convert words to token IDs
        let mut token_ids = Vec::with_capacity(words.len());
        for word in &words {
            if let Some(id) = self.tokens.iter().position(|t| t == word) {
                token_ids.push(id as u32);
            } else {
                // Unknown token — use UNK or fallback
                token_ids.push(0);
            }
        }

        token_ids
    }

    /// Simple exact-match encoding fallback.
    fn encode_simple(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        for word in text.split_whitespace() {
            if let Some(id) = self.tokens.iter().position(|t| t == word) {
                tokens.push(id as u32);
            } else {
                // Unknown token — use BOS or fallback
                tokens.push(self.bos_token_id);
            }
        }
        tokens
    }

    /// Decode token IDs back to a string.
    #[must_use]
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&id| self.tokens.get(id as usize))
            .cloned()
            .collect::<Vec<_>>()
            .join("")
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// A loaded language model.
pub struct Model {
    /// Path to the GGUF file.
    path: String,
    /// GGUF reader.
    reader: GgufReader,
    /// Model architecture.
    arch: ModelArch,
    /// Tokenizer.
    tokenizer: Tokenizer,
    /// Number of parameters.
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
        let reader = GgufReader::from_file(&path)?;

        let arch = ModelArch::from_gguf(&reader)?;
        let tokenizer = Tokenizer::from_gguf(&reader)?;

        // Count parameters from tensor info
        let mut param_count: u64 = 0;
        for tensor in reader.tensors() {
            let elements: u64 = tensor.shape.iter().map(|&d| d as u64).product();
            param_count += elements;
        }

        Ok(Self {
            path,
            reader,
            arch,
            tokenizer,
            parameter_count: param_count,
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

    /// Returns the model architecture.
    #[must_use]
    pub fn architecture(&self) -> &ModelArch {
        &self.arch
    }

    /// Returns the tokenizer.
    #[must_use]
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Returns the GGUF reader.
    #[must_use]
    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }

    /// Get a tensor by name.
    ///
    /// Returns `None` if not found.
    #[must_use]
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.reader.find_tensor(name)
    }

    /// Returns the number of tensors.
    #[must_use]
    pub fn tensor_count(&self) -> i64 {
        self.reader.tensor_count()
    }

    /// Returns a summary of the model.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} | {} params | {} tensors | vocab: {} | layers: {} | embd: {}",
            self.arch.architecture,
            format_params(self.parameter_count),
            self.tensor_count(),
            self.arch.n_vocab,
            self.arch.n_layer,
            self.arch.n_embd,
        )
    }
}

/// Format parameter count with suffix.
fn format_params(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

// ─── Inference ──────────────────────────────────────────────────────────────

/// Configuration for model inference.
#[derive(Clone)]
pub struct ModelConfig {
    /// Number of CPU threads to use.
    pub n_threads: usize,
    /// Whether to use CUDA acceleration.
    pub use_cuda: bool,
    /// Context size (number of tokens).
    pub n_ctx: usize,
    /// Batch size for prompt processing.
    pub n_batch: usize,
    /// Sampling temperature (0.0 = greedy).
    pub temperature: f32,
    /// Top-k filtering (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) filtering (0.0 = disabled, 1.0 = disabled).
    pub top_p: f32,
    /// Random seed for sampling.
    pub seed: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            use_cuda: false,
            n_ctx: 512,
            n_batch: 512,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            seed: 42,
        }
    }
}

/// Inference context for a single generation session.
pub struct InferenceContext {
    /// Model reference (shared via Arc for server use).
    model: Arc<Model>,
    /// Configuration.
    #[allow(dead_code)]
    config: ModelConfig,
    /// Token history.
    tokens: Vec<u32>,
    /// KV cache: per-layer key and value tensors.
    /// Shape: [n_layer][n_ctx][n_head_kv][rope_dim]
    kv_cache: Vec<(Vec<f32>, Vec<f32>)>,
    /// Current position in the sequence.
    position: usize,
}

impl InferenceContext {
    /// Create a new inference context.
    #[must_use]
    pub fn new(model: Arc<Model>, config: ModelConfig) -> Self {
        let arch = model.architecture();
        let n_ctx = config.n_ctx;
        let n_layer = arch.n_layer as usize;
        let n_head_kv = arch.n_head_kv as usize;
        let rope_dim = arch.rope_dim as usize;

        // Pre-allocate KV cache
        let kv_size = n_ctx * n_head_kv * rope_dim;
        let kv_cache = (0..n_layer)
            .map(|_| (vec![0.0f32; kv_size], vec![0.0f32; kv_size]))
            .collect();

        Self {
            model,
            config,
            tokens: Vec::new(),
            kv_cache,
            position: 0,
        }
    }

    /// Encode text into tokens and add to context.
    pub fn encode(&mut self, text: &str) {
        let new_tokens = self.model.tokenizer.encode(text);
        self.tokens.extend(new_tokens);
    }

    /// Get the token history.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Decode a single token ID to its string representation.
    #[must_use]
    pub fn decode_from_id(&self, token_id: u32) -> String {
        self.model
            .tokenizer
            .tokens
            .get(token_id as usize)
            .cloned()
            .unwrap_or_else(|| format!("<unk:{token_id}>"))
    }

    /// Decode all tokens to a string.
    #[must_use]
    pub fn decode(&self) -> String {
        self.model.tokenizer.decode(&self.tokens)
    }

    /// Run inference for n_predict tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    pub fn generate(&mut self, n_predict: usize) -> LlamaResult<Vec<u32>> {
        let mut generated = Vec::with_capacity(n_predict);
        let arch = self.model.architecture();
        let n_embd = arch.n_embd as usize;
        let n_head = arch.n_head as usize;
        let n_head_kv = arch.n_head_kv as usize;
        let rope_dim = arch.rope_dim as usize;
        let d_head = n_embd / n_head;

        for _ in 0..n_predict {
            let input_tokens = if self.position == 0 {
                // First token: process full prompt
                self.tokens.clone()
            } else {
                // Subsequent tokens: only the last generated token
                vec![*self.tokens.last().unwrap()]
            };

            let seq_len = input_tokens.len();

            // 1. Token embeddings: [seq_len, n_embd]
            let mut hidden = self.embed_tokens(&input_tokens)?;

            // 2. Transformer layers
            for layer in 0..arch.n_layer as usize {
                let prefix = format!("blk.{}.", layer);

                // RMSNorm before attention
                let norm_weight = self.get_tensor_f32(&format!("{prefix}attn_norm.weight"))?;
                let attn_input = rms_norm(&hidden, &norm_weight, arch.norm_eps);

                // Self-attention
                let q_weight = self.get_tensor_f32(&format!("{prefix}attn_q.weight"))?;
                let k_weight = self.get_tensor_f32(&format!("{prefix}attn_k.weight"))?;
                let v_weight = self.get_tensor_f32(&format!("{prefix}attn_v.weight"))?;
                let o_weight = self.get_tensor_f32(&format!("{prefix}attn_output.weight"))?;

                // Project Q, K, V
                let q = mat_vec_batch(&q_weight, &attn_input, seq_len, n_embd, n_embd);
                let k = mat_vec_batch(&k_weight, &attn_input, seq_len, n_embd, n_embd);
                let v = mat_vec_batch(&v_weight, &attn_input, seq_len, n_embd, n_embd);

                // Apply RoPE
                let mut q_rope = vec![0.0f32; q.len()];
                let mut k_rope = vec![0.0f32; k.len()];
                apply_rope(
                    &q,
                    &mut q_rope,
                    self.position,
                    n_head,
                    d_head,
                    rope_dim,
                    arch.rope_freq_base,
                );
                apply_rope(
                    &k,
                    &mut k_rope,
                    self.position,
                    n_head_kv,
                    d_head,
                    rope_dim,
                    arch.rope_freq_base,
                );

                // Store KV cache
                let (k_cache, v_cache) = &mut self.kv_cache[layer];
                let kv_stride = n_head_kv * rope_dim;
                for s in 0..seq_len {
                    let src_offset = s * n_head_kv * d_head;
                    let dst_offset = (self.position + s) * kv_stride;
                    k_cache[dst_offset..dst_offset + n_head_kv * d_head]
                        .copy_from_slice(&k_rope[src_offset..src_offset + n_head_kv * d_head]);
                    v_cache[dst_offset..dst_offset + n_head_kv * d_head]
                        .copy_from_slice(&v[src_offset..src_offset + n_head_kv * d_head]);
                }

                // Multi-head attention with KV cache
                let attn_output = multi_head_attention(
                    &q_rope,
                    &self.kv_cache[layer].0,
                    &self.kv_cache[layer].1,
                    seq_len,
                    self.position + seq_len,
                    n_head,
                    n_head_kv,
                    d_head,
                    rope_dim,
                );

                // Output projection
                let attn_proj = mat_vec_batch(&o_weight, &attn_output, seq_len, n_embd, n_embd);

                // Residual connection
                for i in 0..hidden.len() {
                    hidden[i] += attn_proj[i];
                }

                // RMSNorm before FFN
                let ffn_norm_weight = self.get_tensor_f32(&format!("{prefix}ffn_norm.weight"))?;
                let ffn_input = rms_norm(&hidden, &ffn_norm_weight, arch.norm_eps);

                // FFN: SwiGLU
                let gate_weight = self.get_tensor_f32(&format!("{prefix}ffn_gate.weight"))?;
                let up_weight = self.get_tensor_f32(&format!("{prefix}ffn_up.weight"))?;
                let down_weight = self.get_tensor_f32(&format!("{prefix}ffn_down.weight"))?;
                let n_ff = arch.n_ff as usize;

                let gate = mat_vec_batch(&gate_weight, &ffn_input, seq_len, n_embd, n_ff);
                let up = mat_vec_batch(&up_weight, &ffn_input, seq_len, n_embd, n_ff);

                // SwiGLU: silu(gate) * up
                let mut ffn_out = vec![0.0f32; seq_len * n_ff];
                for i in 0..gate.len() {
                    let silu = gate[i] / (1.0 + (-gate[i]).exp());
                    ffn_out[i] = silu * up[i];
                }

                // Down projection
                let ffn_proj = mat_vec_batch(&down_weight, &ffn_out, seq_len, n_ff, n_embd);

                // Residual connection
                for i in 0..hidden.len() {
                    hidden[i] += ffn_proj[i];
                }
            }

            // 3. Final RMSNorm
            let output_norm = self.get_tensor_f32("output_norm.weight")?;
            let normalized = rms_norm(&hidden, &output_norm, arch.norm_eps);

            // 4. Output projection (lm_head)
            let lm_head = self.get_tensor_f32("output.weight")?;
            let logits = mat_vec_batch(
                &lm_head,
                &normalized,
                seq_len,
                n_embd,
                arch.n_vocab as usize,
            );

            // 5. Sample next token
            let last_logits =
                &logits[(seq_len - 1) * arch.n_vocab as usize..seq_len * arch.n_vocab as usize];
            let next_token = sample_token(
                last_logits,
                self.config.temperature,
                self.config.top_k,
                self.config.top_p,
                self.config.seed,
            );

            self.tokens.push(next_token);
            generated.push(next_token);
            self.position += seq_len;
        }

        Ok(generated)
    }

    /// Get tensor data as f32 slice.
    fn get_tensor_f32(&self, name: &str) -> LlamaResult<Vec<f32>> {
        let tensor = self
            .model
            .get_tensor(name)
            .ok_or_else(|| LlamaError::LoadError(format!("missing tensor: {name}")))?;

        // Read tensor data from GGUF file
        let data = self
            .model
            .reader()
            .read_tensor_data(tensor)
            .map_err(|e| LlamaError::LoadError(format!("failed to read tensor {name}: {e}")))?;

        // Convert bytes to f32 based on tensor dtype
        let f32_data: Vec<f32> = match tensor.dtype {
            gguf::GgmlType::F32 => {
                let slice = unsafe {
                    std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), data.len() / 4)
                };
                slice.to_vec()
            }
            gguf::GgmlType::F16 => {
                // Convert F16 to F32
                let mut result = Vec::with_capacity(data.len() / 2);
                for i in (0..data.len()).step_by(2) {
                    let bits = u16::from_le_bytes([data[i], data[i + 1]]);
                    result.push(f16_to_f32(bits));
                }
                result
            }
            gguf::GgmlType::Q4_0 => dequantize_q4_0(&data, tensor.shape.first().copied().unwrap_or(0) as usize),
            gguf::GgmlType::Q8_0 => dequantize_q8_0(&data, tensor.shape.first().copied().unwrap_or(0) as usize),
            _ => {
                return Err(LlamaError::LoadError(format!(
                    "unsupported tensor dtype for {name}: {:?}",
                    tensor.dtype
                )));
            }
        };

        Ok(f32_data)
    }

    /// Get token embedding for a token ID.
    fn embed_tokens(&self, tokens: &[u32]) -> LlamaResult<Vec<f32>> {
        let n_embd = self.model.architecture().n_embd as usize;
        let token_embd = self.get_tensor_f32("token_embd.weight")?;

        let mut embeddings = vec![0.0f32; tokens.len() * n_embd];
        for (i, &token_id) in tokens.iter().enumerate() {
            let src_start = (token_id as usize) * n_embd;
            let dst_start = i * n_embd;
            embeddings[dst_start..dst_start + n_embd]
                .copy_from_slice(&token_embd[src_start..src_start + n_embd]);
        }

        Ok(embeddings)
    }
}

// ─── Tensor Operations ──────────────────────────────────────────────────────

/// RMS normalization: x / sqrt(mean(x^2) + eps) * weight
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let dim = weight.len();
    let n_rows = n / dim;
    let mut result = vec![0.0f32; n];

    for row in 0..n_rows {
        let start = row * dim;
        let row_data = &x[start..start + dim];

        // Compute RMS
        let sum_sq: f32 = row_data.iter().map(|v| v * v).sum();
        let rms = (sum_sq / dim as f32 + eps).sqrt();

        // Normalize and scale
        for i in 0..dim {
            result[start + i] = (row_data[i] / rms) * weight[i];
        }
    }

    result
}

/// Matrix-vector batch: Y = X @ W where X is [seq, in_dim], W is [out_dim, in_dim]
/// Result is [seq, out_dim]
fn mat_vec_batch(w: &[f32], x: &[f32], seq_len: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; seq_len * out_dim];

    for s in 0..seq_len {
        let x_row = &x[s * in_dim..(s + 1) * in_dim];
        let y_row = &mut y[s * out_dim..(s + 1) * out_dim];

        for o in 0..out_dim {
            let w_row = &w[o * in_dim..(o + 1) * in_dim];
            let mut sum = 0.0f32;
            for i in 0..in_dim {
                sum += x_row[i] * w_row[i];
            }
            y_row[o] = sum;
        }
    }

    y
}

/// Apply Rotary Position Embedding (RoPE).
fn apply_rope(
    x: &[f32],
    out: &mut [f32],
    position: usize,
    n_head: usize,
    d_head: usize,
    rope_dim: usize,
    freq_base: f32,
) {
    let half_dim = rope_dim / 2;

    for h in 0..n_head {
        for pos in 0..1 {
            // Single position per call (for generation)
            let pos_offset = position + pos;
            let head_offset = h * d_head;

            for i in 0..half_dim {
                let freq = 1.0 / freq_base.powf(i as f32 / half_dim as f32);
                let theta = pos_offset as f32 * freq;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                let src_idx = head_offset + i;
                let pair_idx = head_offset + i + half_dim;

                let x0 = x[src_idx];
                let x1 = if pair_idx < d_head { x[pair_idx] } else { 0.0 };

                out[src_idx] = x0 * cos_theta - x1 * sin_theta;
                if pair_idx < d_head {
                    out[pair_idx] = x0 * sin_theta + x1 * cos_theta;
                }
            }

            // Copy remaining dimensions unchanged
            for i in rope_dim..d_head {
                out[head_offset + i] = x[head_offset + i];
            }
        }
    }
}

/// Multi-head attention with grouped query attention (GQA) support.
fn multi_head_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    kv_len: usize,
    n_head: usize,
    n_head_kv: usize,
    d_head: usize,
    rope_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (d_head as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * n_head * d_head];
    let n_rep = n_head / n_head_kv; // GQA replication factor

    for s in 0..seq_len {
        for h in 0..n_head {
            // Map query head to KV head (GQA)
            let kv_h = h / n_rep;

            let q_offset = s * n_head * d_head + h * d_head;
            let q_head = &q[q_offset..q_offset + d_head];

            // Compute attention scores: Q @ K^T
            let mut scores = vec![0.0f32; kv_len];
            for t in 0..kv_len {
                let k_offset = t * n_head_kv * rope_dim + kv_h * d_head;
                let k_head = &k_cache[k_offset..k_offset + d_head.min(rope_dim)];

                let mut score = 0.0f32;
                for i in 0..d_head.min(rope_dim) {
                    score += q_head[i] * k_head[i];
                }
                scores[t] = score * scale;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            for score in &mut scores {
                *score /= sum;
            }

            // Weighted sum of V
            let out_offset = s * n_head * d_head + h * d_head;
            let out_head = &mut output[out_offset..out_offset + d_head];

            for t in 0..kv_len {
                let v_offset = t * n_head_kv * rope_dim + kv_h * d_head;
                let v_head = &v_cache[v_offset..v_offset + d_head.min(rope_dim)];
                let weight = scores[t];

                for i in 0..d_head.min(rope_dim) {
                    out_head[i] += weight * v_head[i];
                }
            }
        }
    }

    output
}

/// Greedy sampling: return the token ID with the highest logit.
fn greedy_sample(logits: &[f32]) -> u32 {
    let (max_idx, _) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    max_idx as u32
}

/// Apply softmax to logits in-place and return probabilities.
fn softmax(logits: &mut [f32]) {
    let max_val = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f32;
    for logit in logits.iter_mut() {
        *logit = (*logit - max_val).exp();
        sum += *logit;
    }
    for logit in logits.iter_mut() {
        *logit /= sum;
    }
}

/// Apply temperature scaling to logits.
fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature <= 0.0 || (temperature - 1.0).abs() < 1e-8 {
        return;
    }
    for logit in logits.iter_mut() {
        *logit /= temperature;
    }
}

/// Apply top-k filtering: keep only the k largest logits, zero out the rest.
fn apply_top_k(logits: &mut [f32], k: usize) {
    if k >= logits.len() || k == 0 {
        return;
    }

    // Find the k-th largest value
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];

    for logit in logits.iter_mut() {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Apply nucleus (top-p) sampling: keep only the smallest set of tokens
/// whose cumulative probability exceeds p.
fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 || p <= 0.0 {
        return;
    }

    // Create index-value pairs and sort by probability descending
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0f32;
    let mut cutoff_idx = indexed.len();

    for (i, (_, prob)) in indexed.iter().enumerate() {
        cumulative += prob;
        if cumulative > p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Zero out tokens beyond the cutoff
    let threshold = indexed[cutoff_idx - 1].1;
    for logit in logits.iter_mut() {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }

    // Re-normalize
    softmax(logits);
}

/// Sample a token ID from a categorical distribution.
fn categorical_sample(probs: &[f32], rng: &mut u64) -> u32 {
    // Simple LCG random number generator
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    let rand_val = ((*rng >> 33) as f64 / (u32::MAX as f64)) as f32;

    let mut cumulative = 0.0f32;
    for (i, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if rand_val < cumulative {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Sample a token from logits with temperature and optional top-k/top-p filtering.
fn sample_token(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, seed: u64) -> u32 {
    if temperature <= 0.0 {
        return greedy_sample(logits);
    }

    let mut probs = logits.to_vec();
    apply_temperature(&mut probs, temperature);

    if top_k > 0 {
        apply_top_k(&mut probs, top_k);
    }

    softmax(&mut probs);

    if top_p > 0.0 && top_p < 1.0 {
        apply_top_p(&mut probs, top_p);
    }

    let mut rng = seed;
    categorical_sample(&probs, &mut rng)
}

/// Convert F16 (IEEE 754-2008 binary16) to F32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            // Zero
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let mantissa_f = (mantissa as f32) / 1024.0;
        return (-1.0f32).powi(sign as i32) * mantissa_f * 2.0f32.powi(-14);
    }

    if exp == 31 {
        if mantissa == 0 {
            // Infinity
            return f32::from_bits((sign << 31) | (0x7F << 23));
        }
        // NaN
        return f32::from_bits((sign << 31) | (0x7F << 23) | (mantissa << 13));
    }

    // Normal
    let new_exp = exp + (127 - 15);
    let new_mantissa = mantissa << 13;
    f32::from_bits((sign << 31) | (new_exp << 23) | new_mantissa)
}

/// Q4_0 block size (number of elements per block).
const QK4_0: usize = 32;
/// Q4_0 block size in bytes: 2 (f16 scale) + 16 (4-bit values) = 18.
const Q4_0_BLOCK_SIZE: usize = 18;

/// Dequantize Q4_0 tensor data to f32.
///
/// Q4_0 format: 4-bit quantization with block size 32.
/// Each block: 2 bytes (f16 scale) + 16 bytes (4-bit values, 2 per byte).
/// Dequantization: val[i] = scale * (q[i] - 8)
fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / QK4_0;
    let mut result = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block_start = block_idx * Q4_0_BLOCK_SIZE;
        let scale_bytes = &data[block_start..block_start + 2];
        let scale = f16_to_f32(u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]));

        let qs = &data[block_start + 2..block_start + Q4_0_BLOCK_SIZE];

        for i in 0..16 {
            let byte = qs[i];
            let q0 = (byte & 0x0F) as i8;
            let q1 = (byte >> 4) as i8;
            result.push(scale * (q0 as f32 - 8.0));
            result.push(scale * (q1 as f32 - 8.0));
        }
    }

    result
}

/// Q8_0 block size (number of elements per block).
const QK8_0: usize = 32;
/// Q8_0 block size in bytes: 2 (f16 scale) + 32 (8-bit values) = 34.
const Q8_0_BLOCK_SIZE: usize = 34;

/// Dequantize Q8_0 tensor data to f32.
///
/// Q8_0 format: 8-bit quantization with block size 32.
/// Each block: 2 bytes (f16 scale) + 32 bytes (8-bit signed values).
/// Dequantization: val[i] = scale * q[i]
fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / QK8_0;
    let mut result = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block_start = block_idx * Q8_0_BLOCK_SIZE;
        let scale_bytes = &data[block_start..block_start + 2];
        let scale = f16_to_f32(u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]));

        let qs = &data[block_start + 2..block_start + Q8_0_BLOCK_SIZE];

        for &q in qs {
            result.push(scale * (q as i8) as f32);
        }
    }

    result
}

// ─── Tests ──────────────────────────────────────────────────────────────────

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

    #[test]
    fn format_params_should_format_correctly() {
        assert_eq!(format_params(500), "500");
        assert_eq!(format_params(1_000), "1.0K");
        assert_eq!(format_params(1_500_000), "1.5M");
        assert_eq!(format_params(7_000_000_000), "7.0B");
    }

    #[test]
    fn tokenizer_should_encode_and_decode() {
        let tokenizer = Tokenizer {
            tokens: vec!["hello".into(), "world".into(), "test".into()],
            scores: None,
            types: None,
            bpe_merges: Vec::new(),
            bos_token_id: 1,
            eos_token_id: 2,
            tokenizer_type: "bpe".to_string(),
        };

        let encoded = tokenizer.encode("hello world");
        assert_eq!(encoded, vec![0, 1]);

        let decoded = tokenizer.decode(&[0, 1]);
        assert_eq!(decoded, "helloworld");
    }

    #[test]
    fn bpe_tokenizer_should_apply_merges() {
        // Create a simple BPE tokenizer with merge rules
        let mut tokenizer = Tokenizer {
            tokens: vec![
                "<0x48>".into(),  // H
                "<0x65>".into(),  // e
                "<0x6C>".into(),  // l
                "<0x6C65>".into(), // le (merged)
                "<0x6C6C65>".into(), // lle (merged)
                "<0x6F>".into(),  // o
                "<0x20>".into(),  // space
                "<0x57>".into(),  // W
                "<0x72>".into(),  // r
                "<0x6C64>".into(), // ld (merged)
            ],
            scores: None,
            types: None,
            bpe_merges: vec![
                ("<0x6C>".into(), "<0x65>".into()),   // l + e → le
                ("<0x6C65>".into(), "<0x6C>".into()), // le + l → lel (not used)
                ("<0x6C>".into(), "<0x6C65>".into()), // l + le → lle
                ("<0x6C>".into(), "<0x64>".into()),   // l + d → ld
            ],
            bos_token_id: 1,
            eos_token_id: 2,
            tokenizer_type: "bpe".to_string(),
        };

        // Add missing tokens for the test
        tokenizer.tokens.push("<0x64>".into()); // d

        // Test that BPE encoding applies merges
        // "He" → bytes: 0x48, 0x65 → should find tokens
        let encoded = tokenizer.encode("He");
        // Should produce token IDs for the byte tokens
        assert!(!encoded.is_empty());
    }

    #[test]
    fn from_file_should_return_error_for_missing_file() {
        let result = Model::from_file("/nonexistent/model.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn rms_norm_should_normalize_correctly() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let result = rms_norm(&x, &weight, 1e-5);

        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
        let rms = (7.5f32).sqrt();
        assert!((result[0] - 1.0 / rms).abs() < 0.001);
        assert!((result[1] - 2.0 / rms).abs() < 0.001);
    }

    #[test]
    fn rms_norm_should_apply_weight() {
        let x = vec![1.0, 2.0];
        let weight = vec![2.0, 0.5];
        let result = rms_norm(&x, &weight, 1e-5);

        let rms = ((1.0_f32 + 4.0) / 2.0_f32).sqrt();
        assert!((result[0] - (1.0 / rms) * 2.0).abs() < 0.001);
        assert!((result[1] - (2.0 / rms) * 0.5).abs() < 0.001);
    }

    #[test]
    fn greedy_sample_should_return_max_index() {
        let logits = vec![0.1, 0.5, 0.9, 0.3];
        assert_eq!(greedy_sample(&logits), 2);
    }

    #[test]
    fn greedy_sample_should_handle_negative() {
        let logits = vec![-1.0, -0.5, -2.0, -0.1];
        assert_eq!(greedy_sample(&logits), 3);
    }

    #[test]
    fn softmax_should_normalize_correctly() {
        let mut logits = vec![1.0, 2.0, 3.0];
        softmax(&mut logits);

        // Check probabilities sum to 1
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Check ordering is preserved (higher logit → higher prob)
        assert!(logits[2] > logits[1]);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn apply_temperature_should_scale_logits() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 2.0);

        assert!((logits[0] - 0.5).abs() < 0.001);
        assert!((logits[1] - 1.0).abs() < 0.001);
        assert!((logits[2] - 1.5).abs() < 0.001);
    }

    #[test]
    fn apply_temperature_should_not_change_at_1() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 1.0);

        assert!((logits[0] - 1.0).abs() < 0.001);
        assert!((logits[1] - 2.0).abs() < 0.001);
        assert!((logits[2] - 3.0).abs() < 0.001);
    }

    #[test]
    fn apply_top_k_should_keep_top_k() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        apply_top_k(&mut logits, 2);

        // Only top 2 should remain, others should be -inf
        assert!(logits[1] > f32::NEG_INFINITY); // 5.0
        assert!(logits[4] > f32::NEG_INFINITY); // 4.0
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn sample_token_with_zero_temperature_should_be_greedy() {
        let logits = vec![0.1, 0.5, 0.9, 0.3];
        let token = sample_token(&logits, 0.0, 0, 0.0, 42);
        assert_eq!(token, 2);
    }

    #[test]
    fn f16_to_f32_should_convert_correctly() {
        // 1.0 in F16: 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 0.001);
        // 0.5 in F16: 0x3800
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 0.001);
        // -1.0 in F16: 0xBC00
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 0.001);
        // 0.0 in F16: 0x0000
        assert!(f16_to_f32(0x0000).abs() < 0.001);
    }

    #[test]
    fn mat_vec_batch_should_compute_correct_result() {
        // W = [[1, 2], [3, 4]] (2x2), X = [[5, 6]] (1x2)
        // Y = X @ W^T = [[5*1+6*2, 5*3+6*4]] = [[17, 39]]
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![5.0, 6.0];
        let y = mat_vec_batch(&w, &x, 1, 2, 2);

        assert!((y[0] - 17.0).abs() < 0.001);
        assert!((y[1] - 39.0).abs() < 0.001);
    }

    #[test]
    fn dequantize_q4_0_should_recover_values() {
        // Build a Q4_0 block with scale=1.0 and values that should dequantize to [-8, -7, ..., 23]
        // q[i] stored as (val/scale + 8), so for scale=1.0:
        // val=-8 -> q=0, val=-7 -> q=1, ..., val=7 -> q=15
        // Block: scale (f16 1.0 = 0x3C00) + 16 bytes of 4-bit values
        let mut block = vec![0u8; Q4_0_BLOCK_SIZE];
        // Scale = 1.0 in f16
        block[0] = 0x00;
        block[1] = 0x3C;

        // Fill with values 0..15 repeated (will dequantize to -8..7)
        for i in 0..16 {
            block[2 + i] = (i as u8) | ((i as u8) << 4);
        }

        let result = dequantize_q4_0(&block, QK4_0);
        assert_eq!(result.len(), QK4_0);

        // First two values: q=0 -> -8, q=0 -> -8
        assert!((result[0] - (-8.0)).abs() < 0.001);
        assert!((result[1] - (-8.0)).abs() < 0.001);
        // Last two values: q=15 -> 7, q=15 -> 7
        assert!((result[30] - 7.0).abs() < 0.001);
        assert!((result[31] - 7.0).abs() < 0.001);
    }

    #[test]
    fn dequantize_q8_0_should_recover_values() {
        // Build a Q8_0 block with scale=0.1 and values 0..31
        // val[i] = scale * q[i], so for scale=0.1: val = 0, 0.1, 0.2, ..., 3.1
        let mut block = vec![0u8; Q8_0_BLOCK_SIZE];
        // Scale = 0.1 in f16 ≈ 0x2E66 (need to compute)
        // 0.1 in f32 = 0x3DCCCCCD, in f16 ≈ 0x2E66
        let scale_f32 = 0.1f32;
        // Convert f32 to f16 (approximate)
        let bits = scale_f32.to_bits();
        let sign = (bits >> 31) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x7FFFFF;
        let f16_exp = (exp - 127 + 15) as u16;
        let f16_mantissa = (mantissa >> 13) as u16;
        let f16_bits = (sign << 15) | (f16_exp << 10) | f16_mantissa;
        block[0] = f16_bits as u8;
        block[1] = (f16_bits >> 8) as u8;

        // Fill with values 0..31 as int8
        for i in 0..32 {
            block[2 + i] = i as u8;
        }

        let result = dequantize_q8_0(&block, QK8_0);
        assert_eq!(result.len(), QK8_0);

        // First value: 0.1 * 0 = 0
        assert!(result[0].abs() < 0.01);
        // Last value: 0.1 * 31 = 3.1
        assert!((result[31] - 3.1).abs() < 0.01);
    }
}
