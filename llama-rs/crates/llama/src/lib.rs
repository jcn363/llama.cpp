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

use gguf::{GgufReader, TensorInfo};
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
                    _ => Err(LlamaError::LoadError(format!(
                        "key {key} has wrong type"
                    ))),
                })
        };

        let get_f32 = |key: &str| -> LlamaResult<f32> {
            reader
                .get_kv(key)
                .ok_or_else(|| LlamaError::LoadError(format!("missing key: {key}")))
                .and_then(|v| match v {
                    gguf::GgufValue::F32(val) => Ok(*val),
                    gguf::GgufValue::F64(val) => Ok(*val as f32),
                    _ => Err(LlamaError::LoadError(format!(
                        "key {key} has wrong type"
                    ))),
                })
        };

        let get_str = |key: &str| -> LlamaResult<String> {
            reader
                .get_kv(key)
                .ok_or_else(|| LlamaError::LoadError(format!("missing key: {key}")))
                .and_then(|v| match v {
                    gguf::GgufValue::Str(val) => Ok(val.clone()),
                    _ => Err(LlamaError::LoadError(format!(
                        "key {key} has wrong type"
                    ))),
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
        let norm_eps = get_f32(&format!("{prefix}attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let rope_freq_base =
            get_f32(&format!("{prefix}rope.freq_base")).unwrap_or(10_000.0);
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

        Ok(Self {
            tokens,
            scores,
            types,
        })
    }

    /// Encode a string into token IDs (simple exact match).
    ///
    /// TODO: Implement proper BPE/SPM tokenization.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Simple fallback: find exact token matches
        let mut tokens = Vec::new();
        for word in text.split_whitespace() {
            if let Some(id) = self.tokens.iter().position(|t| t == word) {
                tokens.push(id as u32);
            } else {
                // Unknown token — use BOS or fallback
                tokens.push(1); // BOS token
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

/// Inference context for a single generation session.
pub struct InferenceContext {
    /// Model reference.
    model: Model,
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
    pub fn new(model: Model, config: ModelConfig) -> Self {
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
            let logits = mat_vec_batch(&lm_head, &normalized, seq_len, n_embd, arch.n_vocab as usize);

            // 5. Sample next token (greedy for now)
            let last_logits = &logits[(seq_len - 1) * arch.n_vocab as usize..seq_len * arch.n_vocab as usize];
            let next_token = greedy_sample(last_logits);

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

        // Convert bytes to f32 (assuming F32 or F16)
        let f32_data: Vec<f32> = match tensor.dtype {
            gguf::GgmlType::F32 => {
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr().cast::<f32>(),
                        data.len() / 4,
                    )
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
                let x1 = if pair_idx < d_head {
                    x[pair_idx]
                } else {
                    0.0
                };

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
        };

        let encoded = tokenizer.encode("hello world");
        assert_eq!(encoded, vec![0, 1]);

        let decoded = tokenizer.decode(&[0, 1]);
        assert_eq!(decoded, "helloworld");
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
}
