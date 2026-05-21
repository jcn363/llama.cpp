
/// Compute dot product of two slices (SIMD‑friendly unrolled version).

/// Model representation and loading logic.
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

mod kv_cache;
mod attention;
mod inference;
pub mod tokenizer;

use crate::kv_cache::KvCacheManager;
use crate::attention::multi_head_attention_with_cache;
use crate::inference::{embed_token, rms_norm, mat_vec, mul_vec, add_vec, silu, sample_logits, SamplingConfig};
pub use crate::tokenizer::SimpleTokenizer;
use gguf::{GgufReader, TensorInfo, MmapTensor, GgufError, GgufValue};
use rayon::prelude::*;

/// Simple struct to hold a tensor that can be lazily de‑quantized.
/// Uses memory-mapped access: raw data stays on disk until dequantization.
#[derive(Debug)]
pub struct TensorData {
    /// Memory-mapped reference to the tensor's raw (quantized) data.
    pub mmap_tensor: MmapTensor,
    /// Tensor metadata needed for de‑quantization.
    pub info: TensorInfo,
    /// De‑quantized float values – filled on first access.
    pub data: RwLock<Option<Arc<[f32]>>>,
    /// Shape of the tensor (rows, cols) for 2‑D tensors; empty for scalars.
    pub shape: Vec<usize>,
}

impl TensorData {
    /// Return the de‑quantized data, performing lazy de‑quantization if needed.
    pub fn get(&self) -> Result<Arc<[f32]>, GgufError> {
        // Fast path: already de‑quantized.
        if let Some(ref d) = *self.data.read().unwrap() {
            return Ok(d.clone());
        }
        // Need to de‑quantize from mmap.
        let deq = self.mmap_tensor.dequantize(&self.info)?;
        let arc: Arc<[f32]> = Arc::from(deq.into_boxed_slice());
        // Store for future calls.
        let mut write = self.data.write().unwrap();
        *write = Some(arc.clone());
        Ok(arc)
    }
}


/// Simple interner for strings used throughout the model (e.g., tensor names).
#[derive(Debug, Default)]
pub struct InternedStrings {
    /// Vector of unique strings; index is the interned ID.
    strings: Vec<String>,
    /// Reverse map for fast lookup.
    map: HashMap<String, usize>,
}

impl InternedStrings {
    /// Intern a string, returning its unique ID.
    pub fn intern(&mut self, s: &str) -> usize {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.strings.len();
        self.strings.push(s.to_owned());
        self.map.insert(s.to_owned(), id);
        id
    }
    /// Retrieve a string by its ID.
    pub fn get(&self, id: usize) -> Option<&str> {
        self.strings.get(id).map(|s| s.as_str())
    }
}

/// The core model struct.
#[derive(Debug)]
pub struct Model {
    /// Mapping from interned tensor ID to its data.
    pub tensors: HashMap<usize, TensorData>,
    /// Interner for tensor names and other strings.
    pub interned: InternedStrings,
    /// Model hyper‑parameters extracted from GGUF metadata.
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub d_head: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub n_ff: usize,
    pub n_layers: usize,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Tokenizer vocabulary loaded from GGUF metadata.
    pub vocab_tokens: Vec<String>,
    /// Tokenizer scores (for BPE ranking).
    pub vocab_scores: Vec<f32>,
    /// Tokenizer token types.
    pub vocab_types: Vec<tokenizer::TokenType>,
    /// BOS token ID.
    pub bos_token_id: usize,
    /// EOS token ID.
    pub eos_token_id: usize,
    /// Unknown token ID.
    pub unk_token_id: usize,
    /// Whether to add BOS token automatically.
    pub add_bos_token: bool,
    /// KV cache used during inference (one per layer).
    pub kv_cache: RwLock<KvCacheManager>,
}


/// Configuration for inference.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_threads: usize,
    pub use_cuda: bool,
    pub n_ctx: usize,
    pub n_batch: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_threads: 4,
            use_cuda: false,
            n_ctx: 2048,
            n_batch: 512,
        }
    }
}

/// Inference context holding state for a model.
#[derive(Debug)]
pub struct InferenceContext {
    pub model: Arc<Model>,
    pub config: ModelConfig,
    pub tokenizer: SimpleTokenizer,
    pub sampling: SamplingConfig,
}

impl InferenceContext {
    /// Create a new inference context.
    pub fn new(model: Arc<Model>, config: ModelConfig) -> Self {
        // Create tokenizer from model's vocabulary
        let tokenizer = SimpleTokenizer::from_gguf_vocab(
            model.vocab_tokens.clone(),
            model.vocab_scores.clone(),
            model.vocab_types.clone(),
            model.bos_token_id,
            model.eos_token_id,
            model.unk_token_id,
            model.add_bos_token,
        );
        Self { model, config, tokenizer, sampling: SamplingConfig::default() }
    }
    
    /// Set the sampling configuration.
    pub fn with_sampling(mut self, sampling: SamplingConfig) -> Self {
        self.sampling = sampling;
        self
    }
    /// Encode input text to token ids using the tokenizer.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.tokenizer.encode(text)
    }

    /// Generate token IDs for a prompt using actual inference.
    /// 
    /// This implementation:
    /// 1. Encodes the prompt to token IDs
    /// 2. For each predicted token, runs a forward pass through the model
    /// 3. Samples the next token using temperature/top-k/top-p sampling
    pub fn generate(&self, prompt: &str, n_predict: usize) -> anyhow::Result<Vec<usize>> {
        let mut toks = self.encode(prompt);
        
        // Limit to context size
        if toks.len() > self.config.n_ctx {
            toks.truncate(self.config.n_ctx);
        }
        
        // Generate new tokens
        for _i in 0..n_predict {
            // Get the last token
            let last_token = *toks.last().unwrap_or(&0);
            
            // Run forward pass to get logits for next token
            match self.forward_pass(last_token) {
                Ok(logits) => {
                    // Sample next token using configured sampling
                    let next_token = sample_logits(&logits, &self.sampling);
                    toks.push(next_token);
                    
                    // Stop if we hit end-of-sequence token
                    if next_token == self.model.eos_token_id {
                        break;
                    }
                }
                Err(_) => {
                    // If forward pass fails, just pad with 0
                    toks.push(0);
                }
            }
        }
        
        Ok(toks)
    }
    
    /// Run a single forward pass through the model for a given token.
    /// Returns logits of shape (vocab_size,).
    fn forward_pass(&self, token_id: usize) -> anyhow::Result<Vec<f32>> {
        // Get embedding for this token
        let token_embd = self.model.get_tensor("token_embd.weight")?;
        let mut x = embed_token(token_id, &token_embd, self.model.n_embd)?;
        
        let n_layers = self.model.n_layers();
        if n_layers == 0 {
            return Ok(vec![0.0; self.model.vocab_size]);
        }
        
        let n_head = self.model.n_head;
        let n_head_kv = self.model.n_head_kv;
        let head_dim = self.model.d_head;
        let n_embd = self.model.n_embd;
        let rope_theta = self.model.rope_theta;
        
        // Apply each transformer block
        for layer_idx in 0..n_layers {
            // Save residual
            let residual = x.clone();
            
            // Attention norm
            let attn_norm_name = format!("blk.{}.attn_norm.weight", layer_idx);
            if let Ok(attn_norm_weight) = self.model.get_tensor(&attn_norm_name) {
                x = rms_norm(&x, &attn_norm_weight, 1e-5);
            }
            
            // Get Q, K, V projection weights
            let q_proj_name = format!("blk.{}.attn_q.weight", layer_idx);
            let k_proj_name = format!("blk.{}.attn_k.weight", layer_idx);
            let v_proj_name = format!("blk.{}.attn_v.weight", layer_idx);
            
            if let (Ok(q_weight), Ok(k_weight), Ok(v_weight)) = (
                self.model.get_tensor(&q_proj_name),
                self.model.get_tensor(&k_proj_name),
                self.model.get_tensor(&v_proj_name),
            ) {
                // Project x to Q, K, V
                // Q: (n_embd, n_head * head_dim) @ x -> (n_head * head_dim)
                let mut q = mat_vec(&q_weight, n_head * head_dim, n_embd, &x);
                let mut k = mat_vec(&k_weight, n_head_kv * head_dim, n_embd, &x);
                let v = mat_vec(&v_weight, n_head_kv * head_dim, n_embd, &x);
                
                // Get current position in KV cache
                let kv_cache = self.model.kv_cache.write().unwrap();
                let position_offset = kv_cache.get_layer_ref(layer_idx).cur_len;
                drop(kv_cache);
                
                // Apply attention with KV cache
                let mut kv_cache = self.model.kv_cache.write().unwrap();
                let attn_output = multi_head_attention_with_cache(
                    n_head,
                    n_head_kv,
                    head_dim,
                    1, // seq_len = 1 for single token generation
                    position_offset,
                    &mut q,
                    &mut k,
                    &v,
                    kv_cache.get_layer(layer_idx),
                    rope_theta,
                );
                drop(kv_cache);
                
                // Output projection
                let attn_out_name = format!("blk.{}.attn_output.weight", layer_idx);
                if let Ok(attn_out_weight) = self.model.get_tensor(&attn_out_name) {
                    let attn_proj = mat_vec(&attn_out_weight, n_embd, n_head * head_dim, &attn_output);
                    x = add_vec(&residual, &attn_proj);
                } else {
                    x = add_vec(&residual, &attn_output);
                }
            } else {
                // If QKV weights not found, just use residual
                x = residual;
            }
            
            // Save residual for FFN
            let ffn_residual = x.clone();
            
            // FFN norm
            let ffn_norm_name = format!("blk.{}.ffn_norm.weight", layer_idx);
            if let Ok(ffn_norm_weight) = self.model.get_tensor(&ffn_norm_name) {
                x = rms_norm(&x, &ffn_norm_weight, 1e-5);
            }
            
            // Apply SwiGLU FFN: FFN(x) = (silu(gate @ x) * up @ x) @ down
            let gate_name = format!("blk.{}.ffn_gate.weight", layer_idx);
            let up_name = format!("blk.{}.ffn_up.weight", layer_idx);
            let down_name = format!("blk.{}.ffn_down.weight", layer_idx);
            
            if let (Ok(gate), Ok(up), Ok(down)) = (
                self.model.get_tensor(&gate_name),
                self.model.get_tensor(&up_name),
                self.model.get_tensor(&down_name),
            ) {
                // gate_proj = gate @ x
                let gate_proj = mat_vec(&gate, self.model.n_ff, n_embd, &x);
                // up_proj = up @ x
                let up_proj = mat_vec(&up, self.model.n_ff, n_embd, &x);
                // silu(gate_proj) * up_proj
                let silu_gate = silu(&gate_proj);
                let ffn_hidden = mul_vec(&silu_gate, &up_proj);
                // down_proj = down @ ffn_hidden
                let ffn_output = mat_vec(&down, n_embd, self.model.n_ff, &ffn_hidden);
                // Residual connection
                x = add_vec(&ffn_residual, &ffn_output);
            } else {
                // If FFN tensors not found, just use residual
                x = ffn_residual;
            }
        }
        
        // Final norm
        if let Ok(final_norm) = self.model.get_tensor("output_norm.weight") {
            x = rms_norm(&x, &final_norm, 1e-5);
        }
        
        // Output projection to logits
        if let Ok(output_weight) = self.model.get_tensor("output.weight") {
            let logits = mat_vec(&output_weight, self.model.vocab_size, n_embd, &x);
            Ok(logits)
        } else {
            // Tied embeddings: use token_embd.weight as output
            let logits = mat_vec(&token_embd, self.model.vocab_size, n_embd, &x);
            Ok(logits)
        }
    }

    /// Decode a single token id to string.
    pub fn decode_from_id(&self, id: usize) -> String {
        self.tokenizer.decode(&[id])
    }

    /// Decode a slice of token ids to a string.
    pub fn decode(&self, ids: &[usize]) -> String {
        self.tokenizer.decode(ids)
    }
}



impl Model {
    /// Return a short summary string for debugging.
    pub fn summary(&self) -> String {
        format!(
            "Model: embd={}, heads={}, kv_heads={}, d_head={}, layers={}, seq_len={}, rope_theta={}",
            self.n_embd, self.n_head, self.n_head_kv, self.d_head, self.n_layers, self.max_seq_len, self.rope_theta
        )
    }

    /// Retrieve a tensor by name, returning de-quantized data.
    pub fn get_tensor(&self, name: &str) -> Result<Arc<[f32]>, GgufError> {
        // Find the tensor ID by searching through interned strings
        let id = self.interned.strings.iter().position(|s| s == name)
            .ok_or_else(|| GgufError::DecodeError(format!("Tensor not found: {}", name)))?;
        self.tensors
            .get(&id)
            .ok_or_else(|| GgufError::DecodeError(format!("Tensor not found: {}", name)))?
            .get()
    }

    /// Retrieve a tensor by name, returning its shape.
    pub fn get_tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        let id = self.interned.strings.iter().position(|s| s == name)?;
        self.tensors.get(&id).map(|t| t.shape.clone())
    }

    /// Return the number of transformer blocks in the model.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
}

impl Model {
    /// Load a model from a GGUF file, reading all tensors in parallel and
    /// de‑quantizing them eagerly. This is the primary entry point used by the
    /// CLI and server binaries.
    pub fn load_from_gguf<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        // 1️⃣ Open the GGUF file and parse the header.
        let reader = GgufReader::from_file(&path)?;

        // 2️⃣ Extract required hyper‑parameters from the metadata.
        //    Missing entries will cause an error – the model cannot be used.
        //    Try both `general.*` and `llama.*` key naming conventions.
        let n_embd = reader.get_usize_any(&[
            "general.embedding_length",
            "llama.embedding_length",
        ])?;
        let n_head = reader.get_usize_any(&[
            "general.attention_head_count",
            "llama.attention.head_count",
        ])?;
        let n_head_kv = reader.get_usize_any(&[
            "general.attention_head_count_kv",
            "llama.attention.head_count_kv",
        ])?;
        let d_head = reader.get_usize_any(&[
            "general.attention_head_dim",
            "llama.rope.dimension_count",
        ])?;
        let max_seq_len = reader.get_usize_any(&[
            "general.context_length",
            "llama.context_length",
        ])?;
        let vocab_size = reader.get_usize_any(&[
            "general.vocab_size",
            "llama.vocab_size",
        ])?;
        // n_ff is optional; if not found, estimate as 4 * n_embd (common default)
        let n_ff = reader.get_usize_any(&[
            "llama.feed_forward_length",
            "general.feed_forward_length",
            "llama.intermediate_size",
        ]).unwrap_or(n_embd * 4);
        // n_layers is required
        let n_layers = reader.get_usize_any(&[
            "llama.block_count",
            "general.block_count",
        ])?;
        // rope_theta is optional, default to 10000.0
        let rope_theta = match reader.get_kv("llama.rope.freq_base") {
            Some(GgufValue::F32(v)) => *v,
            Some(GgufValue::F64(v)) => *v as f32,
            _ => 10000.0,
        };

        // 3️⃣ Create memory-mapped tensor references for lazy loading.
        //    Tensor data stays on disk until dequantization is needed.
        //    The OS pages in only the accessed regions, reducing memory usage.
        let interned = std::sync::Arc::new(std::sync::Mutex::new(InternedStrings::default()));
        let shared_mmap = reader.mmap_arc().clone();
        let tensors: HashMap<usize, TensorData> = reader
            .tensors()
            .par_iter()
            .map(|info| {
                // Create mmap reference (no data copied yet).
                let mmap_tensor = reader.mmap_tensor(info, shared_mmap.clone())?;
                let shape = info.shape.iter().map(|&d| d as usize).collect();
                // Intern the name safely.
                let mut guard = interned.lock().unwrap();
                let id = guard.intern(&info.name);
                drop(guard);
                Ok((id, TensorData { mmap_tensor, info: info.clone(), data: RwLock::new(None), shape }))
            })
            .collect::<Result<HashMap<_, _>, GgufError>>()?;
        // Extract the interner out of the Arc.
        let interned = std::sync::Arc::try_unwrap(interned)
            .expect("no other references to interner")
            .into_inner()
            .unwrap();

        // 5️⃣ Initialise the KV cache (one per layer).
        let kv_cache = RwLock::new(KvCacheManager::new(n_layers, max_seq_len, n_head_kv, d_head));

        // 6️⃣ Extract tokenizer data from GGUF metadata.
        //    Try both naming conventions and provide defaults.
        let vocab_tokens = reader.get_string_array("tokenizer.ggml.tokens")
            .unwrap_or_else(|_| (0..vocab_size).map(|i| format!("<token{}>", i)).collect());
        let vocab_scores = reader.get_f32_array("tokenizer.ggml.scores")
            .unwrap_or_else(|_| vec![0.0; vocab_size]);
        let vocab_types_raw = reader.get_i32_array("tokenizer.ggml.token_type")
            .unwrap_or_else(|_| vec![1; vocab_size]);
        let vocab_types: Vec<tokenizer::TokenType> = vocab_types_raw
            .iter()
            .map(|&v| tokenizer::TokenType::from_i32(v))
            .collect();
        let bos_token_id = reader.get_usize_any(&[
            "tokenizer.ggml.bos_token_id",
        ]).unwrap_or(1);
        let eos_token_id = reader.get_usize_any(&[
            "tokenizer.ggml.eos_token_id",
        ]).unwrap_or(2);
        let unk_token_id = reader.get_usize_any(&[
            "tokenizer.ggml.unknown_token_id",
        ]).unwrap_or(0);
        let add_bos_token = match reader.get_kv("tokenizer.ggml.add_bos_token") {
            Some(GgufValue::Bool(b)) => *b,
            _ => true, // Default to adding BOS for Llama models
        };

        Ok(Self {
            tensors,
            interned,
            n_embd,
            n_head,
            n_head_kv,
            d_head,
            max_seq_len,
            vocab_size,
            n_ff,
            n_layers,
            rope_theta,
            vocab_tokens,
            vocab_scores,
            vocab_types,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            add_bos_token,
            kv_cache,
        })
    }

    /// Backwards‑compatible wrapper used by existing code.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        Self::load_from_gguf(path)
    }
}

#[inline(always)]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    let chunks = len / 4;
    for i in 0..chunks {
        let base = i * 4;
        sum += a[base] * b[base]
            + a[base + 1] * b[base + 1]
            + a[base + 2] * b[base + 2]
            + a[base + 3] * b[base + 3];
    }
    for i in (chunks * 4)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Perform a batched matrix‑vector multiplication.
///
/// `mat` is a row‑major matrix of shape `(rows, cols)` stored as a flat slice.
/// `vec` is a column vector of length `cols`.
/// The result is a vector of length `rows`.
///
/// This implementation uses the SIMD‑friendly `dot_product` for each row and
/// parallelises the computation across rows with Rayon.
#[inline(always)]
pub fn mat_vec_batch(mat: &[f32], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    use rayon::prelude::*;
    assert_eq!(mat.len(), rows * cols);
    assert_eq!(vec.len(), cols);
    (0..rows)
        .into_par_iter()
        .map(|r| {
            let start = r * cols;
            let row = &mat[start..start + cols];
            dot_product(row, vec)
        })
        .collect()
}
