/// Inference operations: embedding lookup, forward pass, sampling.
use gguf::GgufError;

/// Look up embedding for a single token ID.
/// 
/// `embeddings` is the token embedding matrix of shape (vocab_size, embed_dim).
/// Returns a vector of length `embed_dim`.
pub fn embed_token(token_id: usize, embeddings: &[f32], embed_dim: usize) -> Result<Vec<f32>, GgufError> {
    let start = token_id * embed_dim;
    let end = start + embed_dim;
    if end > embeddings.len() {
        return Err(GgufError::DecodeError(format!(
            "Token ID {} out of range for embedding matrix",
            token_id
        )));
    }
    Ok(embeddings[start..end].to_vec())
}

/// Apply RMSNorm to a vector.
/// 
/// RMSNorm(x) = x / RMS(x) * weight, where RMS(x) = sqrt(mean(x^2) + eps)
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len(), "RMSNorm: dimension mismatch");
    
    // Compute RMS
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let rms = (mean_sq + eps).sqrt();
    
    // Normalize and scale
    x.iter()
        .zip(weight.iter())
        .map(|(v, w)| (v / rms) * w)
        .collect()
}

/// Apply SiLU (Swish) activation: x * sigmoid(x)
pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|v| {
            let sigmoid = 1.0 / (1.0 + (-v).exp());
            v * sigmoid
        })
        .collect()
}

/// Compute softmax over a vector.
#[allow(dead_code)]
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = x.iter().map(|v| (v - max).exp()).sum();
    x.iter().map(|v| (v - max).exp() / exp_sum).collect()
}

/// Sample the next token from logits using argmax (greedy sampling).
pub fn sample_argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Matrix-vector multiplication: y = mat @ vec
/// 
/// `mat` is row-major of shape (rows, cols).
/// `vec` is of length cols.
/// Returns a vector of length rows.
/// 
/// Optimized with:
/// - Parallel execution across rows using Rayon
/// - SIMD-friendly 4-wide unrolling in dot product
/// - Cache-friendly sequential row access
pub fn mat_vec(mat: &[f32], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    use rayon::prelude::*;
    assert_eq!(mat.len(), rows * cols);
    assert_eq!(vec.len(), cols);
    
    // For small matrices, sequential is faster due to overhead
    if rows < 64 {
        (0..rows)
            .map(|r| {
                let start = r * cols;
                let row = &mat[start..start + cols];
                dot_product(row, vec)
            })
            .collect()
    } else {
        // Parallel for larger matrices
        (0..rows)
            .into_par_iter()
            .map(|r| {
                let start = r * cols;
                let row = &mat[start..start + cols];
                dot_product(row, vec)
            })
            .collect()
    }
}

/// Optimized dot product with 4-wide SIMD-friendly unrolling.
#[inline(always)]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    
    // 8-wide unrolling for better SIMD vectorization
    let chunks = len / 8;
    for i in 0..chunks {
        let base = i * 8;
        sum += a[base] * b[base]
            + a[base + 1] * b[base + 1]
            + a[base + 2] * b[base + 2]
            + a[base + 3] * b[base + 3]
            + a[base + 4] * b[base + 4]
            + a[base + 5] * b[base + 5]
            + a[base + 6] * b[base + 6]
            + a[base + 7] * b[base + 7];
    }
    
    // Handle remaining elements
    for i in (chunks * 8)..len {
        sum += a[i] * b[i];
    }
    
    sum
}

/// Element-wise multiplication of two vectors.
/// Parallelized for vectors > 1024 elements.
pub fn mul_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    use rayon::prelude::*;
    assert_eq!(a.len(), b.len());
    let len = a.len();
    
    if len < 1024 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    } else {
        let mut result = vec![0.0f32; len];
        result.par_iter_mut()
            .zip(a.par_iter().zip(b.par_iter()))
            .for_each(|(out, (x, y))| *out = x * y);
        result
    }
}

/// Add two vectors element-wise.
/// Parallelized for vectors > 1024 elements.
pub fn add_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    use rayon::prelude::*;
    assert_eq!(a.len(), b.len());
    let len = a.len();
    
    if len < 1024 {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    } else {
        let mut result = vec![0.0f32; len];
        result.par_iter_mut()
            .zip(a.par_iter().zip(b.par_iter()))
            .for_each(|(out, (x, y))| *out = x + y);
        result
    }
}

/// Multiply a vector by a scalar.
#[allow(dead_code)]
pub fn scale_vec(v: &[f32], scale: f32) -> Vec<f32> {
    use rayon::prelude::*;
    let len = v.len();
    
    if len < 1024 {
        v.iter().map(|x| x * scale).collect()
    } else {
        let mut result = vec![0.0f32; len];
        result.par_iter_mut()
            .zip(v.par_iter())
            .for_each(|(out, x)| *out = x * scale);
        result
    }
}

/// Apply temperature to logits and compute softmax.
///
/// Temperature controls the randomness of the distribution:
/// - `temp < 1`: More confident/deterministic
/// - `temp = 1`: Original distribution
/// - `temp > 1`: More random/diverse
///
/// Returns probabilities that sum to 1.
pub fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    
    let temp = if temperature <= 0.0 { 1e-6 } else { temperature };
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    let mut probs = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    
    for &v in logits {
        let exp_val = ((v - max) / temp).exp();
        probs.push(exp_val);
        sum += exp_val;
    }
    
    for p in &mut probs {
        *p /= sum;
    }
    
    probs
}

/// Apply top-k filtering to logits.
///
/// Keeps only the k highest probability tokens, setting all others to -infinity.
/// If k >= vocab_size, returns logits unchanged.
pub fn apply_top_k(logits: &[f32], k: usize) -> Vec<f32> {
    if k >= logits.len() || k == 0 {
        return logits.to_vec();
    }
    
    let mut result = logits.to_vec();
    
    // Find the k-th largest value
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];
    
    // Set all values below threshold to -infinity
    for v in &mut result {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
    
    result
}

/// Apply top-p (nucleus) filtering to logits.
///
/// Keeps only the smallest set of tokens whose cumulative probability exceeds p.
/// Returns filtered logits with non-selected tokens set to -infinity.
pub fn apply_top_p(logits: &[f32], p: f32) -> Vec<f32> {
    if p >= 1.0 || p <= 0.0 {
        return logits.to_vec();
    }
    
    // Create index-value pairs and sort by value descending
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // Compute cumulative sum and find cutoff
    let mut cumsum = 0.0f32;
    let mut cutoff_idx = indexed.len();
    
    // First compute softmax for cumulative probability
    let max = indexed.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max).exp()).sum();
    
    for (i, (_, v)) in indexed.iter().enumerate() {
        let prob = (v - max).exp() / exp_sum;
        cumsum += prob;
        if cumsum > p {
            cutoff_idx = i + 1;
            break;
        }
    }
    
    // Create filtered logits
    let mut result = vec![f32::NEG_INFINITY; logits.len()];
    for (idx, _) in indexed.iter().take(cutoff_idx) {
        result[*idx] = logits[*idx];
    }
    
    result
}

/// Sample from a categorical distribution.
///
/// Uses a simple linear search through cumulative probabilities.
/// `probs` should sum to 1.0.
pub fn sample_categorical(probs: &[f32], rng: &mut fastrand::Rng) -> usize {
    let rand_val = rng.f32();
    let mut cumsum = 0.0f32;
    
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rand_val <= cumsum {
            return i;
        }
    }
    
    // Fallback to last token (shouldn't happen with valid probabilities)
    probs.len() - 1
}

/// Sampling configuration for text generation.
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    /// Temperature for softmax (default: 0.8).
    pub temperature: f32,
    /// Top-k filtering (default: 40). 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus) filtering (default: 0.95). 0.0 = disabled.
    pub top_p: f32,
    /// Random seed for reproducibility (default: random).
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            seed: None,
        }
    }
}

/// Sample the next token from logits using the given configuration.
///
/// Applies temperature, top-k, top-p filtering, then samples categorically.
/// If temperature is 0, uses greedy argmax sampling.
pub fn sample_logits(logits: &[f32], config: &SamplingConfig) -> usize {
    // Greedy sampling when temperature is 0
    if config.temperature <= 0.0 {
        return sample_argmax(logits);
    }
    
    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng = fastrand::Rng::with_seed(seed);
    }
    
    // Apply top-k filtering
    let filtered = if config.top_k > 0 && config.top_k < logits.len() {
        apply_top_k(logits, config.top_k)
    } else {
        logits.to_vec()
    };
    
    // Apply top-p filtering
    let filtered = if config.top_p > 0.0 && config.top_p < 1.0 {
        apply_top_p(&filtered, config.top_p)
    } else {
        filtered
    };
    
    // Apply temperature and compute probabilities
    let probs = softmax_with_temperature(&filtered, config.temperature);
    
    // Sample from the distribution
    sample_categorical(&probs, &mut rng)
}
