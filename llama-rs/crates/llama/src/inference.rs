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
pub fn mat_vec(mat: &[f32], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    assert_eq!(mat.len(), rows * cols);
    assert_eq!(vec.len(), cols);
    
    (0..rows)
        .map(|r| {
            let start = r * cols;
            let row = &mat[start..start + cols];
            row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum()
        })
        .collect()
}

/// Element-wise multiplication of two vectors.
pub fn mul_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Add two vectors element-wise.
pub fn add_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Multiply a vector by a scalar.
pub fn scale_vec(v: &[f32], scale: f32) -> Vec<f32> {
    v.iter().map(|x| x * scale).collect()
}
