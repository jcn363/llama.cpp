// Complete multi-head attention implementation with RoPE and KV cache.
// Supports MHA, GQA (Grouped Query Attention), and MQA (Multi-Query Attention).

use crate::dot_product;
use crate::kv_cache::KvCache;

/// Apply Rotary Position Embedding (RoPE) to Q and K vectors.
///
/// RoPE rotates Q and K vectors based on their position in the sequence.
/// This encodes positional information without adding learnable parameters.
///
/// # Arguments
/// * `x` - Input vector of shape (seq_len, head_dim), flattened
/// * `seq_len` - Sequence length
/// * `head_dim` - Dimension of each head
/// * `position_offset` - Starting position (for KV cache continuity)
/// * `rope_theta` - Base frequency for RoPE (typically 10000.0)
pub fn apply_rope(x: &mut [f32], seq_len: usize, head_dim: usize, position_offset: usize, rope_theta: f32) {
    let half_dim = head_dim / 2;
    
    for pos in 0..seq_len {
        let actual_pos = position_offset + pos;
        let row_start = pos * head_dim;
        
        for i in 0..half_dim {
            let freq = 1.0 / rope_theta.powf(i as f32 / half_dim as f32);
            let theta = actual_pos as f32 * freq;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            
            let idx1 = row_start + i;
            let idx2 = row_start + i + half_dim;
            
            let x1 = x[idx1];
            let x2 = x[idx2];
            
            // Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
            x[idx1] = x1 * cos_theta - x2 * sin_theta;
            x[idx2] = x1 * sin_theta + x2 * cos_theta;
        }
    }
}

/// Compute scaled dot-product attention for a single head with causal masking.
///
/// # Arguments
/// * `q` - Query vector of shape (1, head_dim) for current token
/// * `k_cache` - Cached keys of shape (seq_len, head_dim)
/// * `v_cache` - Cached values of shape (seq_len, head_dim)
/// * `seq_len` - Current sequence length (number of cached tokens)
/// * `head_dim` - Dimension of each head
/// * `scores` - Pre-allocated buffer of size `seq_len`
///
/// # Returns
/// Output vector of shape (1, head_dim)
fn attention_head_with_cache(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    head_dim: usize,
    scores: &mut [f32],
) -> Vec<f32> {
    assert_eq!(q.len(), head_dim);
    assert_eq!(k_cache.len(), seq_len * head_dim);
    assert_eq!(v_cache.len(), seq_len * head_dim);
    assert_eq!(scores.len(), seq_len);
    
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    // 1. Compute Q @ K^T for all cached positions
    let mut max_val = f32::NEG_INFINITY;
    for j in 0..seq_len {
        let k_row = &k_cache[j * head_dim..(j + 1) * head_dim];
        let val = dot_product(q, k_row) * scale;
        scores[j] = val;
        if val > max_val {
            max_val = val;
        }
    }
    
    // 2. Softmax with numerical stability
    let mut sum = 0.0f32;
    for j in 0..seq_len {
        let exp_val = (scores[j] - max_val).exp();
        scores[j] = exp_val;
        sum += exp_val;
    }
    for j in 0..seq_len {
        scores[j] /= sum;
    }
    
    // 3. Weighted sum of V
    let mut out = vec![0.0f32; head_dim];
    for j in 0..seq_len {
        let weight = scores[j];
        let v_row = &v_cache[j * head_dim..(j + 1) * head_dim];
        for d in 0..head_dim {
            out[d] += weight * v_row[d];
        }
    }
    
    out
}

/// Multi-head attention with KV cache support.
///
/// # Arguments
/// * `n_head` - Number of query heads
/// * `n_head_kv` - Number of key/value heads (for GQA/MQA)
/// * `head_dim` - Dimension of each head
/// * `seq_len` - Current sequence length
/// * `position_offset` - Starting position in KV cache
/// * `q` - Query projections, shape (seq_len, n_head * head_dim)
/// * `k` - Key projections, shape (seq_len, n_head_kv * head_dim)
/// * `v` - Value projections, shape (seq_len, n_head_kv * head_dim)
/// * `kv_cache` - KV cache to store/retrieve keys and values
/// * `rope_theta` - RoPE base frequency
///
/// # Returns
/// Attention output of shape (seq_len, n_head * head_dim)
pub fn multi_head_attention_with_cache(
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    seq_len: usize,
    position_offset: usize,
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    kv_cache: &mut KvCache,
    rope_theta: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * n_head * head_dim);
    assert_eq!(k.len(), seq_len * n_head_kv * head_dim);
    assert_eq!(v.len(), seq_len * n_head_kv * head_dim);
    
    // Apply RoPE to Q and K
    apply_rope(q, seq_len, head_dim, position_offset, rope_theta);
    apply_rope(k, seq_len, head_dim, position_offset, rope_theta);
    
    // Store K and V in cache
    for pos in 0..seq_len {
        let k_offset = pos * n_head_kv * head_dim;
        let v_offset = pos * n_head_kv * head_dim;
        kv_cache.push(
            &k[k_offset..k_offset + n_head_kv * head_dim],
            &v[v_offset..v_offset + n_head_kv * head_dim],
        );
    }
    
    let total_seq_len = position_offset + seq_len;
    let n_rep = n_head / n_head_kv; // Number of query heads per KV head (for GQA)
    
    // Pre-allocate scores buffer (reused across heads)
    let mut scores = vec![0.0f32; total_seq_len];
    
    // Compute attention for each query head
    let mut output = vec![0.0f32; seq_len * n_head * head_dim];
    
    for h in 0..n_head {
        // For GQA, multiple query heads share the same KV head
        let kv_head = h / n_rep;
        
        for pos in 0..seq_len {
            // Get query for this position and head
            let q_offset = pos * n_head * head_dim + h * head_dim;
            let q_vec = &q[q_offset..q_offset + head_dim];
            
            // Get cached keys and values for the corresponding KV head
            let k_cache = kv_cache.get_head_keys(kv_head, total_seq_len);
            let v_cache = kv_cache.get_head_values(kv_head, total_seq_len);
            
            // Flatten cached keys and values for this head
            let k_flat: Vec<f32> = k_cache.iter().flat_map(|s| s.iter().copied()).collect();
            let v_flat: Vec<f32> = v_cache.iter().flat_map(|s| s.iter().copied()).collect();
            
            // Compute attention
            let out = attention_head_with_cache(
                q_vec,
                &k_flat,
                &v_flat,
                total_seq_len,
                head_dim,
                &mut scores,
            );
            
            // Store output
            let out_offset = pos * n_head * head_dim + h * head_dim;
            output[out_offset..out_offset + head_dim].copy_from_slice(&out);
        }
    }
    
    output
}

/// Simple multi-head attention without KV cache (for batch processing).
///
/// This is a simplified version that processes all tokens at once.
/// Used for prompt encoding (prefill phase).
pub fn multi_head_attention_prefill(
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    seq_len: usize,
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    rope_theta: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * n_head * head_dim);
    assert_eq!(k.len(), seq_len * n_head_kv * head_dim);
    assert_eq!(v.len(), seq_len * n_head_kv * head_dim);
    
    // Apply RoPE to Q and K
    apply_rope(q, seq_len, head_dim, 0, rope_theta);
    apply_rope(k, seq_len, head_dim, 0, rope_theta);
    
    let n_rep = n_head / n_head_kv;
    let mut scores = vec![0.0f32; seq_len * seq_len];
    let mut output = vec![0.0f32; seq_len * n_head * head_dim];
    
    for h in 0..n_head {
        let kv_head = h / n_rep;
        
        for i in 0..seq_len {
            // Get query for position i
            let q_offset = i * n_head * head_dim + h * head_dim;
            let q_vec = &q[q_offset..q_offset + head_dim];
            
            // Compute attention scores for all positions j <= i (causal mask)
            let mut max_val = f32::NEG_INFINITY;
            let row_start = i * seq_len;
            
            for j in 0..=i {
                let k_offset = j * n_head_kv * head_dim + kv_head * head_dim;
                let k_vec = &k[k_offset..k_offset + head_dim];
                let val = dot_product(q_vec, k_vec) / (head_dim as f32).sqrt();
                scores[row_start + j] = val;
                if val > max_val {
                    max_val = val;
                }
            }
            
            // Softmax over valid positions
            let mut sum = 0.0f32;
            for j in 0..=i {
                let exp_val = (scores[row_start + j] - max_val).exp();
                scores[row_start + j] = exp_val;
                sum += exp_val;
            }
            for j in 0..=i {
                scores[row_start + j] /= sum;
            }
            
            // Weighted sum of V
            let out_offset = i * n_head * head_dim + h * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for j in 0..=i {
                    let v_offset = j * n_head_kv * head_dim + kv_head * head_dim + d;
                    val += scores[row_start + j] * v[v_offset];
                }
                output[out_offset + d] = val;
            }
        }
    }
    
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rope_basic() {
        let head_dim = 4;
        let seq_len = 2;
        let mut x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        
        apply_rope(&mut x, seq_len, head_dim, 0, 10000.0);
        
        // Position 0: theta = 0, cos(0) = 1, sin(0) = 0, no change
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - 0.0).abs() < 1e-6);
        assert!((x[2] - 0.0).abs() < 1e-6);
        assert!((x[3] - 1.0).abs() < 1e-6);
        
        // Position 1: should be rotated
        // The values should have changed from the original
        assert!((x[4] - 1.0).abs() > 1e-6 || (x[5] - 0.0).abs() > 1e-6);
    }
    
    #[test]
    fn test_attention_prefill() {
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let seq_len = 3;
        
        let mut q = vec![0.1; seq_len * n_head * head_dim];
        let mut k = vec![0.1; seq_len * n_head_kv * head_dim];
        let v = vec![0.1; seq_len * n_head_kv * head_dim];
        
        let output = multi_head_attention_prefill(
            n_head, n_head_kv, head_dim, seq_len,
            &mut q, &mut k, &v, 10000.0,
        );
        
        assert_eq!(output.len(), seq_len * n_head * head_dim);
        // Output should be non-zero
        assert!(output.iter().any(|&x| x.abs() > 1e-6));
    }
}
