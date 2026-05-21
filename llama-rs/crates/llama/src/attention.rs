// Complete multi-head attention implementation with RoPE and KV cache.
// Supports MHA, GQA (Grouped Query Attention), and MQA (Multi-Query Attention).
// Includes flash attention for memory-efficient prefill.

use crate::dot_product;
use crate::kv_cache::KvCache;

/// Flash attention: compute softmax(Q @ K^T) @ V in a single pass without materializing the full attention matrix.
/// Uses the online softmax trick: track running max and sum to compute softmax incrementally.
///
/// Memory complexity: O(N * head_dim) instead of O(N²)
/// where N = seq_len (context length)
///
/// # Arguments
/// * `q` - Query vector of shape (1, head_dim) for current token
/// * `k_cache` - Cached keys as slice of slices: each element is a head_dim-length slice
/// * `v_cache` - Cached values as slice of slices: each element is a head_dim-length slice
/// * `seq_len` - Current sequence length (number of cached tokens)
/// * `head_dim` - Dimension of each head
///
/// # Returns
/// Output vector of shape (1, head_dim)
fn flash_attention_head(
    q: &[f32],
    k_cache: &[&[f32]],
    v_cache: &[&[f32]],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), head_dim);
    assert_eq!(k_cache.len(), seq_len);
    assert_eq!(v_cache.len(), seq_len);

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Online softmax: track running max and sum
    let mut max_val = f32::NEG_INFINITY;
    let mut sum_exp = 0.0f32;
    let mut output = vec![0.0f32; head_dim];

    // Single pass: compute scores, update max/sum, accumulate weighted V
    for j in 0..seq_len {
        let k_row = k_cache[j];
        let score = dot_product(q, k_row) * scale;

        // Update running max and sum
        let prev_max = max_val;
        if score > max_val {
            max_val = score;
        }
        let exp_val = (score - max_val).exp();

        // Rescale previous output and sum
        let rescale = (prev_max - max_val).exp();
        sum_exp = sum_exp * rescale + exp_val;
        for d in 0..head_dim {
            output[d] = output[d] * rescale;
        }

        // Add weighted V
        let v_row = v_cache[j];
        for d in 0..head_dim {
            output[d] += exp_val * v_row[d];
        }
    }

    // Normalize by sum
    if sum_exp > 0.0 {
        let inv_sum = 1.0 / sum_exp;
        for d in 0..head_dim {
            output[d] *= inv_sum;
        }
    }

    output
}

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
/// (Legacy implementation, kept for reference. Flash attention is preferred.)
#[allow(dead_code)]
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

            // Convert to slice-of-slices for flash attention
            let k_refs: Vec<&[f32]> = k_cache.iter().map(|s| &s[..]).collect();
            let v_refs: Vec<&[f32]> = v_cache.iter().map(|s| &s[..]).collect();

            // Use flash attention: single pass, O(N) memory
            let out = flash_attention_head(
                q_vec,
                &k_refs,
                &v_refs,
                total_seq_len,
                head_dim,
            );

            // Store output
            let out_offset = pos * n_head * head_dim + h * head_dim;
            output[out_offset..out_offset + head_dim].copy_from_slice(&out);
        }
    }

    output
}

/// Flash attention for prefill phase (multi-token processing).
///
/// Uses online softmax to compute attention in a single pass without
/// materializing the full seq_len × seq_len attention matrix.
/// Memory complexity: O(seq_len * head_dim) instead of O(seq_len²).
///
/// # Arguments
/// * `n_head` - Number of query heads
/// * `n_head_kv` - Number of key/value heads (for GQA/MQA)
/// * `head_dim` - Dimension of each head
/// * `seq_len` - Sequence length
/// * `q` - Query projections, shape (seq_len, n_head * head_dim)
/// * `k` - Key projections, shape (seq_len, n_head_kv * head_dim)
/// * `v` - Value projections, shape (seq_len, n_head_kv * head_dim)
/// * `rope_theta` - RoPE base frequency
///
/// # Returns
/// Attention output of shape (seq_len, n_head * head_dim)
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
    let mut output = vec![0.0f32; seq_len * n_head * head_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..n_head {
        let kv_head = h / n_rep;

        for i in 0..seq_len {
            let q_offset = i * n_head * head_dim + h * head_dim;
            let q_vec = &q[q_offset..q_offset + head_dim];

            // Online softmax: single pass over causal context
            let mut max_val = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;
            let out_offset = i * n_head * head_dim + h * head_dim;
            let mut out_vec = vec![0.0f32; head_dim];

            for j in 0..=i {
                let k_offset = j * n_head_kv * head_dim + kv_head * head_dim;
                let k_vec = &k[k_offset..k_offset + head_dim];
                let score = dot_product(q_vec, k_vec) * scale;

                let prev_max = max_val;
                if score > max_val {
                    max_val = score;
                }
                let exp_val = (score - max_val).exp();

                // Rescale
                let rescale = (prev_max - max_val).exp();
                sum_exp = sum_exp * rescale + exp_val;
                for d in 0..head_dim {
                    out_vec[d] *= rescale;
                }

                // Accumulate weighted V
                let v_offset = j * n_head_kv * head_dim + kv_head * head_dim;
                for d in 0..head_dim {
                    out_vec[d] += exp_val * v[v_offset + d];
                }
            }

            // Normalize
            if sum_exp > 0.0 {
                let inv_sum = 1.0 / sum_exp;
                for d in 0..head_dim {
                    out_vec[d] *= inv_sum;
                }
            }

            output[out_offset..out_offset + head_dim].copy_from_slice(&out_vec);
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
