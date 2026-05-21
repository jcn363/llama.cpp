// Multi-head attention implementation (simplified)
// This is a minimal, parallel version using the SIMD-friendly `dot_product`
// and the `mat_vec_batch` helper defined in lib.rs.

use crate::{dot_product, mat_vec_batch};
use crate::kv_cache::KvCache;

/// Compute scaled dot‑product attention for a single head.
/// `q`, `k`, `v` are slices of length `seq_len * head_dim`.
/// Returns a vector of length `seq_len * head_dim`.
/// Fused attention head: computes QK^T, softmax, and weighted sum in a single routine.
/// `scores` is a pre‑allocated buffer of size `seq_len * seq_len` reused across heads.
fn attention_head_fused(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scores: &mut [f32],
) -> Vec<f32> {
    // 1. Compute raw scores and find max per row for numerical stability.
    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        let row_start = i * seq_len;
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..seq_len {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            let val = dot_product(q_row, k_row) / (head_dim as f32).sqrt();
            scores[row_start + j] = val;
            if val > max_val {
                max_val = val;
            }
        }
        // 2. Exponentiate and accumulate sum in the same pass.
        let mut sum = 0.0f32;
        for j in 0..seq_len {
            let idx = row_start + j;
            let exp_val = (scores[idx] - max_val).exp();
            scores[idx] = exp_val; // reuse buffer for softmax numerator
            sum += exp_val;
        }
        // 3. Normalize to obtain softmax probabilities.
        for j in 0..seq_len {
            let idx = row_start + j;
            scores[idx] /= sum;
        }
    }

    // 4. Weighted sum of V using the softmax scores.
    let mut out = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        let out_slice = &mut out[i * head_dim..(i + 1) * head_dim];
        for j in 0..seq_len {
            let weight = scores[i * seq_len + j];
            let v_row = &v[j * head_dim..(j + 1) * head_dim];
            for d in 0..head_dim {
                out_slice[d] += weight * v_row[d];
            }
        }
    }
    out
}

/// Backward‑compatible wrapper used by `multi_head_attention`.
fn attention_head(q: &[f32], k: &[f32], v: &[f32], seq_len: usize, head_dim: usize) -> Vec<f32> {
    // Allocate a temporary scores buffer sized for this head.
    let mut scores = vec![0.0f32; seq_len * seq_len];
    attention_head_fused(q, k, v, seq_len, head_dim, &mut scores)
}

/// Batched multi‑head attention.
///
/// * `q_proj`, `k_proj`, `v_proj` are weight matrices of shape `(embed, n_head * head_dim)`.
/// * `input` is a sequence of token embeddings of shape `(seq_len, embed)`.
/// * Returns the attention output of shape `(seq_len, embed)`.
pub fn multi_head_attention(
    embed: usize,
    n_head: usize,
    head_dim: usize,
    seq_len: usize,
    input: &[f32],
    q_proj: &[f32],
    k_proj: &[f32],
    v_proj: &[f32],
    _kv_cache: Option<&mut KvCache>,
) -> Vec<f32> {
    // Project input to Q, K, V using batched mat‑vec multiplication.
    let q = mat_vec_batch(q_proj, embed, n_head * head_dim, input);
    let k = mat_vec_batch(k_proj, embed, n_head * head_dim, input);
    let v = mat_vec_batch(v_proj, embed, n_head * head_dim, input);

    // Split into heads and compute attention sequentially.
    let mut head_outputs: Vec<Vec<f32>> = Vec::with_capacity(n_head);
    for h in 0..n_head {
        let offset = h * head_dim;
        // Extract per‑head slices.
        let q_head: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                let start = i * n_head * head_dim + offset;
                q[start..start + head_dim].to_vec()
            })
            .collect();
        let k_head: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                let start = i * n_head * head_dim + offset;
                k[start..start + head_dim].to_vec()
            })
            .collect();
        let v_head: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                let start = i * n_head * head_dim + offset;
                v[start..start + head_dim].to_vec()
            })
            .collect();
        let out = attention_head(&q_head, &k_head, &v_head, seq_len, head_dim);
        head_outputs.push(out);
    }
    // Concatenate heads back into (seq_len, embed).
    let mut output = vec![0.0f32; seq_len * embed];
    for (h, head_out) in head_outputs.iter().enumerate() {
        for i in 0..seq_len {
            let out_start = i * embed + h * head_dim;
            let src_start = i * head_dim;
            output[out_start..out_start + head_dim]
                .copy_from_slice(&head_out[src_start..src_start + head_dim]);
        }
    }
    output
}
