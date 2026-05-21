// Simple Structure‑of‑Arrays KV cache for multi‑head attention.
// Stores keys and values separately for better cache locality.
// This is a minimal implementation sufficient for the Rust inference engine.

use rayon::prelude::*;

/// KV cache layout:
/// - `keys`  : [max_seq, n_head, head_dim]
/// - `values`: [max_seq, n_head, head_dim]
///
/// Flattened in row‑major order (seq -> head -> dim).
#[derive(Debug, Clone)]
pub struct KvCache {
    pub max_seq: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub cur_len: usize,
}

impl KvCache {
    /// Create a new KV cache.
    pub fn new(max_seq: usize, n_head: usize, head_dim: usize) -> Self {
        let size = max_seq * n_head * head_dim;
        Self {
            max_seq,
            n_head,
            head_dim,
            keys: vec![0.0; size],
            values: vec![0.0; size],
            cur_len: 0,
        }
    }

    /// Append a new token's key and value vectors.
    /// `k` and `v` must be of length `n_head * head_dim`.
    pub fn push(&mut self, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.n_head * self.head_dim);
        assert_eq!(v.len(), self.n_head * self.head_dim);
        assert!(self.cur_len < self.max_seq, "KV cache overflow");
        let offset = self.cur_len * self.n_head * self.head_dim;
        self.keys[offset..offset + k.len()].copy_from_slice(k);
        self.values[offset..offset + v.len()].copy_from_slice(v);
        self.cur_len += 1;
    }

    /// Retrieve a slice for a specific head and position.
    /// Returns (key_slice, value_slice) each of length `head_dim`.
    pub fn get(&self, pos: usize, head: usize) -> (&[f32], &[f32]) {
        assert!(pos < self.cur_len);
        assert!(head < self.n_head);
        let base = (pos * self.n_head + head) * self.head_dim;
        (
            &self.keys[base..base + self.head_dim],
            &self.values[base..base + self.head_dim],
        )
    }

    /// Parallel retrieval of all keys for a given head up to `len` tokens.
    pub fn get_head_keys(&self, head: usize, len: usize) -> Vec<&[f32]> {
        (0..len)
            .into_par_iter()
            .map(|pos| {
                let base = (pos * self.n_head + head) * self.head_dim;
                &self.keys[base..base + self.head_dim]
            })
            .collect()
    }

    /// Parallel retrieval of all values for a given head up to `len` tokens.
    pub fn get_head_values(&self, head: usize, len: usize) -> Vec<&[f32]> {
        (0..len)
            .into_par_iter()
            .map(|pos| {
                let base = (pos * self.n_head + head) * self.head_dim;
                &self.values[base..base + self.head_dim]
            })
            .collect()
    }
}
