// Per-layer KV cache for multi-head attention.
// Supports MHA, GQA (Grouped Query Attention), and MQA (Multi-Query Attention).

/// Per-layer KV cache.
/// 
/// Layout:
/// - `keys`  : [max_seq, n_head_kv, head_dim]
/// - `values`: [max_seq, n_head_kv, head_dim]
///
/// Flattened in row‑major order (seq -> head -> dim).
#[derive(Debug, Clone)]
pub struct KvCache {
    pub max_seq: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub cur_len: usize,
}

impl KvCache {
    /// Create a new KV cache for a single layer.
    pub fn new(max_seq: usize, n_head_kv: usize, head_dim: usize) -> Self {
        let size = max_seq * n_head_kv * head_dim;
        Self {
            max_seq,
            n_head_kv,
            head_dim,
            keys: vec![0.0; size],
            values: vec![0.0; size],
            cur_len: 0,
        }
    }
    
    /// Reset the cache (clear all entries).
    pub fn reset(&mut self) {
        self.cur_len = 0;
        self.keys.fill(0.0);
        self.values.fill(0.0);
    }

    /// Append a new token's key and value vectors.
    /// `k` and `v` must be of length `n_head_kv * head_dim`.
    pub fn push(&mut self, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.n_head_kv * self.head_dim);
        assert_eq!(v.len(), self.n_head_kv * self.head_dim);
        assert!(self.cur_len < self.max_seq, "KV cache overflow");
        let offset = self.cur_len * self.n_head_kv * self.head_dim;
        self.keys[offset..offset + k.len()].copy_from_slice(k);
        self.values[offset..offset + v.len()].copy_from_slice(v);
        self.cur_len += 1;
    }

    /// Retrieve a slice for a specific head and position.
    /// Returns (key_slice, value_slice) each of length `head_dim`.
    pub fn get(&self, pos: usize, head: usize) -> (&[f32], &[f32]) {
        assert!(pos < self.cur_len);
        assert!(head < self.n_head_kv);
        let base = (pos * self.n_head_kv + head) * self.head_dim;
        (
            &self.keys[base..base + self.head_dim],
            &self.values[base..base + self.head_dim],
        )
    }
    
    /// Get all keys for a specific KV head up to current length.
    /// Returns a flattened vector of shape (cur_len, head_dim).
    pub fn get_head_keys(&self, head: usize, len: usize) -> Vec<&[f32]> {
        assert!(head < self.n_head_kv);
        assert!(len <= self.cur_len);
        (0..len)
            .map(|pos| {
                let base = (pos * self.n_head_kv + head) * self.head_dim;
                &self.keys[base..base + self.head_dim]
            })
            .collect()
    }

    /// Get all values for a specific KV head up to current length.
    /// Returns a flattened vector of shape (cur_len, head_dim).
    pub fn get_head_values(&self, head: usize, len: usize) -> Vec<&[f32]> {
        assert!(head < self.n_head_kv);
        assert!(len <= self.cur_len);
        (0..len)
            .map(|pos| {
                let base = (pos * self.n_head_kv + head) * self.head_dim;
                &self.values[base..base + self.head_dim]
            })
            .collect()
    }
}

/// Multi-layer KV cache manager.
/// 
/// Holds one KvCache per transformer layer.
#[derive(Debug)]
pub struct KvCacheManager {
    pub caches: Vec<KvCache>,
    pub n_layers: usize,
}

impl KvCacheManager {
    /// Create a new KV cache manager with one cache per layer.
    pub fn new(n_layers: usize, max_seq: usize, n_head_kv: usize, head_dim: usize) -> Self {
        Self {
            caches: (0..n_layers)
                .map(|_| KvCache::new(max_seq, n_head_kv, head_dim))
                .collect(),
            n_layers,
        }
    }
    
    /// Reset all layer caches.
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }
    
    /// Get mutable reference to a specific layer's cache.
    pub fn get_layer(&mut self, layer: usize) -> &mut KvCache {
        &mut self.caches[layer]
    }
    
    /// Get immutable reference to a specific layer's cache.
    pub fn get_layer_ref(&self, layer: usize) -> &KvCache {
        &self.caches[layer]
    }
}
