//! Tokenizer for LLaMA models.
//!
//! Supports loading vocabulary from GGUF metadata and provides
//! byte-level encoding/decoding.

use std::collections::HashMap;

/// Tokenizer that loads vocabulary from GGUF metadata.
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    /// Mapping from token string to token ID.
    token_to_id: HashMap<String, usize>,
    /// Mapping from token ID to token string.
    id_to_token: HashMap<usize, String>,
    /// BOS token ID.
    pub bos_token_id: usize,
    /// EOS token ID.
    pub eos_token_id: usize,
    /// Unknown token ID.
    pub unk_token_id: usize,
    /// Whether to add BOS token automatically.
    pub add_bos_token: bool,
}

impl SimpleTokenizer {
    /// Create a new tokenizer with a basic ASCII vocabulary.
    pub fn new() -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Reserve token 0 for unknown/padding
        token_to_id.insert("<unk>".to_string(), 0);
        id_to_token.insert(0, "<unk>".to_string());

        // Add printable ASCII characters (32-126)
        for (i, ch) in (32u8..=126).enumerate() {
            let c = ch as char;
            let id = i + 1;
            let token = c.to_string();
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        // Add special tokens
        token_to_id.insert("<s>".to_string(), 96);
        id_to_token.insert(96, "<s>".to_string());
        token_to_id.insert("</s>".to_string(), 97);
        id_to_token.insert(97, "</s>".to_string());

        Self {
            token_to_id,
            id_to_token,
            bos_token_id: 96,
            eos_token_id: 97,
            unk_token_id: 0,
            add_bos_token: false,
        }
    }

    /// Create a tokenizer from GGUF vocabulary.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Array of token strings from GGUF metadata
    /// * `bos_token_id` - BOS token ID
    /// * `eos_token_id` - EOS token ID
    /// * `unk_token_id` - Unknown token ID
    /// * `add_bos_token` - Whether to add BOS token automatically
    pub fn from_gguf_vocab(
        tokens: Vec<String>,
        bos_token_id: usize,
        eos_token_id: usize,
        unk_token_id: usize,
        add_bos_token: bool,
    ) -> Self {
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut id_to_token = HashMap::with_capacity(tokens.len());

        for (id, token) in tokens.into_iter().enumerate() {
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        Self {
            token_to_id,
            id_to_token,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            add_bos_token,
        }
    }

    /// Encode text to token IDs using byte-level encoding.
    ///
    /// This is a simplified encoder that:
    /// 1. Tries to find the exact token in the vocabulary
    /// 2. Falls back to byte-level encoding for unknown sequences
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();

        // Add BOS token if configured
        if self.add_bos_token {
            tokens.push(self.bos_token_id);
        }

        // Try to encode using the vocabulary
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            // Try to find the longest matching token
            let mut found = false;
            for len in (1..=bytes.len() - i).rev() {
                if let Ok(s) = std::str::from_utf8(&bytes[i..i + len]) {
                    if let Some(&id) = self.token_to_id.get(s) {
                        tokens.push(id);
                        i += len;
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                // Fall back to unknown token
                tokens.push(self.unk_token_id);
                i += 1;
            }
        }

        tokens
    }

    /// Decode token IDs to text.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut result = String::new();
        for &id in tokens {
            if let Some(token) = self.id_to_token.get(&id) {
                // Skip special tokens
                if token == "<unk>" || token == "<s>" || token == "</s>" {
                    continue;
                }
                result.push_str(token);
            }
        }
        result
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let tokenizer = SimpleTokenizer::new();
        let text = "Hello, World!";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_vocab_size() {
        let tokenizer = SimpleTokenizer::new();
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_unknown_char() {
        let tokenizer = SimpleTokenizer::new();
        let tokens = tokenizer.encode("Hello\u{1F600}World"); // emoji
        // Unknown chars should map to 0
        assert!(tokens.contains(&0));
    }

    #[test]
    fn test_from_gguf_vocab() {
        let tokens: Vec<String> = (0..100).map(|i| format!("token{}", i)).collect();
        let tokenizer = SimpleTokenizer::from_gguf_vocab(tokens, 1, 2, 0, true);
        
        assert_eq!(tokenizer.vocab_size(), 100);
        assert_eq!(tokenizer.bos_token_id, 1);
        assert_eq!(tokenizer.eos_token_id, 2);
        assert!(tokenizer.add_bos_token);
        
        // Test encoding
        let tokens = tokenizer.encode("token5");
        assert!(!tokens.is_empty());
    }
}
