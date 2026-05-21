//! Tokenizer for LLaMA models.
//!
//! Supports loading vocabulary from GGUF metadata and provides
//! byte-level encoding/decoding with greedy longest-match.

use std::collections::HashMap;

/// Token type from GGUF metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    /// Normal token (text).
    Normal,
    /// Unknown token.
    Unknown,
    /// Control token (e.g., BOS, EOS).
    Control,
    /// User-defined token.
    UserDefined,
    /// Byte token.
    Byte,
}

impl TokenType {
    /// Convert from GGUF token type integer.
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Byte,
            _ => TokenType::Normal,
        }
    }
}

/// Tokenizer that loads vocabulary from GGUF metadata.
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    /// Mapping from token string to token ID.
    token_to_id: HashMap<String, usize>,
    /// Mapping from token ID to token string.
    id_to_token: HashMap<usize, String>,
    /// Token types for each token ID.
    token_types: Vec<TokenType>,
    /// Scores for each token ID (used for BPE).
    scores: Vec<f32>,
    /// BOS token ID.
    pub bos_token_id: usize,
    /// EOS token ID.
    pub eos_token_id: usize,
    /// Unknown token ID.
    pub unk_token_id: usize,
    /// Whether to add BOS token automatically.
    pub add_bos_token: bool,
    /// Maximum token length for efficient matching.
    max_token_len: usize,
}

impl SimpleTokenizer {
    /// Create a new tokenizer with a basic ASCII vocabulary.
    pub fn new() -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut token_types = Vec::new();
        let mut scores = Vec::new();

        // Reserve token 0 for unknown/padding
        token_to_id.insert("<unk>".to_string(), 0);
        id_to_token.insert(0, "<unk>".to_string());
        token_types.push(TokenType::Unknown);
        scores.push(0.0);

        // Add printable ASCII characters (32-126)
        for (i, ch) in (32u8..=126).enumerate() {
            let c = ch as char;
            let id = i + 1;
            let token = c.to_string();
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token);
            token_types.push(TokenType::Normal);
            scores.push(0.0);
        }

        // Add special tokens
        token_to_id.insert("<s>".to_string(), 96);
        id_to_token.insert(96, "<s>".to_string());
        token_types.push(TokenType::Control);
        scores.push(0.0);
        token_to_id.insert("</s>".to_string(), 97);
        id_to_token.insert(97, "</s>".to_string());
        token_types.push(TokenType::Control);
        scores.push(0.0);

        Self {
            token_to_id,
            id_to_token,
            token_types,
            scores,
            bos_token_id: 96,
            eos_token_id: 97,
            unk_token_id: 0,
            add_bos_token: false,
            max_token_len: 1,
        }
    }

    /// Create a tokenizer from GGUF vocabulary.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Array of token strings from GGUF metadata
    /// * `scores` - Array of token scores (for BPE ranking)
    /// * `token_types` - Array of token types
    /// * `bos_token_id` - BOS token ID
    /// * `eos_token_id` - EOS token ID
    /// * `unk_token_id` - Unknown token ID
    /// * `add_bos_token` - Whether to add BOS token automatically
    pub fn from_gguf_vocab(
        tokens: Vec<String>,
        scores: Vec<f32>,
        token_types: Vec<TokenType>,
        bos_token_id: usize,
        eos_token_id: usize,
        unk_token_id: usize,
        add_bos_token: bool,
    ) -> Self {
        let vocab_size = tokens.len();
        let mut token_to_id = HashMap::with_capacity(vocab_size);
        let mut id_to_token = HashMap::with_capacity(vocab_size);
        let mut max_token_len = 0;

        for (id, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
            if token.len() > max_token_len {
                max_token_len = token.len();
            }
        }

        Self {
            token_to_id,
            id_to_token,
            token_types,
            scores,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            add_bos_token,
            max_token_len,
        }
    }

    /// Encode text to token IDs using greedy longest-match.
    ///
    /// This algorithm:
    /// 1. Tries to find the longest matching token in the vocabulary
    /// 2. Falls back to byte-level encoding for unknown sequences
    /// 3. Handles special tokens (BOS, EOS) appropriately
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();

        // Add BOS token if configured
        if self.add_bos_token {
            tokens.push(self.bos_token_id);
        }

        // Greedy longest-match encoding
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            // Try to find the longest matching token
            let mut found = false;
            // Try from longest to shortest
            for len in (1..=self.max_token_len.min(bytes.len() - i)).rev() {
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
                // Fall back to byte-level encoding
                // Try to find byte token
                let byte_val = bytes[i];
                let byte_token = format!("<0x{:02X}>", byte_val);
                if let Some(&id) = self.token_to_id.get(&byte_token) {
                    tokens.push(id);
                } else {
                    // Last resort: unknown token
                    tokens.push(self.unk_token_id);
                }
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
                match self.token_types.get(id) {
                    Some(TokenType::Control) | Some(TokenType::Unknown) => continue,
                    Some(TokenType::Byte) => {
                        // Parse byte token like <0xXX>
                        if let Some(byte_str) = token.strip_prefix("<0x").and_then(|s| s.strip_suffix(">")) {
                            if let Ok(byte_val) = u8::from_str_radix(byte_str, 16) {
                                result.push(byte_val as char);
                            }
                        }
                    }
                    _ => result.push_str(token),
                }
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
        let scores: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let types: Vec<TokenType> = (0..100).map(|_| TokenType::Normal).collect();
        let tokenizer = SimpleTokenizer::from_gguf_vocab(tokens, scores, types, 1, 2, 0, true);
        
        assert_eq!(tokenizer.vocab_size(), 100);
        assert_eq!(tokenizer.bos_token_id, 1);
        assert_eq!(tokenizer.eos_token_id, 2);
        assert!(tokenizer.add_bos_token);
        
        // Test encoding
        let tokens = tokenizer.encode("token5");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_bpe_style_encoding() {
        // Test with a vocabulary that has multi-character tokens
        let tokens = vec![
            "<unk>".to_string(),
            "Hello".to_string(),
            " World".to_string(),
            "!".to_string(),
            "H".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
        ];
        let scores = vec![0.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5];
        let types = vec![
            TokenType::Unknown,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
        ];
        let tokenizer = SimpleTokenizer::from_gguf_vocab(
            tokens, scores, types, 0, 0, 0, false,
        );
        
        // "Hello World!" should encode as ["Hello", " World", "!"]
        let tokens = tokenizer.encode("Hello World!");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], 1); // "Hello"
        assert_eq!(tokens[1], 2); // " World"
        assert_eq!(tokens[2], 3); // "!"
    }
}
