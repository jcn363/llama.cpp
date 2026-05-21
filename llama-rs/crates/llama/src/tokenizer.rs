//! Simple tokenizer for text encoding/decoding.
//!
//! This is a minimal character-level tokenizer for demonstration.
//! Production systems would use BPE or SentencePiece tokenizers.

use std::collections::HashMap;

/// A simple character-level tokenizer.
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    /// Mapping from character to token ID.
    char_to_id: HashMap<char, usize>,
    /// Mapping from token ID to character.
    id_to_char: HashMap<usize, char>,
}

impl SimpleTokenizer {
    /// Create a new tokenizer with a basic ASCII vocabulary.
    pub fn new() -> Self {
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();

        // Reserve token 0 for unknown/padding
        char_to_id.insert('\0', 0);
        id_to_char.insert(0, '\0');

        // Add printable ASCII characters (32-126)
        for (i, ch) in (32u8..=126).enumerate() {
            let c = ch as char;
            let id = i + 1;
            char_to_id.insert(c, id);
            id_to_char.insert(id, c);
        }

        // Add common whitespace and control characters
        let special_chars = ['\n', '\t', '\r'];
        for (i, &ch) in special_chars.iter().enumerate() {
            let id = 96 + i; // Start after ASCII printables
            char_to_id.insert(ch, id);
            id_to_char.insert(id, ch);
        }

        Self { char_to_id, id_to_char }
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|ch| *self.char_to_id.get(&ch).unwrap_or(&0))
            .collect()
    }

    /// Decode token IDs to text.
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .filter_map(|&id| self.id_to_char.get(&id).copied())
            .collect()
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.char_to_id.len()
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
}
