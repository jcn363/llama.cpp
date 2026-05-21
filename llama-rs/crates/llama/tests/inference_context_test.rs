use std::sync::Arc;
use std::io::{Write, Seek};
use llama::{InferenceContext, Model, ModelConfig};

/// Create a minimal GGUF file for testing.
fn create_test_gguf(path: &std::path::Path) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    
    // Header
    f.write_all(&0x4655_4747u32.to_le_bytes())?; // GGUF magic
    f.write_all(&3u32.to_le_bytes())?; // version
    f.write_all(&7i64.to_le_bytes())?; // tensor count
    f.write_all(&10i64.to_le_bytes())?; // kv count
    
    // Helper to write a string
    let mut write_string = |f: &mut std::fs::File, s: &str| -> std::io::Result<()> {
        f.write_all(&(s.len() as u64).to_le_bytes())?;
        f.write_all(s.as_bytes())?;
        Ok(())
    };
    
    // Helper to write a KV pair with u32 value
    let mut write_kv_u32 = |f: &mut std::fs::File, key: &str, v: u32| -> std::io::Result<()> {
        write_string(f, key)?;
        f.write_all(&4i32.to_le_bytes())?; // type = uint32
        f.write_all(&v.to_le_bytes())?;
        Ok(())
    };
    
    // Helper to write a KV pair with string value
    let mut write_kv_str = |f: &mut std::fs::File, key: &str, v: &str| -> std::io::Result<()> {
        write_string(f, key)?;
        f.write_all(&8i32.to_le_bytes())?; // type = string
        write_string(f, v)?;
        Ok(())
    };
    
    // KV pairs (10 total)
    write_kv_str(&mut f, "general.version", "1.0")?;
    write_kv_u32(&mut f, "general.vocab_size", 4)?;
    write_kv_u32(&mut f, "general.embedding_length", 8)?;
    write_kv_u32(&mut f, "general.attention_head_count", 1)?;
    write_kv_u32(&mut f, "general.attention_head_count_kv", 1)?;
    write_kv_u32(&mut f, "general.attention_head_dim", 8)?;
    write_kv_u32(&mut f, "general.context_length", 512)?;
    write_kv_u32(&mut f, "llama.feed_forward_length", 16)?;
    write_kv_str(&mut f, "general.architecture", "llama")?;
    write_kv_u32(&mut f, "general.alignment", 32)?;
    
    // Tensor info
    let mut write_tensor_info = |f: &mut std::fs::File, name: &str, shape: &[i64], dtype: i32| -> std::io::Result<()> {
        write_string(f, name)?;
        f.write_all(&(shape.len() as u32).to_le_bytes())?;
        for &d in shape {
            f.write_all(&d.to_le_bytes())?;
        }
        f.write_all(&dtype.to_le_bytes())?;
        f.write_all(&0u64.to_le_bytes())?; // offset (placeholder)
        Ok(())
    };
    
    // dtype 0 = F32
    write_tensor_info(&mut f, "token_embd.weight", &[8, 4], 0)?;
    write_tensor_info(&mut f, "output.weight", &[4, 8], 0)?;
    write_tensor_info(&mut f, "output_norm.weight", &[8], 0)?;
    write_tensor_info(&mut f, "blk.0.attn_norm.weight", &[8], 0)?;
    write_tensor_info(&mut f, "blk.0.ffn_norm.weight", &[8], 0)?;
    write_tensor_info(&mut f, "blk.0.ffn_gate.weight", &[16, 8], 0)?;
    write_tensor_info(&mut f, "blk.0.ffn_up.weight", &[16, 8], 0)?;
    write_tensor_info(&mut f, "blk.0.ffn_down.weight", &[8, 16], 0)?;
    
    // Align to 32 bytes
    let pos = f.stream_position()?;
    let padding = (32 - (pos % 32)) % 32;
    f.write_all(&vec![0u8; padding as usize])?;
    
    // Tensor data (all F32)
    let mut write_tensor = |f: &mut std::fs::File, data: &[f32]| -> std::io::Result<()> {
        for &v in data {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    };
    
    // token_embd.weight: [8, 4] = 32 floats
    let mut rng = fastrand::Rng::with_seed(42);
    let embd: Vec<f32> = (0..32).map(|_| rng.f32()).collect();
    write_tensor(&mut f, &embd)?;
    
    // output.weight: [4, 8] = 32 floats
    let out: Vec<f32> = (0..32).map(|_| rng.f32()).collect();
    write_tensor(&mut f, &out)?;
    
    // output_norm.weight: [8]
    let norm: Vec<f32> = (0..8).map(|_| 1.0).collect();
    write_tensor(&mut f, &norm)?;
    
    // blk.0.attn_norm.weight: [8]
    write_tensor(&mut f, &norm)?;
    
    // blk.0.ffn_norm.weight: [8]
    write_tensor(&mut f, &norm)?;
    
    // blk.0.ffn_gate.weight: [16, 8] = 128 floats
    let gate: Vec<f32> = (0..128).map(|_| rng.f32()).collect();
    write_tensor(&mut f, &gate)?;
    
    // blk.0.ffn_up.weight: [16, 8] = 128 floats
    let up: Vec<f32> = (0..128).map(|_| rng.f32()).collect();
    write_tensor(&mut f, &up)?;
    
    // blk.0.ffn_down.weight: [8, 16] = 128 floats
    let down: Vec<f32> = (0..128).map(|_| rng.f32()).collect();
    write_tensor(&mut f, &down)?;
    
    Ok(())
}

#[test]
fn test_inference_context_encode_decode() {
    let tokenizer = llama::SimpleTokenizer::new();
    let text = "Hello, World!";
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens);
    assert_eq!(decoded, text);
}

#[test]
fn test_model_config_default() {
    let config = ModelConfig::default();
    assert_eq!(config.n_threads, 4);
    assert_eq!(config.n_ctx, 2048);
    assert_eq!(config.n_batch, 512);
    assert!(!config.use_cuda);
}

#[test]
fn test_load_dummy_model() {
    let tmp_dir = std::env::temp_dir();
    let gguf_path = tmp_dir.join("test_model.gguf");
    
    create_test_gguf(&gguf_path).expect("Failed to create test GGUF");
    
    let model = Model::load_from_gguf(&gguf_path).expect("Failed to load test model");
    
    assert_eq!(model.n_embd, 8);
    assert_eq!(model.n_head, 1);
    assert_eq!(model.n_head_kv, 1);
    assert_eq!(model.d_head, 8);
    assert_eq!(model.max_seq_len, 512);
    assert_eq!(model.vocab_size, 4);
    assert_eq!(model.n_ff, 16);
    assert_eq!(model.n_layers(), 1);
    
    std::fs::remove_file(&gguf_path).ok();
}

#[test]
fn test_forward_pass_produces_logits() {
    let tmp_dir = std::env::temp_dir();
    let gguf_path = tmp_dir.join("test_model_forward.gguf");
    
    create_test_gguf(&gguf_path).expect("Failed to create test GGUF");
    
    let model = Arc::new(Model::load_from_gguf(&gguf_path).expect("Failed to load test model"));
    let config = ModelConfig::default();
    let ctx = InferenceContext::new(model, config);
    
    let result = ctx.generate("test", 5);
    assert!(result.is_ok());
    let tokens = result.unwrap();
    assert!(!tokens.is_empty());
    assert!(tokens.len() >= 1);
    
    std::fs::remove_file(&gguf_path).ok();
}

/// Test loading a real GGUF model file if available.
/// This test is skipped if the test model file doesn't exist.
#[test]
fn test_load_real_gguf_model() {
    // Check if the test model file exists (downloaded separately)
    let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test-models/tiny-llm-Q4_K_M.gguf");
    
    if !model_path.exists() {
        println!("Skipping test_load_real_gguf_model: test model not found at {:?}", model_path);
        return;
    }
    
    let model = Model::load_from_gguf(&model_path).expect("Failed to load real test model");
    
    // Verify model loaded with reasonable parameters
    assert!(model.n_embd > 0);
    assert!(model.n_head > 0);
    assert!(model.vocab_size > 0);
    assert!(model.n_layers() > 0);
    
    println!("Loaded real model: {}", model.summary());
}
