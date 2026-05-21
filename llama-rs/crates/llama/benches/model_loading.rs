use criterion::{black_box, criterion_group, criterion_main, Criterion};
use llama::Model;
use std::path::Path;

/// Benchmark model loading from a GGUF file.
fn bench_model_loading(c: &mut Criterion) {
    // Try to find the test model file
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test-models/tiny-llm-Q4_K_M.gguf");
    
    if !model_path.exists() {
        println!("Skipping model loading benchmark: test model not found");
        return;
    }
    
    c.bench_function("load_model_from_gguf", |b| {
        b.iter(|| {
            let model = Model::load_from_gguf(&model_path).expect("Failed to load model");
            black_box(model.summary())
        })
    });
}

/// Benchmark tensor de-quantization.
fn bench_dequantization(c: &mut Criterion) {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test-models/tiny-llm-Q4_K_M.gguf");
    
    if !model_path.exists() {
        println!("Skipping dequantization benchmark: test model not found");
        return;
    }
    
    let model = Model::load_from_gguf(&model_path).expect("Failed to load model");
    
    // Find a tensor that can be de-quantized (skip unsupported types)
    let tensor_id = model.tensors.keys().find(|&&id| {
        model.tensors.get(&id).map(|t| {
            // Try to dequantize to see if it's supported
            t.get().is_ok()
        }).unwrap_or(false)
    }).copied().expect("No dequantizable tensors found");
    
    let tensor = model.tensors.get(&tensor_id).expect("Tensor not found");
    let tensor_name = model.interned.get(tensor_id).unwrap_or("unknown");
    
    c.bench_function(&format!("dequantize_tensor_{}", tensor_name), |b| {
        b.iter(|| {
            let data = tensor.get().expect("Failed to dequantize");
            black_box(data.len())
        })
    });
}

criterion_group!(benches, bench_model_loading, bench_dequantization);
criterion_main!(benches);
