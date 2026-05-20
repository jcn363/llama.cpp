use criterion::{criterion_group, criterion_main, Criterion};
use ggml::{DType, Tensor};

fn tensor_creation_benchmark(c: &mut Criterion) {
    c.bench_function("tensor_create_f32_256x256", |b| {
        b.iter(|| Tensor::new(DType::F32, &[256, 256]))
    });
}

criterion_group!(benches, tensor_creation_benchmark);
criterion_main!(benches);
