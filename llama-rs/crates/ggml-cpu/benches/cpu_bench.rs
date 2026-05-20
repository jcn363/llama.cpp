use criterion::{criterion_group, criterion_main, Criterion};
use ggml::{DType, Tensor};
use ggml_cpu::CpuBackend;

fn matmul_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new(1);
    let a = Tensor::new(DType::F32, &[64, 64]);
    let b = Tensor::new(DType::F32, &[64, 64]);

    c.bench_function("cpu_matmul_64x64", |b| {
        b.iter(|| backend.matmul(&a, &b))
    });
}

criterion_group!(benches, matmul_benchmark);
criterion_main!(benches);
