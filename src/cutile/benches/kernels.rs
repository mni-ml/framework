//! Criterion benchmarks for the cuTile backend.
//!
//! Compares cuTile kernels against naive CPU implementations so we can report
//! a speed-up.  The CPU baseline is a single-threaded tight loop in release
//! mode; it is not meant to be a competitive baseline but provides a
//! familiar reference point.  Each bench does a warm-up pass before timing.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mni_framework_cutile::ops::{elementwise, matmul, reduce};
use mni_framework_cutile::tensor::TensorStore;
use std::hint::black_box;

fn cpu_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn cpu_saxpy(a: f32, x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(xi, yi)| a * xi + yi).collect()
}

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");
    for &n in &[1024usize, 1 << 14, 1 << 20, 1 << 22] {
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.125).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25).collect();
        group.throughput(Throughput::Elements(n as u64));

        // cuTile path — allocate once, reuse the IDs each iteration.
        {
            let mut store = TensorStore::new();
            let ia = store.from_slice(&a, &[n]);
            let ib = store.from_slice(&b, &[n]);
            // Warm up: triggers PTX compile + cache.
            let _ = elementwise::add(&mut store, ia, ib);
            group.bench_with_input(BenchmarkId::new("cutile", n), &n, |bch, _| {
                bch.iter(|| {
                    let id = elementwise::add(&mut store, ia, ib);
                    black_box(id);
                    store.free(id);
                });
            });
        }

        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bch, _| {
            bch.iter(|| black_box(cpu_add(black_box(&a), black_box(&b))));
        });
    }
    group.finish();
}

fn bench_saxpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("saxpy");
    for &n in &[1024usize, 1 << 14, 1 << 20, 1 << 22] {
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.125).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        group.throughput(Throughput::Elements(n as u64));

        {
            let mut store = TensorStore::new();
            let ix = store.from_slice(&x, &[n]);
            let iy = store.from_slice(&y, &[n]);
            let _ = elementwise::saxpy(&mut store, 2.5, ix, iy); // warm up
            group.bench_with_input(BenchmarkId::new("cutile", n), &n, |bch, _| {
                bch.iter(|| {
                    let id = elementwise::saxpy(&mut store, 2.5, ix, iy);
                    black_box(id);
                    store.free(id);
                });
            });
        }

        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bch, _| {
            bch.iter(|| black_box(cpu_saxpy(2.5, black_box(&x), black_box(&y))));
        });
    }
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    // Square matmuls — (M=N=K).  Sizes chosen to be divisible by our tile
    // picker (all multiples of 64).
    for &sz in &[64usize, 128, 256, 512] {
        let m = sz;
        let n = sz;
        let k = sz;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.07).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.05).collect();
        let flops = 2.0 * (m * n * k) as f64;
        group.throughput(Throughput::Elements((m * n * k) as u64));

        {
            let mut store = TensorStore::new();
            let ia = store.from_slice(&a, &[m, k]);
            let ib = store.from_slice(&b, &[k, n]);
            let _ = matmul::matmul(&mut store, ia, ib); // warm up
            group.bench_with_input(
                BenchmarkId::new("cutile", format!("{m}x{n}x{k}")),
                &sz,
                |bch, _| {
                    bch.iter(|| {
                        let id = matmul::matmul(&mut store, ia, ib);
                        black_box(id);
                        store.free(id);
                    });
                },
            );
        }

        // CPU baseline only for smaller sizes — it gets prohibitive beyond 512.
        if sz <= 256 {
            group.bench_with_input(
                BenchmarkId::new("cpu", format!("{m}x{n}x{k}")),
                &sz,
                |bch, _| {
                    bch.iter(|| {
                        black_box(cpu_matmul(
                            black_box(&a),
                            black_box(&b),
                            black_box(m),
                            black_box(n),
                            black_box(k),
                        ))
                    });
                },
            );
        }

        eprintln!("matmul {m}x{n}x{k}: {:.2} GFLOPs per call", flops / 1e9);
    }
    group.finish();
}

fn bench_sum_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_all");
    for &n in &[1 << 14usize, 1 << 20, 1 << 22] {
        let a: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.01).collect();
        group.throughput(Throughput::Elements(n as u64));

        {
            let mut store = TensorStore::new();
            let ia = store.from_slice(&a, &[n]);
            let warm = reduce::sum_all(&mut store, ia);
            store.free(warm);
            group.bench_with_input(BenchmarkId::new("cutile", n), &n, |bch, _| {
                bch.iter(|| {
                    let id = reduce::sum_all(&mut store, ia);
                    black_box(id);
                    store.free(id);
                });
            });
        }

        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bch, _| {
            bch.iter(|| {
                let s: f32 = black_box(&a).iter().copied().sum();
                black_box(s)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_add, bench_saxpy, bench_matmul, bench_sum_all);
criterion_main!(benches);
