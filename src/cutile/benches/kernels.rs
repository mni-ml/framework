//! Criterion benchmarks for the cuTile backend.
//!
//! ## Scope
//!
//! Each bench times a cuTile op **end-to-end** — the criterion closure issues
//! the kernel launch and calls `sync_on(&rt.stream)`, so the reported number
//! is `launch + device execution + sync + free`.  This matches what mni-ml
//! sees from a per-op NAPI call.
//!
//! The set of benched ops is the subset that mni-ml actually exercises on the
//! training critical path: forward + backward for the big layers
//! (matmul, gelu, softmax, layernorm, flash-attention, conv2d, embedding,
//! cross-entropy, dropout), the optimizer step (`adamw`), the data-loader
//! (`sample_batch`), and `sum_all` (the reduction that powers grad-norm).
//! Internal building blocks that have no direct user surface (elementwise
//! `add`/`saxpy`, etc.) are not benched — they're exercised transitively
//! through the layers above.
//!
//! ## Comparing to the CUDA C++ SIMT backend
//!
//! `src/native` is a sibling crate with the same op surface — running the
//! same mni-ml training loop under each backend produces directly comparable
//! end-to-end wall-clocks, which is the comparison that matters for the
//! production path.  These in-crate benches intentionally don't link against
//! `src/native` so the two backends stay decoupled.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mni_framework_cutile::ops::{
    activation, attention, conv, data, dropout, embedding, matmul, norm, optimizer, reduce,
    softmax,
};
use mni_framework_cutile::tensor::TensorStore;
use std::hint::black_box;

// NOTE: `ops::loss::cross_entropy_{forward,backward}` is on the mni-ml critical
// path and should be benched here, but the forward kernel hits a pre-existing
// cuTile-IR shape-metadata mismatch on `picked_row + eps` (concrete vs
// symbolic `BM` after a `reduce_sum`).  Out of scope for this PR — tracked
// separately; revisit once a minimal repro lands in `tests/fused_ops.rs`.

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    // Square matmuls (M=N=K).  Sizes divisible by the tile picker (mult. of 64).
    for &sz in &[64usize, 128, 256, 512, 1024] {
        let (m, n, k) = (sz, sz, sz);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.07).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.05).collect();
        group.throughput(Throughput::Elements((m * n * k) as u64));

        let mut store = TensorStore::new();
        let ia = store.from_slice(&a, &[m, k]);
        let ib = store.from_slice(&b, &[k, n]);
        let warm = matmul::matmul(&mut store, ia, ib);
        store.free(warm);
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
    group.finish();
}

fn bench_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gelu");
    for &n in &[1 << 14usize, 1 << 18, 1 << 20, 1 << 22] {
        let x: Vec<f32> = (0..n).map(|i| ((i % 257) as f32) * 0.01 - 1.0).collect();
        let dy: Vec<f32> = (0..n).map(|i| ((i % 101) as f32) * 0.02 - 1.0).collect();
        group.throughput(Throughput::Elements(n as u64));

        let mut store = TensorStore::new();
        let ix = store.from_slice(&x, &[n]);
        let idy = store.from_slice(&dy, &[n]);
        let warm_fwd = activation::gelu(&mut store, ix);
        store.free(warm_fwd);
        let warm_bwd = activation::gelu_backward(&mut store, idy, ix);
        store.free(warm_bwd);

        group.bench_with_input(BenchmarkId::new("cutile_fwd", n), &n, |bch, _| {
            bch.iter(|| {
                let id = activation::gelu(&mut store, ix);
                black_box(id);
                store.free(id);
            });
        });
        group.bench_with_input(BenchmarkId::new("cutile_bwd", n), &n, |bch, _| {
            bch.iter(|| {
                let id = activation::gelu_backward(&mut store, idy, ix);
                black_box(id);
                store.free(id);
            });
        });
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");
    for &(n, c_dim) in &[(64usize, 256usize), (256, 1024), (1024, 1024), (4096, 1024)] {
        let total = n * c_dim;
        let x: Vec<f32> = (0..total).map(|i| ((i % 17) as f32) * 0.05 - 0.4).collect();
        let dy: Vec<f32> = (0..total).map(|i| ((i % 23) as f32) * 0.03 - 0.3).collect();
        group.throughput(Throughput::Elements(total as u64));

        let mut store = TensorStore::new();
        let ix = store.from_slice(&x, &[n, c_dim]);
        let idy = store.from_slice(&dy, &[n, c_dim]);
        let warm_fwd = softmax::softmax_forward(&mut store, ix, 1);
        let warm_bwd = softmax::softmax_backward(&mut store, idy, warm_fwd, 1);
        store.free(warm_bwd);

        group.bench_with_input(
            BenchmarkId::new("cutile_fwd", format!("{n}x{c_dim}")),
            &total,
            |bch, _| {
                bch.iter(|| {
                    let id = softmax::softmax_forward(&mut store, ix, 1);
                    black_box(id);
                    store.free(id);
                });
            },
        );
        // Backward uses the softmax output `warm_fwd` (kept live) + dy.
        group.bench_with_input(
            BenchmarkId::new("cutile_bwd", format!("{n}x{c_dim}")),
            &total,
            |bch, _| {
                bch.iter(|| {
                    let id = softmax::softmax_backward(&mut store, idy, warm_fwd, 1);
                    black_box(id);
                    store.free(id);
                });
            },
        );
        store.free(warm_fwd);
    }
    group.finish();
}

fn bench_layernorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("layernorm");
    for &(n, c_dim) in &[(64usize, 256usize), (256, 1024), (1024, 1024), (4096, 1024)] {
        let total = n * c_dim;
        let x: Vec<f32> = (0..total).map(|i| ((i % 19) as f32) * 0.02 - 0.2).collect();
        let dy: Vec<f32> = (0..total).map(|i| ((i % 13) as f32) * 0.01 - 0.1).collect();
        let g: Vec<f32> = (0..c_dim).map(|i| 1.0 + (i as f32) * 0.001).collect();
        let bv: Vec<f32> = (0..c_dim).map(|i| (i as f32) * 0.0005).collect();
        group.throughput(Throughput::Elements(total as u64));

        let mut store = TensorStore::new();
        let ix = store.from_slice(&x, &[n, c_dim]);
        let idy = store.from_slice(&dy, &[n, c_dim]);
        let ig = store.from_slice(&g, &[c_dim]);
        let ib = store.from_slice(&bv, &[c_dim]);
        // Run forward once to get mean/rstd for the backward bench.
        let warm = norm::layernorm_forward(&mut store, ix, ig, ib, 1e-5);
        store.free(warm.out);
        let mean = warm.mean;
        let rstd = warm.rstd;
        // Warm up backward too.
        let wb = norm::layernorm_backward(&mut store, idy, ix, mean, rstd, ig);
        store.free(wb.dx);
        store.free(wb.dgamma);
        store.free(wb.dbeta);

        group.bench_with_input(
            BenchmarkId::new("cutile_fwd", format!("{n}x{c_dim}")),
            &total,
            |bch, _| {
                bch.iter(|| {
                    let st = norm::layernorm_forward(&mut store, ix, ig, ib, 1e-5);
                    black_box(&st);
                    store.free(st.out);
                    store.free(st.mean);
                    store.free(st.rstd);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cutile_bwd", format!("{n}x{c_dim}")),
            &total,
            |bch, _| {
                bch.iter(|| {
                    let st = norm::layernorm_backward(&mut store, idy, ix, mean, rstd, ig);
                    black_box(&st);
                    store.free(st.dx);
                    store.free(st.dgamma);
                    store.free(st.dbeta);
                });
            },
        );
        store.free(mean);
        store.free(rstd);
    }
    group.finish();
}

fn bench_flash_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention");
    for &(bh, s, d) in &[(4usize, 64usize, 64usize), (4, 256, 64), (8, 512, 64)] {
        let total = bh * s * d;
        let scale = 1.0 / (d as f32).sqrt();
        let q: Vec<f32> = (0..total).map(|i| ((i % 13) as f32) * 0.01 - 0.05).collect();
        let k: Vec<f32> = (0..total).map(|i| ((i % 11) as f32) * 0.012 - 0.05).collect();
        let v: Vec<f32> = (0..total).map(|i| ((i % 17) as f32) * 0.008 - 0.05).collect();
        let dy: Vec<f32> = (0..total).map(|i| ((i % 19) as f32) * 0.01 - 0.05).collect();
        group.throughput(Throughput::Elements(total as u64));

        let mut store = TensorStore::new();
        let iq = store.from_slice(&q, &[bh, s, d]);
        let ik = store.from_slice(&k, &[bh, s, d]);
        let iv = store.from_slice(&v, &[bh, s, d]);
        let idy = store.from_slice(&dy, &[bh, s, d]);
        let warm = attention::flash_attention_forward(&mut store, iq, ik, iv, scale, false);
        let out = warm.out;
        let lse = warm.lse;
        let wb = attention::flash_attention_backward(
            &mut store, idy, iq, ik, iv, out, lse, scale, false,
        );
        store.free(wb.dq);
        store.free(wb.dk);
        store.free(wb.dv);

        group.bench_with_input(
            BenchmarkId::new("cutile_fwd", format!("bh{bh}_s{s}_d{d}")),
            &total,
            |bch, _| {
                bch.iter(|| {
                    let st =
                        attention::flash_attention_forward(&mut store, iq, ik, iv, scale, false);
                    black_box(&st);
                    store.free(st.out);
                    store.free(st.lse);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cutile_bwd", format!("bh{bh}_s{s}_d{d}")),
            &total,
            |bch, _| {
                bch.iter(|| {
                    let st = attention::flash_attention_backward(
                        &mut store, idy, iq, ik, iv, out, lse, scale, false,
                    );
                    black_box(&st);
                    store.free(st.dq);
                    store.free(st.dk);
                    store.free(st.dv);
                });
            },
        );
        store.free(out);
        store.free(lse);
    }
    group.finish();
}

fn bench_conv2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d");
    // `conv2d_backward_input` / `..._weight` take CO/KH/KW as raw const
    // generics into an iota → broadcast chain, and cuTile IR requires all
    // broadcast dims to be powers of two.  Forward has a padded-generic
    // workaround (CIP/KHP/KWP) but backward doesn't, so stick to k=2 here.
    for &(n, c_in, h, w, c_out, k) in
        &[(2usize, 16usize, 32usize, 32usize, 32usize, 2usize), (2, 32, 64, 64, 64, 2)]
    {
        let stride = 1usize;
        let padding = 0usize;
        let inp_total = n * c_in * h * w;
        let w_total = c_out * c_in * k * k;
        let x: Vec<f32> = (0..inp_total)
            .map(|i| ((i % 23) as f32) * 0.01 - 0.1)
            .collect();
        let wt: Vec<f32> = (0..w_total).map(|i| ((i % 31) as f32) * 0.005 - 0.075).collect();
        let h_out = (h + 2 * padding - k) / stride + 1;
        let w_out = (w + 2 * padding - k) / stride + 1;
        let out_total = n * c_out * h_out * w_out;
        let dy: Vec<f32> = (0..out_total).map(|i| ((i % 29) as f32) * 0.007 - 0.1).collect();
        group.throughput(Throughput::Elements(out_total as u64));

        let mut store = TensorStore::new();
        let ix = store.from_slice(&x, &[n, c_in, h, w]);
        let iw = store.from_slice(&wt, &[c_out, c_in, k, k]);
        let idy = store.from_slice(&dy, &[n, c_out, h_out, w_out]);
        let warm = conv::conv2d_forward(&mut store, ix, iw, stride, padding);
        store.free(warm);
        let wb_i = conv::conv2d_backward_input(&mut store, idy, iw, h, w, stride, padding);
        store.free(wb_i);
        let wb_w = conv::conv2d_backward_weight(&mut store, idy, ix, k, k, stride, padding);
        store.free(wb_w);

        let label = format!("n{n}_ci{c_in}_co{c_out}_{h}x{w}_k{k}");
        group.bench_with_input(
            BenchmarkId::new("cutile_fwd", &label),
            &out_total,
            |bch, _| {
                bch.iter(|| {
                    let id = conv::conv2d_forward(&mut store, ix, iw, stride, padding);
                    black_box(id);
                    store.free(id);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cutile_bwd_input", &label),
            &out_total,
            |bch, _| {
                bch.iter(|| {
                    let id =
                        conv::conv2d_backward_input(&mut store, idy, iw, h, w, stride, padding);
                    black_box(id);
                    store.free(id);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cutile_bwd_weight", &label),
            &out_total,
            |bch, _| {
                bch.iter(|| {
                    let id =
                        conv::conv2d_backward_weight(&mut store, idy, ix, k, k, stride, padding);
                    black_box(id);
                    store.free(id);
                });
            },
        );
    }
    group.finish();
}

fn bench_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding");
    for &(vocab, d, t) in &[(1024usize, 128usize, 512usize), (4096, 256, 2048)] {
        let weight: Vec<f32> = (0..vocab * d).map(|i| ((i % 37) as f32) * 0.003 - 0.05).collect();
        let indices: Vec<i32> = (0..t).map(|i| ((i * 7 + 3) % vocab) as i32).collect();
        let dy: Vec<f32> = (0..t * d).map(|i| ((i % 29) as f32) * 0.01 - 0.1).collect();
        group.throughput(Throughput::Elements(t as u64));

        let mut store = TensorStore::new();
        let iw = store.from_slice(&weight, &[vocab, d]);
        let idy = store.from_slice(&dy, &[t, d]);
        let warm_fwd = embedding::embedding_forward(&mut store, iw, &indices);
        store.free(warm_fwd);
        let warm_bwd = embedding::embedding_backward(&mut store, idy, &indices, vocab);
        store.free(warm_bwd);

        let label = format!("v{vocab}_d{d}_t{t}");
        group.bench_with_input(BenchmarkId::new("cutile_fwd", &label), &t, |bch, _| {
            bch.iter(|| {
                let id = embedding::embedding_forward(&mut store, iw, &indices);
                black_box(id);
                store.free(id);
            });
        });
        group.bench_with_input(BenchmarkId::new("cutile_bwd", &label), &t, |bch, _| {
            bch.iter(|| {
                let id = embedding::embedding_backward(&mut store, idy, &indices, vocab);
                black_box(id);
                store.free(id);
            });
        });
    }
    group.finish();
}

fn bench_dropout(c: &mut Criterion) {
    let mut group = c.benchmark_group("dropout");
    for &n in &[1 << 16usize, 1 << 20, 1 << 22] {
        let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.01 - 0.1).collect();
        let mask: Vec<f32> = (0..n).map(|i| if i % 10 < 8 { 1.0 } else { 0.0 }).collect();
        let dy: Vec<f32> = (0..n).map(|i| ((i % 23) as f32) * 0.01 - 0.1).collect();
        let p = 0.2f32;
        group.throughput(Throughput::Elements(n as u64));

        let mut store = TensorStore::new();
        let ix = store.from_slice(&x, &[n]);
        let im = store.from_slice(&mask, &[n]);
        let idy = store.from_slice(&dy, &[n]);
        let warm_fwd = dropout::dropout_apply(&mut store, ix, im, p);
        store.free(warm_fwd);
        let warm_bwd = dropout::dropout_backward(&mut store, idy, im, p);
        store.free(warm_bwd);

        group.bench_with_input(BenchmarkId::new("cutile_fwd", n), &n, |bch, _| {
            bch.iter(|| {
                let id = dropout::dropout_apply(&mut store, ix, im, p);
                black_box(id);
                store.free(id);
            });
        });
        group.bench_with_input(BenchmarkId::new("cutile_bwd", n), &n, |bch, _| {
            bch.iter(|| {
                let id = dropout::dropout_backward(&mut store, idy, im, p);
                black_box(id);
                store.free(id);
            });
        });
    }
    group.finish();
}

fn bench_adamw(c: &mut Criterion) {
    let mut group = c.benchmark_group("adamw_step");
    for &n in &[1 << 16usize, 1 << 20, 1 << 22] {
        let init: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.01).collect();
        group.throughput(Throughput::Elements(n as u64));

        let mut store = TensorStore::new();
        let param = store.from_slice(&init, &[n]);
        let m = store.from_slice(&vec![0.0f32; n], &[n]);
        let v = store.from_slice(&vec![0.0f32; n], &[n]);
        let grad = store.from_slice(&init, &[n]);
        // Warm.
        optimizer::adamw_step(
            &mut store, param, m, v, grad, 1e-3, 0.9, 0.999, 1e-8, 0.01, 0.1, 0.001,
        );

        group.bench_with_input(BenchmarkId::new("cutile", n), &n, |bch, _| {
            bch.iter(|| {
                optimizer::adamw_step(
                    &mut store, param, m, v, grad, 1e-3, 0.9, 0.999, 1e-8, 0.01, 0.1, 0.001,
                );
            });
        });
    }
    group.finish();
}

fn bench_sample_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_batch");
    for &(dataset_len, batch_size, block_size) in
        &[(1 << 20usize, 16usize, 64usize), (1 << 22, 32, 256)]
    {
        let host: Vec<i32> = (0..dataset_len).map(|i| i as i32).collect();
        let dataset = data::Dataset::from_host(&host);
        let offsets: Vec<i32> = (0..batch_size)
            .map(|i| ((i * 4099) % (dataset_len - block_size - 1)) as i32)
            .collect();
        group.throughput(Throughput::Elements((batch_size * block_size) as u64));
        // Warm up.
        let _ = data::sample_batch(&dataset, &offsets, block_size);

        let label = format!("b{batch_size}_bs{block_size}");
        group.bench_with_input(BenchmarkId::new("cutile", &label), &batch_size, |bch, _| {
            bch.iter(|| {
                let out = data::sample_batch(&dataset, &offsets, block_size);
                black_box(&out);
            });
        });
    }
    group.finish();
}

fn bench_sum_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_all");
    for &n in &[1 << 14usize, 1 << 20, 1 << 22] {
        let a: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.01).collect();
        group.throughput(Throughput::Elements(n as u64));

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
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_gelu,
    bench_softmax,
    bench_layernorm,
    bench_flash_attention,
    bench_conv2d,
    bench_embedding,
    bench_dropout,
    bench_adamw,
    bench_sample_batch,
    bench_sum_all,
);
criterion_main!(benches);
