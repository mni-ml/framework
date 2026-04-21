//! Correctness tests for the cuTile backend.
//!
//! Each test compares cuTile output against a CPU reference.  Tolerance is
//! 1e-4 for elementwise ops and 2e-3 (relative) for GEMM, which matches
//! typical fp32 reductions.

use mni_framework_cutile::ops::{activation, elementwise, matmul, reduce};
use mni_framework_cutile::tensor::TensorStore;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol + tol * a.abs().max(b.abs())
}

fn assert_slice_close(got: &[f32], expect: &[f32], tol: f32, ctx: &str) {
    assert_eq!(
        got.len(),
        expect.len(),
        "{ctx}: length mismatch got={} expect={}",
        got.len(),
        expect.len()
    );
    let mut bad = 0usize;
    for (i, (&g, &e)) in got.iter().zip(expect.iter()).enumerate() {
        if !approx_eq(g, e, tol) {
            if bad < 4 {
                eprintln!("{ctx}: mismatch at {i}: got={} expect={}", g, e);
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{ctx}: {bad} mismatches");
}

fn make_ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * scale).collect()
}

#[test]
fn test_add() {
    let mut s = TensorStore::new();
    let n = 1024;
    let a = make_ramp(n, 0.5);
    let b = make_ramp(n, -0.25);
    let ida = s.from_slice(&a, &[n]);
    let idb = s.from_slice(&b, &[n]);
    let idz = elementwise::add(&mut s, ida, idb);
    let got = s.to_host(idz);
    let want: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    assert_slice_close(&got, &want, 1e-5, "add");
}

#[test]
fn test_sub() {
    let mut s = TensorStore::new();
    let n = 1024;
    let a = make_ramp(n, 0.5);
    let b = make_ramp(n, -0.25);
    let ida = s.from_slice(&a, &[n]);
    let idb = s.from_slice(&b, &[n]);
    let idz = elementwise::sub(&mut s, ida, idb);
    let got = s.to_host(idz);
    let want: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    assert_slice_close(&got, &want, 1e-5, "sub");
}

#[test]
fn test_mul() {
    let mut s = TensorStore::new();
    let n = 512;
    let a = make_ramp(n, 0.1);
    let b = make_ramp(n, 0.2);
    let ida = s.from_slice(&a, &[n]);
    let idb = s.from_slice(&b, &[n]);
    let idz = elementwise::mul(&mut s, ida, idb);
    let got = s.to_host(idz);
    let want: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
    assert_slice_close(&got, &want, 1e-4, "mul");
}

#[test]
fn test_div() {
    let mut s = TensorStore::new();
    let n = 512;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
    let b: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.5).collect();
    let ida = s.from_slice(&a, &[n]);
    let idb = s.from_slice(&b, &[n]);
    let idz = elementwise::div(&mut s, ida, idb);
    let got = s.to_host(idz);
    let want: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x / y).collect();
    assert_slice_close(&got, &want, 1e-4, "div");
}

#[test]
fn test_neg() {
    let mut s = TensorStore::new();
    let n = 256;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.125 - 10.0).collect();
    let ida = s.from_slice(&a, &[n]);
    let idz = elementwise::neg(&mut s, ida);
    let got = s.to_host(idz);
    let want: Vec<f32> = a.iter().map(|x| -x).collect();
    assert_slice_close(&got, &want, 1e-5, "neg");
}

#[test]
fn test_mul_scalar() {
    let mut s = TensorStore::new();
    let n = 1024;
    let a = make_ramp(n, 0.25);
    let ida = s.from_slice(&a, &[n]);
    let idz = elementwise::mul_scalar(&mut s, ida, 3.5);
    let got = s.to_host(idz);
    let want: Vec<f32> = a.iter().map(|x| x * 3.5).collect();
    assert_slice_close(&got, &want, 1e-4, "mul_scalar");
}

#[test]
fn test_saxpy() {
    let mut s = TensorStore::new();
    let n = 1024;
    let x = make_ramp(n, 0.125);
    let y = make_ramp(n, 0.5);
    let ix = s.from_slice(&x, &[n]);
    let iy = s.from_slice(&y, &[n]);
    let iz = elementwise::saxpy(&mut s, 2.5, ix, iy);
    let got = s.to_host(iz);
    let want: Vec<f32> = x.iter().zip(y.iter()).map(|(a, b)| 2.5 * a + b).collect();
    assert_slice_close(&got, &want, 1e-4, "saxpy");
}

#[test]
fn test_relu() {
    let mut s = TensorStore::new();
    let n = 256;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 32.0).collect();
    let ida = s.from_slice(&a, &[n]);
    let idz = activation::relu(&mut s, ida);
    let got = s.to_host(idz);
    let want: Vec<f32> = a.iter().map(|x| x.max(0.0)).collect();
    assert_slice_close(&got, &want, 1e-5, "relu");
}

#[test]
fn test_relu_backward() {
    let mut s = TensorStore::new();
    let n = 256;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 32.0).collect();
    let g: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let ix = s.from_slice(&x, &[n]);
    let ig = s.from_slice(&g, &[n]);
    let idx = activation::relu_backward(&mut s, ix, ig);
    let got = s.to_host(idx);
    let want: Vec<f32> = x
        .iter()
        .zip(g.iter())
        .map(|(xi, gi)| if *xi > 0.0 { *gi } else { 0.0 })
        .collect();
    assert_slice_close(&got, &want, 1e-4, "relu_backward");
}

#[test]
fn test_matmul_small() {
    // 4x3 @ 3x2
    let mut s = TensorStore::new();
    let a: Vec<f32> = vec![
        1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, //
        1.0, 0.0, -1.0,
    ];
    let b: Vec<f32> = vec![
        1.0, 0.0, //
        0.0, 1.0, //
        1.0, 1.0,
    ];
    let ia = s.from_slice(&a, &[4, 3]);
    let ib = s.from_slice(&b, &[3, 2]);
    let iz = matmul::matmul(&mut s, ia, ib);
    let got = s.to_host(iz);
    // row_i . col_j
    let want: Vec<f32> = vec![
        1.0 + 0.0 + 3.0,
        0.0 + 2.0 + 3.0,
        4.0 + 0.0 + 6.0,
        0.0 + 5.0 + 6.0,
        7.0 + 0.0 + 9.0,
        0.0 + 8.0 + 9.0,
        1.0 + 0.0 + -1.0,
        0.0 + 0.0 + -1.0,
    ];
    assert_slice_close(&got, &want, 1e-4, "matmul 4x3 @ 3x2");
    assert_eq!(s.shape(iz), &[4, 2]);
}

#[test]
fn test_matmul_square() {
    // 32x32 @ 32x32, all-ones → every entry = 32.
    let mut s = TensorStore::new();
    let a = vec![1.0f32; 32 * 32];
    let b = vec![1.0f32; 32 * 32];
    let ia = s.from_slice(&a, &[32, 32]);
    let ib = s.from_slice(&b, &[32, 32]);
    let iz = matmul::matmul(&mut s, ia, ib);
    let got = s.to_host(iz);
    assert_eq!(got.len(), 32 * 32);
    for (i, &v) in got.iter().enumerate() {
        assert!(
            approx_eq(v, 32.0, 1e-3),
            "matmul square: got[{i}]={v} want=32.0"
        );
    }
}

#[test]
fn test_matmul_rect() {
    // 64x128 @ 128x256 with deterministic data.
    let m = 64;
    let k = 128;
    let n = 256;
    let mut s = TensorStore::new();
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.2 - 0.5).collect();
    let ia = s.from_slice(&a, &[m, k]);
    let ib = s.from_slice(&b, &[k, n]);
    let iz = matmul::matmul(&mut s, ia, ib);
    let got = s.to_host(iz);

    // CPU reference.
    let mut want = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            want[i * n + j] = acc;
        }
    }
    assert_slice_close(&got, &want, 2e-3, "matmul 64x128 @ 128x256");
}

#[test]
fn test_matmul_batched_leading() {
    // 2x3x4 @ 4x5 → 2x3x5  (leading 2x3 collapsed into M=6).
    let mut s = TensorStore::new();
    let a: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..4 * 5).map(|i| (i as f32) * 0.05).collect();
    let ia = s.from_slice(&a, &[2, 3, 4]);
    let ib = s.from_slice(&b, &[4, 5]);
    let iz = matmul::matmul(&mut s, ia, ib);
    let got = s.to_host(iz);
    assert_eq!(s.shape(iz), &[2, 3, 5]);

    let mut want = vec![0.0f32; 2 * 3 * 5];
    for i in 0..6 {
        for j in 0..5 {
            let mut acc = 0.0f32;
            for kk in 0..4 {
                acc += a[i * 4 + kk] * b[kk * 5 + j];
            }
            want[i * 5 + j] = acc;
        }
    }
    assert_slice_close(&got, &want, 1e-3, "matmul batched");
}

#[test]
fn test_reduce_sum_all() {
    let mut s = TensorStore::new();
    let n = 1024;
    let a = make_ramp(n, 0.5);
    let ida = s.from_slice(&a, &[32, 32]);
    let ids = reduce::sum_all(&mut s, ida);
    let got = s.to_host(ids);
    let want: f32 = a.iter().copied().sum();
    assert_eq!(s.shape(ids), &[1]);
    assert!(
        approx_eq(got[0], want, 1e-3),
        "sum_all: got={} want={}",
        got[0],
        want
    );
}

#[test]
fn test_reduce_sum_all_multipass() {
    // 1M elements => with BLOCK=256 this is 3 passes (1M -> 4K -> 16 -> 1),
    // exercising the ping-pong between partial buffers.
    let mut s = TensorStore::new();
    let n = 1 << 20;
    // Small values so the fp32 sum doesn't explode.
    let a: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.01).collect();
    let ida = s.from_slice(&a, &[n]);
    let ids = reduce::sum_all(&mut s, ida);
    let got = s.to_host(ids);
    let want: f32 = a.iter().copied().sum();
    assert_eq!(s.shape(ids), &[1]);
    // 1e-3 rel tolerance is generous for a 1M-elt fp32 reduction.  The GPU
    // tree reduction and the CPU serial sum will disagree in the last few
    // bits; the tree answer is usually more accurate but we just want "same
    // ballpark."
    let rel = (got[0] - want).abs() / want.abs().max(1.0);
    assert!(
        rel < 1e-3,
        "sum_all multipass: got={} want={} rel={}",
        got[0],
        want,
        rel
    );
}

#[test]
fn test_reduce_sum_all_host_tail() {
    // 510 = 2 * 3 * 5 * 17 — BLOCK=2 reduces to 255, which has no divisor
    // in {256,128,...,2}, so the tail finishes on the host.
    let mut s = TensorStore::new();
    let n = 510;
    let a = make_ramp(n, 0.1);
    let ida = s.from_slice(&a, &[n]);
    let ids = reduce::sum_all(&mut s, ida);
    let got = s.to_host(ids);
    let want: f32 = a.iter().copied().sum();
    assert_eq!(s.shape(ids), &[1]);
    assert!(
        approx_eq(got[0], want, 1e-3),
        "sum_all host tail: got={} want={}",
        got[0],
        want
    );
}

#[test]
fn test_reduce_mean_all() {
    let mut s = TensorStore::new();
    let n = 2048;
    let a = make_ramp(n, 0.25);
    let ida = s.from_slice(&a, &[16, 128]);
    let idm = reduce::mean_all(&mut s, ida);
    let got = s.to_host(idm);
    let want: f32 = a.iter().copied().sum::<f32>() / n as f32;
    assert_eq!(s.shape(idm), &[1]);
    assert!(
        approx_eq(got[0], want, 1e-3),
        "mean_all: got={} want={}",
        got[0],
        want
    );
}

#[test]
fn test_shape_and_free() {
    let mut s = TensorStore::new();
    let id = s.zeros(&[2, 3, 4]);
    assert_eq!(s.shape(id), &[2, 3, 4]);
    assert_eq!(s.size(id), 24);
    let host = s.to_host(id);
    assert!(host.iter().all(|&v| v == 0.0));
    s.free(id);
    // Recycled ID should work too:
    let id2 = s.ones(&[5]);
    let h2 = s.to_host(id2);
    assert!(h2.iter().all(|&v| v == 1.0));
}

#[test]
fn test_non_power_of_two_size() {
    // Size 300 = 4 * 75, divides by 4 → tile block 4 picked.
    let mut s = TensorStore::new();
    let n = 300;
    let a = make_ramp(n, 0.1);
    let b = make_ramp(n, 0.2);
    let ida = s.from_slice(&a, &[n]);
    let idb = s.from_slice(&b, &[n]);
    let idz = elementwise::add(&mut s, ida, idb);
    let got = s.to_host(idz);
    let want: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    assert_slice_close(&got, &want, 1e-5, "add n=300");
}
