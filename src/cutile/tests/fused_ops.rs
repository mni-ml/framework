//! Correctness tests for the fused cuTile op wrappers in `src/ops/`.
//!
//! Coverage: activations + their backwards, softmax + backward, LayerNorm
//! forward + backward, residual LayerNorm, bias-GELU + backward, dropout,
//! AdamW, pooling (avg + max), conv1d/2d forward, KV-cache quantize round-trip.
//!
//! Each test compares cuTile output against a CPU reference.  Tolerances are
//! generous (1e-3 absolute or 1e-3 relative) since the kernels do reductions
//! in fp32 and the CPU reference uses the same.

use approx::assert_relative_eq;
use mni_framework_cutile::ops::{
    activation, attention, conv, dropout, elementwise, fused, grad_util, kv_quant,
    mixed_precision, norm, optimizer, pooling, reduce, softmax,
};
use mni_framework_cutile::tensor::TensorStore;

const TOL_ABS: f32 = 1e-3;
const TOL_REL: f32 = 1e-3;

fn assert_close_slice(got: &[f32], expect: &[f32], ctx: &str) {
    assert_eq!(got.len(), expect.len(), "{ctx}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expect.iter()).enumerate() {
        let abs = (g - e).abs();
        let rel = abs / e.abs().max(1e-6);
        assert!(
            abs <= TOL_ABS || rel <= TOL_REL,
            "{ctx}: mismatch at {i}: got={g} expect={e} (abs={abs} rel={rel})"
        );
    }
}

fn cpu_gelu(x: f32) -> f32 {
    // Tanh-approx GELU, same as the kernel.
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = c * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn cpu_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn test_gelu_forward() {
    let mut s = TensorStore::new();
    let n = 256;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 12.0).collect();
    let id = s.from_slice(&x, &[n]);
    let out = activation::gelu(&mut s, id);
    let got = s.to_host(out);
    let want: Vec<f32> = x.iter().map(|&v| cpu_gelu(v)).collect();
    assert_close_slice(&got, &want, "gelu");
}

#[test]
fn test_gelu_backward() {
    let mut s = TensorStore::new();
    let n = 256;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 12.0).collect();
    let dy: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let ix = s.from_slice(&x, &[n]);
    let idy = s.from_slice(&dy, &[n]);
    let out = activation::gelu_backward(&mut s, idy, ix);
    let got = s.to_host(out);

    // CPU GELU derivative reference.
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let want: Vec<f32> = x
        .iter()
        .zip(dy.iter())
        .map(|(&xi, &dyi)| {
            let inner = c * (xi + 0.044715 * xi * xi * xi);
            let tanh_inner = inner.tanh();
            let dinner_dx = c * (1.0 + 3.0 * 0.044715 * xi * xi);
            let sech2 = 1.0 - tanh_inner * tanh_inner;
            let dgelu = 0.5 * (1.0 + tanh_inner) + 0.5 * xi * sech2 * dinner_dx;
            dyi * dgelu
        })
        .collect();
    assert_close_slice(&got, &want, "gelu_backward");
}

#[test]
fn test_sigmoid_forward() {
    let mut s = TensorStore::new();
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 16.0).collect();
    let id = s.from_slice(&x, &[n]);
    let out = activation::sigmoid(&mut s, id);
    let got = s.to_host(out);
    let want: Vec<f32> = x.iter().map(|&v| cpu_sigmoid(v)).collect();
    assert_close_slice(&got, &want, "sigmoid");
}

#[test]
fn test_sigmoid_backward() {
    let mut s = TensorStore::new();
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 16.0).collect();
    let dy: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let ix = s.from_slice(&x, &[n]);
    let out_fwd = activation::sigmoid(&mut s, ix);
    let idy = s.from_slice(&dy, &[n]);
    let dx_id = activation::sigmoid_backward(&mut s, idy, out_fwd);
    let got = s.to_host(dx_id);

    let want: Vec<f32> = x
        .iter()
        .zip(dy.iter())
        .map(|(&xi, &dyi)| {
            let s = cpu_sigmoid(xi);
            dyi * s * (1.0 - s)
        })
        .collect();
    assert_close_slice(&got, &want, "sigmoid_backward");
}

#[test]
fn test_softmax_forward() {
    let mut s = TensorStore::new();
    let (n, c) = (4, 64);
    let x: Vec<f32> = (0..n * c).map(|i| ((i % 13) as f32) * 0.1 - 0.5).collect();
    let id = s.from_slice(&x, &[n, c]);
    let out = softmax::softmax_forward(&mut s, id, 1);
    let got = s.to_host(out);

    // CPU reference.
    let mut want = vec![0.0f32; n * c];
    for r in 0..n {
        let row = &x[r * c..(r + 1) * c];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
        let s: f32 = exps.iter().sum();
        for (j, e) in exps.iter().enumerate() {
            want[r * c + j] = e / s;
        }
    }
    assert_close_slice(&got, &want, "softmax");
    // Row sums must be 1.
    for r in 0..n {
        let row_sum: f32 = got[r * c..(r + 1) * c].iter().sum();
        assert_relative_eq!(row_sum, 1.0, max_relative = 1e-4);
    }
}

#[test]
fn test_layernorm_forward_backward() {
    let mut s = TensorStore::new();
    let (n, c) = (4, 32);
    let x: Vec<f32> = (0..n * c)
        .map(|i| ((i % 11) as f32) * 0.1 - 0.5)
        .collect();
    let g: Vec<f32> = (0..c).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let bv: Vec<f32> = (0..c).map(|i| (i as f32) * 0.005).collect();
    let ix = s.from_slice(&x, &[n, c]);
    let ig = s.from_slice(&g, &[c]);
    let ib = s.from_slice(&bv, &[c]);

    let fwd = norm::layernorm_forward(&mut s, ix, ig, ib, 1e-5);
    let got_out = s.to_host(fwd.out);
    let got_mean = s.to_host(fwd.mean);
    let got_rstd = s.to_host(fwd.rstd);

    // CPU reference.
    let mut want_out = vec![0.0f32; n * c];
    let mut want_mean = vec![0.0f32; n];
    let mut want_rstd = vec![0.0f32; n];
    for r in 0..n {
        let row = &x[r * c..(r + 1) * c];
        let m: f32 = row.iter().copied().sum::<f32>() / c as f32;
        let var: f32 = row.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / c as f32;
        let rstd = 1.0 / (var + 1e-5).sqrt();
        want_mean[r] = m;
        want_rstd[r] = rstd;
        for j in 0..c {
            want_out[r * c + j] = (row[j] - m) * rstd * g[j] + bv[j];
        }
    }
    assert_close_slice(&got_out, &want_out, "layernorm out");
    assert_close_slice(&got_mean, &want_mean, "layernorm mean");
    assert_close_slice(&got_rstd, &want_rstd, "layernorm rstd");
}

#[test]
fn test_residual_layernorm() {
    let mut s = TensorStore::new();
    let (n, c) = (2, 32);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.05).collect();
    let r: Vec<f32> = (0..n * c).map(|i| ((i % 7) as f32) * 0.1).collect();
    let g: Vec<f32> = (0..c).map(|_| 1.0).collect();
    let b: Vec<f32> = (0..c).map(|_| 0.0).collect();
    let ix = s.from_slice(&x, &[n, c]);
    let ir = s.from_slice(&r, &[n, c]);
    let ig = s.from_slice(&g, &[c]);
    let ib = s.from_slice(&b, &[c]);
    let out = fused::residual_layernorm_forward(&mut s, ix, ir, ig, ib, 1e-5);

    let got_residual = s.to_host(out.residual);
    let got_out = s.to_host(out.out);

    let want_residual: Vec<f32> = x.iter().zip(r.iter()).map(|(a, b)| a + b).collect();
    assert_close_slice(&got_residual, &want_residual, "residual");

    // LayerNorm of residual with γ=1, β=0.
    let mut want_out = vec![0.0f32; n * c];
    for row in 0..n {
        let row_data = &want_residual[row * c..(row + 1) * c];
        let m: f32 = row_data.iter().sum::<f32>() / c as f32;
        let var: f32 =
            row_data.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / c as f32;
        let rstd = 1.0 / (var + 1e-5).sqrt();
        for j in 0..c {
            want_out[row * c + j] = (row_data[j] - m) * rstd;
        }
    }
    assert_close_slice(&got_out, &want_out, "residual_layernorm out");
}

#[test]
fn test_bias_gelu_forward_backward() {
    let mut s = TensorStore::new();
    let (n, c) = (4, 32);
    let x: Vec<f32> = (0..n * c).map(|i| ((i % 17) as f32) * 0.05 - 0.5).collect();
    let bv: Vec<f32> = (0..c).map(|i| (i as f32) * 0.01).collect();
    let dy: Vec<f32> = (0..n * c).map(|i| 1.0 + (i as f32) * 0.001).collect();

    let ix = s.from_slice(&x, &[n, c]);
    let ib = s.from_slice(&bv, &[c]);
    let fwd = fused::bias_gelu_forward(&mut s, ix, ib);
    let got_fwd = s.to_host(fwd);

    // CPU reference: GELU(x + b).
    let want_fwd: Vec<f32> = (0..n * c)
        .map(|i| {
            let r = i / c;
            let cidx = i % c;
            let _ = r;
            cpu_gelu(x[i] + bv[cidx])
        })
        .collect();
    assert_close_slice(&got_fwd, &want_fwd, "bias_gelu fwd");

    // Backward.
    let idy = s.from_slice(&dy, &[n, c]);
    let bw = fused::bias_gelu_backward(&mut s, idy, ix, ib);
    let got_dx = s.to_host(bw.dx);
    let got_db = s.to_host(bw.dbias);

    // dx = dy · GELU'(x + b);  db_c = Σ_n dx[n, c]
    let cgelu = (2.0_f32 / std::f32::consts::PI).sqrt();
    let mut want_dx = vec![0.0f32; n * c];
    let mut want_db = vec![0.0f32; c];
    for r in 0..n {
        for j in 0..c {
            let i = r * c + j;
            let v = x[i] + bv[j];
            let inner = cgelu * (v + 0.044715 * v * v * v);
            let tanh_inner = inner.tanh();
            let dinner_dv = cgelu * (1.0 + 3.0 * 0.044715 * v * v);
            let sech2 = 1.0 - tanh_inner * tanh_inner;
            let dgelu = 0.5 * (1.0 + tanh_inner) + 0.5 * v * sech2 * dinner_dv;
            want_dx[i] = dy[i] * dgelu;
            want_db[j] += want_dx[i];
        }
    }
    assert_close_slice(&got_dx, &want_dx, "bias_gelu dx");
    assert_close_slice(&got_db, &want_db, "bias_gelu dbias");
}

#[test]
fn test_dropout_apply_and_backward() {
    let mut s = TensorStore::new();
    let n = 256;
    let x: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let mask_host: Vec<f32> = (0..n).map(|i| if i % 4 == 0 { 0.0 } else { 1.0 }).collect();
    let p = 0.25_f32;
    let scale = 1.0 / (1.0 - p);

    let ix = s.from_slice(&x, &[n]);
    let im = s.from_slice(&mask_host, &[n]);
    let out = dropout::dropout_apply(&mut s, ix, im, p);
    let got = s.to_host(out);
    let want: Vec<f32> = x.iter().zip(mask_host.iter()).map(|(a, m)| a * m * scale).collect();
    assert_close_slice(&got, &want, "dropout fwd");

    // Backward: dx = dy · mask · scale.
    let dy: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.005).collect();
    let idy = s.from_slice(&dy, &[n]);
    let dx = dropout::dropout_backward(&mut s, idy, im, p);
    let got_dx = s.to_host(dx);
    let want_dx: Vec<f32> = dy.iter().zip(mask_host.iter()).map(|(d, m)| d * m * scale).collect();
    assert_close_slice(&got_dx, &want_dx, "dropout bwd");
}

#[test]
fn test_adamw_step() {
    let mut s = TensorStore::new();
    let n = 64;
    let p_init: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let g: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32) * 0.001).collect();

    let p = s.from_slice(&p_init, &[n]);
    let m = s.zeros(&[n]);
    let v = s.zeros(&[n]);
    let grad = s.from_slice(&g, &[n]);

    let lr = 1e-3_f32;
    let beta1 = 0.9_f32;
    let beta2 = 0.999_f32;
    let eps = 1e-8_f32;
    let wd = 0.01_f32;
    let t = 1;
    let bc1 = 1.0 - beta1.powi(t);
    let bc2 = 1.0 - beta2.powi(t);
    optimizer::adamw_step(&mut s, p, m, v, grad, lr, beta1, beta2, eps, wd, bc1, bc2);

    let got_p = s.to_host(p);
    let got_m = s.to_host(m);
    let got_v = s.to_host(v);

    // CPU reference for one step from m=v=0.
    let mut want_p = p_init.clone();
    let mut want_m = vec![0.0f32; n];
    let mut want_v = vec![0.0f32; n];
    for i in 0..n {
        want_m[i] = beta1 * 0.0 + (1.0 - beta1) * g[i];
        want_v[i] = beta2 * 0.0 + (1.0 - beta2) * g[i] * g[i];
        let m_hat = want_m[i] / bc1;
        let v_hat = want_v[i] / bc2;
        // AdamW: decoupled weight decay applied to params before update.
        want_p[i] = want_p[i] - lr * wd * want_p[i];
        want_p[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
    assert_close_slice(&got_m, &want_m, "adamw m");
    assert_close_slice(&got_v, &want_v, "adamw v");
    assert_close_slice(&got_p, &want_p, "adamw param");
}

#[test]
fn test_grad_clip_and_norm_sq() {
    let mut s = TensorStore::new();
    let n = 256;
    let g: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let id = s.from_slice(&g, &[n]);
    let _ = grad_util::grad_clip(&mut s, id, 0.5);
    let got = s.to_host(id);
    let want: Vec<f32> = g.iter().map(|x| x * 0.5).collect();
    assert_close_slice(&got, &want, "grad_clip");

    let id2 = s.from_slice(&g, &[n]);
    let nrm = grad_util::grad_norm_sq(&mut s, id2);
    let got_nrm = s.to_host(nrm)[0];
    let want_nrm: f32 = g.iter().map(|x| x * x).sum();
    assert_relative_eq!(got_nrm, want_nrm, max_relative = 1e-3);
}

#[test]
fn test_scale_inplace() {
    let mut s = TensorStore::new();
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let id = s.from_slice(&x, &[n]);
    let _ = mixed_precision::scale(&mut s, id, 2.0);
    let got = s.to_host(id);
    let want: Vec<f32> = x.iter().map(|v| v * 2.0).collect();
    assert_close_slice(&got, &want, "scale");
}

#[test]
fn test_check_inf_nan() {
    let mut s = TensorStore::new();
    let n = 64;

    // All finite.
    let ok: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
    let id_ok = s.from_slice(&ok, &[n]);
    let flag_ok = mixed_precision::check_inf_nan(&mut s, id_ok);
    assert_eq!(s.to_host(flag_ok)[0], 0.0);

    // One NaN.
    let mut nan = ok.clone();
    nan[3] = f32::NAN;
    let id_nan = s.from_slice(&nan, &[n]);
    let flag_nan = mixed_precision::check_inf_nan(&mut s, id_nan);
    assert_eq!(s.to_host(flag_nan)[0], 1.0);

    // One inf.
    let mut inf = ok.clone();
    inf[5] = f32::INFINITY;
    let id_inf = s.from_slice(&inf, &[n]);
    let flag_inf = mixed_precision::check_inf_nan(&mut s, id_inf);
    assert_eq!(s.to_host(flag_inf)[0], 1.0);
}

#[test]
fn test_avgpool2d_forward_backward() {
    let mut s = TensorStore::new();
    let (n, c, h, w) = (1, 1, 4, 4);
    let kh = 2;
    let kw = 2;
    let x: Vec<f32> = (0..n * c * h * w).map(|i| (i + 1) as f32).collect();
    let id = s.from_slice(&x, &[n, c, h, w]);
    let out = pooling::avgpool2d_forward(&mut s, id, kh, kw);
    let got = s.to_host(out);

    // 4x4 → 2x2 with 2x2 windows.
    let want = vec![
        (1.0 + 2.0 + 5.0 + 6.0) / 4.0,
        (3.0 + 4.0 + 7.0 + 8.0) / 4.0,
        (9.0 + 10.0 + 13.0 + 14.0) / 4.0,
        (11.0 + 12.0 + 15.0 + 16.0) / 4.0,
    ];
    assert_close_slice(&got, &want, "avgpool fwd");

    // Backward: each of the 4 inputs in a window gets dout/4.
    let dout: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let idout = s.from_slice(&dout, &[n, c, 2, 2]);
    let dinp = pooling::avgpool2d_backward(&mut s, idout, kh, kw);
    let got_dinp = s.to_host(dinp);
    let want_dinp = vec![
        0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 0.75, 0.75, 1.0, 1.0,
    ];
    assert_close_slice(&got_dinp, &want_dinp, "avgpool bwd");
}

#[test]
fn test_maxpool2d_forward_backward() {
    let mut s = TensorStore::new();
    let (n, c, h, w) = (1, 1, 4, 4);
    let kh = 2;
    let kw = 2;
    // Within each 2x2 window the max is at a known position.
    let x: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, //
        5.0, 6.0, 7.0, 8.0, //
        9.0, 10.0, 11.0, 12.0, //
        13.0, 14.0, 15.0, 16.0,
    ];
    let id = s.from_slice(&x, &[n, c, h, w]);
    let state = pooling::maxpool2d_forward(&mut s, id, kh, kw);
    let got = s.to_host(state.out);
    // Max of each 2x2 window.
    let want = vec![6.0, 8.0, 14.0, 16.0];
    assert_close_slice(&got, &want, "maxpool fwd");

    // Backward: dout flows only to the argmax position.
    let dout = vec![1.0, 2.0, 3.0, 4.0];
    let idout = s.from_slice(&dout, &[n, c, 2, 2]);
    let dinp = pooling::maxpool2d_backward(&mut s, &state, idout);
    let got_dinp = s.to_host(dinp);
    // Argmax positions: bottom-right of each window — indices 5, 7, 13, 15.
    let mut want_dinp = vec![0.0f32; 16];
    want_dinp[5] = 1.0;
    want_dinp[7] = 2.0;
    want_dinp[13] = 3.0;
    want_dinp[15] = 4.0;
    assert_close_slice(&got_dinp, &want_dinp, "maxpool bwd");
}

#[test]
fn test_kv_quant_roundtrip() {
    let mut s = TensorStore::new();
    let (n, d) = (4, 64);
    let x: Vec<f32> = (0..n * d)
        .map(|i| ((i as f32) * 0.01).sin() * 5.0)
        .collect();
    let id = s.from_slice(&x, &[n, d]);
    let q = kv_quant::quantize_rows(&s, id);
    let dq = kv_quant::dequantize_rows(&mut s, &q);
    let got = s.to_host(dq);

    // Per-row max abs error should be ≤ scale (= max(|row|)/127).
    for r in 0..n {
        let row = &x[r * d..(r + 1) * d];
        let max_abs = row.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
        let scale = (max_abs / 127.0).max(1e-8);
        for j in 0..d {
            let err = (got[r * d + j] - row[j]).abs();
            assert!(
                err <= scale,
                "kv_quant row {r} col {j}: got={} want={} err={} scale={}",
                got[r * d + j],
                row[j],
                err,
                scale
            );
        }
    }
}

#[test]
fn test_sum_along_dim_last() {
    let mut s = TensorStore::new();
    let (n, c) = (8, 16);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.05).collect();
    let id = s.from_slice(&x, &[n, c]);
    let out = reduce::sum_along_dim(&mut s, id, 1, false);
    let got = s.to_host(out);
    assert_eq!(s.shape(out), &[n]);
    let want: Vec<f32> = (0..n)
        .map(|r| x[r * c..(r + 1) * c].iter().copied().sum())
        .collect();
    assert_close_slice(&got, &want, "sum_along_dim last");
}

#[test]
fn test_broadcast_add() {
    let mut s = TensorStore::new();
    let (n, c) = (4, 8);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..c).map(|i| i as f32).collect();
    let ix = s.from_slice(&x, &[n, c]);
    let ib = s.from_slice(&b, &[c]);
    let out = elementwise::broadcast_add(&mut s, ix, ib);
    let got = s.to_host(out);
    let want: Vec<f32> = (0..n * c).map(|i| x[i] + b[i % c]).collect();
    assert_close_slice(&got, &want, "broadcast_add");
}

#[test]
fn test_conv1d_forward() {
    let mut s = TensorStore::new();
    let (n, c_in, l, c_out, k) = (2, 3, 8, 4, 3);
    let stride = 1usize;
    let padding = 1usize;
    let l_out = (l + 2 * padding - k) / stride + 1;
    let x: Vec<f32> = (0..n * c_in * l).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let w: Vec<f32> = (0..c_out * c_in * k)
        .map(|i| (i as f32) * 0.03 - 0.3)
        .collect();
    let ix = s.from_slice(&x, &[n, c_in, l]);
    let iw = s.from_slice(&w, &[c_out, c_in, k]);
    let out = conv::conv1d_forward(&mut s, ix, iw, stride, padding);
    let got = s.to_host(out);
    assert_eq!(s.shape(out), &[n, c_out, l_out]);

    // CPU reference.
    let mut want = vec![0.0f32; n * c_out * l_out];
    for ni in 0..n {
        for co in 0..c_out {
            for lo in 0..l_out {
                let mut acc = 0.0f32;
                for ci in 0..c_in {
                    for kk in 0..k {
                        let il = (lo * stride) as i32 + kk as i32 - padding as i32;
                        if il >= 0 && il < l as i32 {
                            acc += x[ni * c_in * l + ci * l + il as usize]
                                * w[co * c_in * k + ci * k + kk];
                        }
                    }
                }
                want[ni * c_out * l_out + co * l_out + lo] = acc;
            }
        }
    }
    assert_close_slice(&got, &want, "conv1d_forward");
}

#[test]
fn test_conv2d_forward() {
    let mut s = TensorStore::new();
    let (n, c_in, h, w, c_out, kh, kw) = (2, 3, 6, 6, 4, 3, 3);
    let stride = 1usize;
    let padding = 1usize;
    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;
    let x_h: Vec<f32> = (0..n * c_in * h * w)
        .map(|i| (i as f32) * 0.01 - 0.5)
        .collect();
    let wh: Vec<f32> = (0..c_out * c_in * kh * kw)
        .map(|i| (i as f32) * 0.02 - 0.2)
        .collect();
    let ix = s.from_slice(&x_h, &[n, c_in, h, w]);
    let iw = s.from_slice(&wh, &[c_out, c_in, kh, kw]);
    let out = conv::conv2d_forward(&mut s, ix, iw, stride, padding);
    let got = s.to_host(out);
    assert_eq!(s.shape(out), &[n, c_out, h_out, w_out]);

    let mut want = vec![0.0f32; n * c_out * h_out * w_out];
    for ni in 0..n {
        for co in 0..c_out {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc = 0.0f32;
                    for ci in 0..c_in {
                        for khh in 0..kh {
                            for kww in 0..kw {
                                let ih = (ho * stride) as i32 + khh as i32 - padding as i32;
                                let iw_ = (wo * stride) as i32 + kww as i32 - padding as i32;
                                if ih >= 0 && ih < h as i32 && iw_ >= 0 && iw_ < w as i32 {
                                    let x_idx = ni * c_in * h * w
                                        + ci * h * w
                                        + (ih as usize) * w
                                        + iw_ as usize;
                                    let w_idx =
                                        co * c_in * kh * kw + ci * kh * kw + khh * kw + kww;
                                    acc += x_h[x_idx] * wh[w_idx];
                                }
                            }
                        }
                    }
                    want[ni * c_out * h_out * w_out + co * h_out * w_out + ho * w_out + wo] = acc;
                }
            }
        }
    }
    assert_close_slice(&got, &want, "conv2d_forward");
}

#[test]
fn test_flash_attention_forward() {
    let mut s = TensorStore::new();
    let (bh, sq, d) = (2usize, 16usize, 32usize);
    let scale = 1.0 / (d as f32).sqrt();
    let q_h: Vec<f32> = (0..bh * sq * d)
        .map(|i| ((i % 13) as f32) * 0.02 - 0.1)
        .collect();
    let k_h: Vec<f32> = (0..bh * sq * d)
        .map(|i| ((i % 11) as f32) * 0.025 - 0.1)
        .collect();
    let v_h: Vec<f32> = (0..bh * sq * d)
        .map(|i| ((i % 17) as f32) * 0.015 - 0.1)
        .collect();
    let iq = s.from_slice(&q_h, &[bh, sq, d]);
    let ik = s.from_slice(&k_h, &[bh, sq, d]);
    let iv = s.from_slice(&v_h, &[bh, sq, d]);

    let out_state = attention::flash_attention_forward(&mut s, iq, ik, iv, scale, false);
    let got = s.to_host(out_state.out);
    assert_eq!(s.shape(out_state.out), &[bh, sq, d]);

    // CPU reference: softmax(Q Kᵀ / √d) V, no causal mask.
    let mut want = vec![0.0f32; bh * sq * d];
    for b in 0..bh {
        for i in 0..sq {
            // Compute row scores = Q[i] · K / √d.
            let mut scores = vec![0.0f32; sq];
            for j in 0..sq {
                let mut acc = 0.0f32;
                for dd in 0..d {
                    acc += q_h[b * sq * d + i * d + dd] * k_h[b * sq * d + j * d + dd];
                }
                scores[j] = acc * scale;
            }
            // Softmax.
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                denom += *s;
            }
            for s in &mut scores {
                *s /= denom;
            }
            for j in 0..sq {
                for dd in 0..d {
                    want[b * sq * d + i * d + dd] += scores[j] * v_h[b * sq * d + j * d + dd];
                }
            }
        }
    }
    assert_close_slice(&got, &want, "flash_attention_forward");
}

// Conv1D backward tests use pow2 dims (k=2, c_out=4) because the cuTile
// `iota` on the backward_input kernel takes the raw `K`/`CO` const generics
// (no padding), and tile dims must be pow2.  The forward kernel has the
// padded-generics workaround so its test can use k=3.
#[test]
fn test_conv1d_backward_input() {
    let mut s = TensorStore::new();
    let (n, c_in, l_in, c_out, k) = (2, 4, 8, 4, 2);
    let stride = 1usize;
    let padding = 1usize;
    let l_out = (l_in + 2 * padding - k) / stride + 1;
    let dout_h: Vec<f32> = (0..n * c_out * l_out)
        .map(|i| (i as f32) * 0.02 - 0.3)
        .collect();
    let w_h: Vec<f32> = (0..c_out * c_in * k)
        .map(|i| (i as f32) * 0.03 - 0.2)
        .collect();
    let idout = s.from_slice(&dout_h, &[n, c_out, l_out]);
    let iw = s.from_slice(&w_h, &[c_out, c_in, k]);
    let dinp = conv::conv1d_backward_input(&mut s, idout, iw, l_in, stride, padding);
    let got = s.to_host(dinp);
    assert_eq!(s.shape(dinp), &[n, c_in, l_in]);

    // CPU reference: dinp[n, ci, il] = Σ_{co, kk} dout[n, co, ol] · w[co, ci, kk]
    // where ol_raw = il + pad - kk, valid iff ol_raw ≥ 0, ol_raw % stride == 0,
    // ol_raw / stride < l_out.
    let mut want = vec![0.0f32; n * c_in * l_in];
    for ni in 0..n {
        for ci in 0..c_in {
            for il in 0..l_in {
                let mut acc = 0.0f32;
                for co in 0..c_out {
                    for kk in 0..k {
                        let ol_raw = il as i32 + padding as i32 - kk as i32;
                        if ol_raw < 0 || ol_raw % stride as i32 != 0 {
                            continue;
                        }
                        let ol = (ol_raw / stride as i32) as usize;
                        if ol >= l_out {
                            continue;
                        }
                        acc += dout_h[ni * c_out * l_out + co * l_out + ol]
                            * w_h[co * c_in * k + ci * k + kk];
                    }
                }
                want[ni * c_in * l_in + ci * l_in + il] = acc;
            }
        }
    }
    assert_close_slice(&got, &want, "conv1d_backward_input");
}

#[test]
fn test_conv1d_backward_weight() {
    let mut s = TensorStore::new();
    // Weight backward kernel only iotas over `BL` (32, pow2), so k can be
    // any value — use k=3 to mirror the forward test shape.
    let (n, c_in, l_in, c_out, k) = (2, 3, 8, 4, 3);
    let stride = 1usize;
    let padding = 1usize;
    let l_out = (l_in + 2 * padding - k) / stride + 1;
    let dout_h: Vec<f32> = (0..n * c_out * l_out)
        .map(|i| (i as f32) * 0.02 - 0.3)
        .collect();
    let inp_h: Vec<f32> = (0..n * c_in * l_in)
        .map(|i| (i as f32) * 0.025 - 0.2)
        .collect();
    let idout = s.from_slice(&dout_h, &[n, c_out, l_out]);
    let iinp = s.from_slice(&inp_h, &[n, c_in, l_in]);
    let dw = conv::conv1d_backward_weight(&mut s, idout, iinp, k, stride, padding);
    let got = s.to_host(dw);
    assert_eq!(s.shape(dw), &[c_out, c_in, k]);

    // CPU reference: dw[co, ci, kk] = Σ_{n, ol} dout[n, co, ol] · inp[n, ci, il]
    // where il = ol*stride - pad + kk, valid iff il ∈ [0, l_in).
    let mut want = vec![0.0f32; c_out * c_in * k];
    for co in 0..c_out {
        for ci in 0..c_in {
            for kk in 0..k {
                let mut acc = 0.0f32;
                for ni in 0..n {
                    for ol in 0..l_out {
                        let il = ol as i32 * stride as i32 - padding as i32 + kk as i32;
                        if il < 0 || il >= l_in as i32 {
                            continue;
                        }
                        acc += dout_h[ni * c_out * l_out + co * l_out + ol]
                            * inp_h[ni * c_in * l_in + ci * l_in + il as usize];
                    }
                }
                want[co * c_in * k + ci * k + kk] = acc;
            }
        }
    }
    assert_close_slice(&got, &want, "conv1d_backward_weight");
}

#[test]
fn test_conv2d_backward_input() {
    let mut s = TensorStore::new();
    // Backward_input kernel takes raw KH/KW/CO as const generics for iotas,
    // so all three must be pow2.  Forward test uses kh=kw=3 because
    // conv2d_forward has the padded-generics workaround; the backward
    // kernels don't have that workaround yet.
    let (n, c_in, h_in, w_in, c_out, kh, kw) = (2, 4, 6, 6, 4, 2, 2);
    let stride = 1usize;
    let padding = 1usize;
    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;
    let dout_h: Vec<f32> = (0..n * c_out * h_out * w_out)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect();
    let w_h: Vec<f32> = (0..c_out * c_in * kh * kw)
        .map(|i| (i as f32) * 0.02 - 0.2)
        .collect();
    let idout = s.from_slice(&dout_h, &[n, c_out, h_out, w_out]);
    let iw = s.from_slice(&w_h, &[c_out, c_in, kh, kw]);
    let dinp = conv::conv2d_backward_input(&mut s, idout, iw, h_in, w_in, stride, padding);
    let got = s.to_host(dinp);
    assert_eq!(s.shape(dinp), &[n, c_in, h_in, w_in]);

    // CPU reference: dinp[n, ci, ih, iw] = Σ_{co, khh, kww} dout[n, co, oh, ow] · w[...]
    // where oh_raw = ih + pad - khh, ow_raw = iw + pad - kww (each valid if
    // ≥ 0, divisible by stride, and /stride < out_dim).
    let mut want = vec![0.0f32; n * c_in * h_in * w_in];
    for ni in 0..n {
        for ci in 0..c_in {
            for ih in 0..h_in {
                for iw_ in 0..w_in {
                    let mut acc = 0.0f32;
                    for co in 0..c_out {
                        for khh in 0..kh {
                            for kww in 0..kw {
                                let oh_raw = ih as i32 + padding as i32 - khh as i32;
                                let ow_raw = iw_ as i32 + padding as i32 - kww as i32;
                                if oh_raw < 0
                                    || ow_raw < 0
                                    || oh_raw % stride as i32 != 0
                                    || ow_raw % stride as i32 != 0
                                {
                                    continue;
                                }
                                let oh = (oh_raw / stride as i32) as usize;
                                let ow = (ow_raw / stride as i32) as usize;
                                if oh >= h_out || ow >= w_out {
                                    continue;
                                }
                                let d_idx = ni * c_out * h_out * w_out
                                    + co * h_out * w_out
                                    + oh * w_out
                                    + ow;
                                let w_idx =
                                    co * c_in * kh * kw + ci * kh * kw + khh * kw + kww;
                                acc += dout_h[d_idx] * w_h[w_idx];
                            }
                        }
                    }
                    want[ni * c_in * h_in * w_in + ci * h_in * w_in + ih * w_in + iw_] = acc;
                }
            }
        }
    }
    assert_close_slice(&got, &want, "conv2d_backward_input");
}

#[test]
fn test_conv2d_backward_weight() {
    let mut s = TensorStore::new();
    // Weight backward kernel only iotas over `BW` (32, pow2), so kh/kw can
    // be any value — use 3x3 to mirror the forward test shape.
    let (n, c_in, h_in, w_in, c_out, kh, kw) = (2, 3, 6, 6, 4, 3, 3);
    let stride = 1usize;
    let padding = 1usize;
    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;
    let dout_h: Vec<f32> = (0..n * c_out * h_out * w_out)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect();
    let inp_h: Vec<f32> = (0..n * c_in * h_in * w_in)
        .map(|i| (i as f32) * 0.015 - 0.2)
        .collect();
    let idout = s.from_slice(&dout_h, &[n, c_out, h_out, w_out]);
    let iinp = s.from_slice(&inp_h, &[n, c_in, h_in, w_in]);
    let dw = conv::conv2d_backward_weight(&mut s, idout, iinp, kh, kw, stride, padding);
    let got = s.to_host(dw);
    assert_eq!(s.shape(dw), &[c_out, c_in, kh, kw]);

    // CPU reference: dw[co, ci, khh, kww] = Σ_{n, oh, ow} dout[n, co, oh, ow]
    //   · inp[n, ci, oh*s - p + khh, ow*s - p + kww], with the input window
    // bounds-checked to [0, h_in)×[0, w_in).
    let mut want = vec![0.0f32; c_out * c_in * kh * kw];
    for co in 0..c_out {
        for ci in 0..c_in {
            for khh in 0..kh {
                for kww in 0..kw {
                    let mut acc = 0.0f32;
                    for ni in 0..n {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let ih = oh as i32 * stride as i32 - padding as i32
                                    + khh as i32;
                                let iw_ = ow as i32 * stride as i32 - padding as i32
                                    + kww as i32;
                                if ih < 0
                                    || ih >= h_in as i32
                                    || iw_ < 0
                                    || iw_ >= w_in as i32
                                {
                                    continue;
                                }
                                let d_idx = ni * c_out * h_out * w_out
                                    + co * h_out * w_out
                                    + oh * w_out
                                    + ow;
                                let i_idx = ni * c_in * h_in * w_in
                                    + ci * h_in * w_in
                                    + (ih as usize) * w_in
                                    + iw_ as usize;
                                acc += dout_h[d_idx] * inp_h[i_idx];
                            }
                        }
                    }
                    want[co * c_in * kh * kw + ci * kh * kw + khh * kw + kww] = acc;
                }
            }
        }
    }
    assert_close_slice(&got, &want, "conv2d_backward_weight");
}

#[test]
fn test_flash_attention_backward() {
    let mut s = TensorStore::new();
    let (bh, sq, d) = (2usize, 16usize, 32usize);
    let scale = 1.0 / (d as f32).sqrt();
    let q_h: Vec<f32> = (0..bh * sq * d)
        .map(|i| ((i % 13) as f32) * 0.02 - 0.1)
        .collect();
    let k_h: Vec<f32> = (0..bh * sq * d)
        .map(|i| ((i % 11) as f32) * 0.025 - 0.1)
        .collect();
    let v_h: Vec<f32> = (0..bh * sq * d)
        .map(|i| ((i % 17) as f32) * 0.015 - 0.1)
        .collect();
    let dout_h: Vec<f32> = (0..bh * sq * d)
        .map(|i| ((i % 7) as f32) * 0.01 - 0.03)
        .collect();
    let iq = s.from_slice(&q_h, &[bh, sq, d]);
    let ik = s.from_slice(&k_h, &[bh, sq, d]);
    let iv = s.from_slice(&v_h, &[bh, sq, d]);

    // Forward to get out + lse for the backward pass.
    let fwd = attention::flash_attention_forward(&mut s, iq, ik, iv, scale, false);

    let idout = s.from_slice(&dout_h, &[bh, sq, d]);
    let bw = attention::flash_attention_backward(
        &mut s, idout, iq, ik, iv, fwd.out, fwd.lse, scale, false,
    );
    let got_dq = s.to_host(bw.dq);
    let got_dk = s.to_host(bw.dk);
    let got_dv = s.to_host(bw.dv);

    // CPU reference for FlashAttention backward.  Standard derivation:
    //   P  = softmax(Q Kᵀ / √d)
    //   O  = P V
    //   D_i = Σ_d O_id · dO_id           (rowsum)
    //   dV = Pᵀ dO
    //   dP = dO Vᵀ
    //   dS = P · (dP - D_i)              (softmax backward)
    //   dQ = (dS K) · scale
    //   dK = (dSᵀ Q) · scale
    let mut want_dq = vec![0.0f32; bh * sq * d];
    let mut want_dk = vec![0.0f32; bh * sq * d];
    let mut want_dv = vec![0.0f32; bh * sq * d];

    for b in 0..bh {
        let qb = &q_h[b * sq * d..(b + 1) * sq * d];
        let kb = &k_h[b * sq * d..(b + 1) * sq * d];
        let vb = &v_h[b * sq * d..(b + 1) * sq * d];
        let dob = &dout_h[b * sq * d..(b + 1) * sq * d];

        // P[i, j] = softmax_j(Q[i] · K[j] · scale).
        let mut p = vec![0.0f32; sq * sq];
        for i in 0..sq {
            let mut scores = vec![0.0f32; sq];
            for j in 0..sq {
                let mut acc = 0.0f32;
                for dd in 0..d {
                    acc += qb[i * d + dd] * kb[j * d + dd];
                }
                scores[j] = acc * scale;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for sc in &mut scores {
                *sc = (*sc - m).exp();
                denom += *sc;
            }
            for j in 0..sq {
                p[i * sq + j] = scores[j] / denom;
            }
        }

        // O = P V, then D_i = Σ_d O_id · dO_id.
        let mut d_row = vec![0.0f32; sq];
        for i in 0..sq {
            for dd in 0..d {
                let mut o_id = 0.0f32;
                for j in 0..sq {
                    o_id += p[i * sq + j] * vb[j * d + dd];
                }
                d_row[i] += o_id * dob[i * d + dd];
            }
        }

        // dV[j, d] = Σ_i P[i, j] · dO[i, d].
        let dv_b = &mut want_dv[b * sq * d..(b + 1) * sq * d];
        for j in 0..sq {
            for dd in 0..d {
                let mut acc = 0.0f32;
                for i in 0..sq {
                    acc += p[i * sq + j] * dob[i * d + dd];
                }
                dv_b[j * d + dd] = acc;
            }
        }

        // dP[i, j] = Σ_d dO[i, d] · V[j, d];  dS[i, j] = P[i, j] · (dP[i, j] - D_i).
        let mut ds = vec![0.0f32; sq * sq];
        for i in 0..sq {
            for j in 0..sq {
                let mut dp_ij = 0.0f32;
                for dd in 0..d {
                    dp_ij += dob[i * d + dd] * vb[j * d + dd];
                }
                ds[i * sq + j] = p[i * sq + j] * (dp_ij - d_row[i]);
            }
        }

        // dQ[i, d] = Σ_j dS[i, j] · K[j, d] · scale.
        let dq_b = &mut want_dq[b * sq * d..(b + 1) * sq * d];
        for i in 0..sq {
            for dd in 0..d {
                let mut acc = 0.0f32;
                for j in 0..sq {
                    acc += ds[i * sq + j] * kb[j * d + dd];
                }
                dq_b[i * d + dd] = acc * scale;
            }
        }

        // dK[j, d] = Σ_i dS[i, j] · Q[i, d] · scale.
        let dk_b = &mut want_dk[b * sq * d..(b + 1) * sq * d];
        for j in 0..sq {
            for dd in 0..d {
                let mut acc = 0.0f32;
                for i in 0..sq {
                    acc += ds[i * sq + j] * qb[i * d + dd];
                }
                dk_b[j * d + dd] = acc * scale;
            }
        }
    }

    assert_close_slice(&got_dq, &want_dq, "flash_attn_bwd dq");
    assert_close_slice(&got_dk, &want_dk, "flash_attn_bwd dk");
    assert_close_slice(&got_dv, &want_dv, "flash_attn_bwd dv");
}
