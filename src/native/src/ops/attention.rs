use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore};

#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchConfig, PushKernelArg};

// =========================================================================
// Flash Attention forward
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn flash_attention(
    q: TensorId, k: TensorId, v: TensorId,
    scale: f32, causal: bool,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let q_shape = store.shape(q).to_vec();
    let bh = q_shape[0];
    let s = q_shape[1];
    let d = q_shape[2];

    let q_data = store.to_host(q);
    let k_data = store.to_host(k);
    let v_data = store.to_host(v);

    let mut out = vec![0.0f32; bh * s * d];
    let mut lse = vec![0.0f32; bh * s];

    for b in 0..bh {
        for row in 0..s {
            let col_end = if causal { row + 1 } else { s };
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            let mut acc = vec![0.0f32; d];

            for col in 0..col_end {
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q_data[b * s * d + row * d + dd] * k_data[b * s * d + col * d + dd];
                }
                dot *= scale;

                let old_max = running_max;
                running_max = running_max.max(dot);
                let exp_diff = (old_max - running_max).exp();

                running_sum = running_sum * exp_diff + (dot - running_max).exp();
                for dd in 0..d {
                    acc[dd] = acc[dd] * exp_diff + (dot - running_max).exp() * v_data[b * s * d + col * d + dd];
                }
            }

            let inv_sum = 1.0 / running_sum;
            for dd in 0..d {
                out[b * s * d + row * d + dd] = acc[dd] * inv_sum;
            }
            lse[b * s + row] = running_max + running_sum.ln();
        }
    }

    let out_id = store.from_vec(out, &[bh, s, d]);
    let lse_id = store.from_vec(lse, &[bh, s]);

    tape.record(TapeEntry {
        op: BackwardOp::FlashAttention,
        output_id: out_id,
        input_ids: smallvec![q, k, v],
        saved: SavedContext::FlashAttention {
            q, k, v, out: out_id, lse: lse_id,
            scale, s, d, causal,
        },
    });
    out_id
}

#[cfg(feature = "cuda")]
pub fn flash_attention(
    q: TensorId, k: TensorId, v: TensorId,
    scale: f32, causal: bool,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let q_shape = store.shape(q).to_vec();
    let bh = q_shape[0];
    let s = q_shape[1];
    let d = q_shape[2];

    let dev = GpuDevice::instance();

    let q_ptr = store.dev_ptr(q);
    let k_ptr = store.dev_ptr(k);
    let v_ptr = store.dev_ptr(v);

    let out_id = store.zeros(&[bh, s, d]);
    let out_ptr = store.dev_ptr(out_id);
    let lse_id = store.zeros(&[bh, s]);
    let lse_ptr = store.dev_ptr(lse_id);

    let causal_i = if causal { 1i32 } else { 0i32 };
    let block_size = 32u32;
    let grid = (bh as u32, (s as u32 + block_size - 1) / block_size, 1);
    let block = (d.min(32) as u32, block_size, 1);

    let func = dev.get_func("flash_attention_forward_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr)
            .arg(&lse_ptr)
            .arg(&q_ptr)
            .arg(&k_ptr)
            .arg(&v_ptr)
            .arg(&scale)
            .arg(&(s as i32))
            .arg(&(d as i32))
            .arg(&causal_i)
            .launch(LaunchConfig {
                grid_dim: grid,
                block_dim: block,
                shared_mem_bytes: 0,
            })
            .unwrap();
    }

    tape.record(TapeEntry {
        op: BackwardOp::FlashAttention,
        output_id: out_id,
        input_ids: smallvec![q, k, v],
        saved: SavedContext::FlashAttention {
            q, k, v, out: out_id, lse: lse_id,
            scale, s, d, causal,
        },
    });
    out_id
}

// =========================================================================
// Flash Attention backward
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn flash_attention_backward(
    grad: TensorId, saved: &SavedContext, store: &mut TensorStore,
) -> Vec<Option<TensorId>> {
    if let SavedContext::FlashAttention { q, k, v, out, lse, scale, s, d, causal } = saved {
        let bh = store.shape(*q)[0];
        let q_data = store.to_host(*q);
        let k_data = store.to_host(*k);
        let v_data = store.to_host(*v);
        let out_data = store.to_host(*out);
        let lse_data = store.to_host(*lse);
        let do_data = store.to_host(grad);

        let total = bh * s * d;
        let mut dq = vec![0.0f32; total];
        let mut dk = vec![0.0f32; total];
        let mut dv = vec![0.0f32; total];

        for b in 0..bh {
            for row in 0..*s {
                let col_end = if *causal { row + 1 } else { *s };
                let lse_val = lse_data[b * s + row];

                let mut di = 0.0f32;
                for dd in 0..*d {
                    di += do_data[b * s * d + row * d + dd] * out_data[b * s * d + row * d + dd];
                }

                for col in 0..col_end {
                    let mut dot = 0.0f32;
                    for dd in 0..*d {
                        dot += q_data[b * s * d + row * d + dd] * k_data[b * s * d + col * d + dd];
                    }
                    dot *= scale;
                    let p = (dot - lse_val).exp();

                    let mut dp = 0.0f32;
                    for dd in 0..*d {
                        dp += do_data[b * s * d + row * d + dd] * v_data[b * s * d + col * d + dd];
                        dv[b * s * d + col * d + dd] += p * do_data[b * s * d + row * d + dd];
                    }

                    let ds = p * (dp - di) * scale;
                    for dd in 0..*d {
                        dq[b * s * d + row * d + dd] += ds * k_data[b * s * d + col * d + dd];
                        dk[b * s * d + col * d + dd] += ds * q_data[b * s * d + row * d + dd];
                    }
                }
            }
        }

        let shape = vec![bh, *s, *d];
        vec![
            Some(store.from_vec(dq, &shape)),
            Some(store.from_vec(dk, &shape)),
            Some(store.from_vec(dv, &shape)),
        ]
    } else { vec![None, None, None] }
}

#[cfg(feature = "cuda")]
pub fn flash_attention_backward(
    grad: TensorId, saved: &SavedContext, store: &mut TensorStore,
) -> Vec<Option<TensorId>> {
    if let SavedContext::FlashAttention { q, k, v, out, lse, scale, s, d, causal } = saved {
        let bh = store.shape(*q)[0];
        let dev = GpuDevice::instance();

        let dq_id = store.zeros(&[bh, *s, *d]);
        let dk_id = store.zeros(&[bh, *s, *d]);
        let dv_id = store.zeros(&[bh, *s, *d]);

        let dq_ptr = store.dev_ptr(dq_id);
        let dk_ptr = store.dev_ptr(dk_id);
        let dv_ptr = store.dev_ptr(dv_id);
        let do_ptr = store.dev_ptr(grad);
        let q_ptr = store.dev_ptr(*q);
        let k_ptr = store.dev_ptr(*k);
        let v_ptr = store.dev_ptr(*v);
        let out_ptr = store.dev_ptr(*out);
        let lse_ptr = store.dev_ptr(*lse);

        let causal_i = if *causal { 1i32 } else { 0i32 };
        let block_size = 32u32;
        let grid = (bh as u32, (*s as u32 + block_size - 1) / block_size, 1);
        let block = ((*d).min(32) as u32, block_size, 1);

        let func = dev.get_func("flash_attention_backward_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&dq_ptr)
                .arg(&dk_ptr)
                .arg(&dv_ptr)
                .arg(&do_ptr)
                .arg(&q_ptr)
                .arg(&k_ptr)
                .arg(&v_ptr)
                .arg(&out_ptr)
                .arg(&lse_ptr)
                .arg(scale)
                .arg(&(*s as i32))
                .arg(&(*d as i32))
                .arg(&causal_i)
                .launch(LaunchConfig {
                    grid_dim: grid,
                    block_dim: block,
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }

        vec![Some(dq_id), Some(dk_id), Some(dv_id)]
    } else { vec![None, None, None] }
}
