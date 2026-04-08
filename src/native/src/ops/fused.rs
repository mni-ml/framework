use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore, shape_size};

#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchConfig, PushKernelArg};

#[cfg(feature = "cuda")]
fn launch_cfg(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

// =========================================================================
// Fused residual + LayerNorm: out = LN(x + residual)
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn residual_layernorm(
    x: TensorId, residual: TensorId,
    gamma: TensorId, beta: TensorId, eps: f32,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let x_shape = store.shape(x).to_vec();
    let c = *x_shape.last().unwrap();
    let n = shape_size(&x_shape) / c;

    let x_data = store.to_host(x);
    let r_data = store.to_host(residual);
    let gamma_data = store.to_host(gamma);
    let beta_data = store.to_host(beta);

    let mut res_data = vec![0.0f32; n * c];
    let mut out = vec![0.0f32; n * c];
    let mut mean_buf = vec![0.0f32; n];
    let mut rstd_buf = vec![0.0f32; n];

    for row in 0..n {
        let off = row * c;
        let mut sum = 0.0f32;
        for j in 0..c {
            let val = x_data[off + j] + r_data[off + j];
            res_data[off + j] = val;
            sum += val;
        }
        let m = sum / c as f32;
        mean_buf[row] = m;

        let mut var = 0.0f32;
        for j in 0..c {
            let d = res_data[off + j] - m;
            var += d * d;
        }
        let rstd = 1.0 / (var / c as f32 + eps).sqrt();
        rstd_buf[row] = rstd;

        for j in 0..c {
            out[off + j] = (res_data[off + j] - m) * rstd * gamma_data[j] + beta_data[j];
        }
    }

    let res_id = store.from_vec(res_data, &x_shape);
    let mean_id = store.from_vec(mean_buf, &[n]);
    let rstd_id = store.from_vec(rstd_buf, &[n]);
    let out_id = store.from_vec(out, &x_shape);

    tape.record(TapeEntry {
        op: BackwardOp::ResidualLayerNorm, output_id: out_id,
        input_ids: smallvec![x, residual, gamma, beta],
        saved: SavedContext::Tensors(smallvec![res_id, gamma, mean_id, rstd_id]),
    });
    out_id
}

#[cfg(feature = "cuda")]
pub fn residual_layernorm(
    x: TensorId, residual: TensorId,
    gamma: TensorId, beta: TensorId, eps: f32,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let x_shape = store.shape(x).to_vec();
    let c = *x_shape.last().unwrap();
    let n = shape_size(&x_shape) / c;

    let dev = GpuDevice::instance();
    let x_ptr = store.dev_ptr(x);
    let res_ptr_in = store.dev_ptr(residual);
    let gamma_ptr = store.dev_ptr(gamma);
    let beta_ptr = store.dev_ptr(beta);

    let out_id = store.zeros(&x_shape);
    let out_ptr = store.dev_ptr(out_id);
    let res_id = store.zeros(&x_shape);
    let res_ptr_out = store.dev_ptr(res_id);
    let mean_id = store.zeros(&[n]);
    let mean_ptr = store.dev_ptr(mean_id);
    let rstd_id = store.zeros(&[n]);
    let rstd_ptr = store.dev_ptr(rstd_id);

    let func = dev.get_func("residual_layernorm_forward_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr)
            .arg(&res_ptr_out)
            .arg(&mean_ptr)
            .arg(&rstd_ptr)
            .arg(&x_ptr)
            .arg(&res_ptr_in)
            .arg(&gamma_ptr)
            .arg(&beta_ptr)
            .arg(&(n as i32))
            .arg(&(c as i32))
            .arg(&eps)
            .launch(LaunchConfig {
                grid_dim: (n as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }

    tape.record(TapeEntry {
        op: BackwardOp::ResidualLayerNorm, output_id: out_id,
        input_ids: smallvec![x, residual, gamma, beta],
        saved: SavedContext::Tensors(smallvec![res_id, gamma, mean_id, rstd_id]),
    });
    out_id
}

// Backward reuses the standard layernorm backward, producing grads for
// x and residual (both receive the same dx), plus dgamma and dbeta.
#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn residual_layernorm_backward(
    grad: TensorId, saved: &SavedContext, store: &mut TensorStore,
) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensors(ids) = saved {
        let res_id = ids[0]; let gamma = ids[1]; let mean_id = ids[2]; let rstd_id = ids[3];
        let x_shape = store.shape(res_id).to_vec();
        let c = *x_shape.last().unwrap();
        let n = shape_size(&x_shape) / c;

        let res_data = store.to_host(res_id);
        let gamma_data = store.to_host(gamma);
        let mean_data = store.to_host(mean_id);
        let rstd_data = store.to_host(rstd_id);
        let grad_data = store.to_host(grad);

        let mut dx = vec![0.0f32; n * c];
        let mut dgamma = vec![0.0f32; c];
        let mut dbeta = vec![0.0f32; c];

        for row in 0..n {
            let off = row * c;
            let m = mean_data[row];
            let rstd = rstd_data[row];

            let mut sum_dg = 0.0f32;
            let mut sum_dgx = 0.0f32;
            for j in 0..c {
                let xhat = (res_data[off + j] - m) * rstd;
                let dg = grad_data[off + j] * gamma_data[j];
                sum_dg += dg;
                sum_dgx += dg * xhat;
                dgamma[j] += grad_data[off + j] * xhat;
                dbeta[j] += grad_data[off + j];
            }

            for j in 0..c {
                let xhat = (res_data[off + j] - m) * rstd;
                let dg = grad_data[off + j] * gamma_data[j];
                dx[off + j] = rstd * (dg - sum_dg / c as f32 - xhat * sum_dgx / c as f32);
            }
        }

        let gamma_shape = store.shape(gamma).to_vec();
        let dx_id = store.from_vec(dx, &x_shape);
        vec![
            Some(dx_id),             // grad for x
            Some(dx_id),             // grad for residual (same as dx)
            Some(store.from_vec(dgamma, &gamma_shape)),
            Some(store.from_vec(dbeta, &gamma_shape)),
        ]
    } else { vec![None, None, None, None] }
}

#[cfg(feature = "cuda")]
pub fn residual_layernorm_backward(
    grad: TensorId, saved: &SavedContext, store: &mut TensorStore,
) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensors(ids) = saved {
        let res_id = ids[0]; let gamma = ids[1]; let mean_id = ids[2]; let rstd_id = ids[3];
        let x_shape = store.shape(res_id).to_vec();
        let c = *x_shape.last().unwrap();
        let n = shape_size(&x_shape) / c;
        let gamma_shape = store.shape(gamma).to_vec();

        let dev = GpuDevice::instance();
        let grad_ptr = store.dev_ptr(grad);
        let res_ptr = store.dev_ptr(res_id);
        let mean_ptr = store.dev_ptr(mean_id);
        let rstd_ptr = store.dev_ptr(rstd_id);
        let gamma_ptr = store.dev_ptr(gamma);

        let dx_id = store.zeros(&x_shape);
        let dx_ptr = store.dev_ptr(dx_id);
        let dgamma_id = store.zeros(&gamma_shape);
        let dgamma_ptr = store.dev_ptr(dgamma_id);
        let dbeta_id = store.zeros(&gamma_shape);
        let dbeta_ptr = store.dev_ptr(dbeta_id);

        let func = dev.get_func("layernorm_backward_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&dx_ptr)
                .arg(&dgamma_ptr)
                .arg(&dbeta_ptr)
                .arg(&grad_ptr)
                .arg(&res_ptr)
                .arg(&mean_ptr)
                .arg(&rstd_ptr)
                .arg(&gamma_ptr)
                .arg(&(n as i32))
                .arg(&(c as i32))
                .launch(LaunchConfig {
                    grid_dim: (n as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }

        vec![
            Some(dx_id),
            Some(dx_id),
            Some(dgamma_id),
            Some(dbeta_id),
        ]
    } else { vec![None, None, None, None] }
}

// =========================================================================
// Fused bias + GELU: out = gelu(x + bias)
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn bias_gelu(
    x: TensorId, bias: TensorId,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let x_shape = store.shape(x).to_vec();
    let c = *x_shape.last().unwrap();
    let total = shape_size(&x_shape);

    let x_data = store.to_host(x);
    let bias_data = store.to_host(bias);

    let mut out = vec![0.0f32; total];
    for i in 0..total {
        let j = i % c;
        let val = x_data[i] + bias_data[j];
        let inner = 0.7978845608_f32 * (val + 0.044715_f32 * val * val * val);
        out[i] = 0.5 * val * (1.0 + inner.tanh());
    }

    let out_id = store.from_vec(out, &x_shape);
    tape.record(TapeEntry {
        op: BackwardOp::BiasGelu, output_id: out_id,
        input_ids: smallvec![x, bias],
        saved: SavedContext::Tensors(smallvec![x, bias]),
    });
    out_id
}

#[cfg(feature = "cuda")]
pub fn bias_gelu(
    x: TensorId, bias: TensorId,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let x_shape = store.shape(x).to_vec();
    let c = *x_shape.last().unwrap();
    let total = shape_size(&x_shape);
    let n = total / c;

    let dev = GpuDevice::instance();
    let x_ptr = store.dev_ptr(x);
    let bias_ptr = store.dev_ptr(bias);

    let out_id = store.zeros(&x_shape);
    let out_ptr = store.dev_ptr(out_id);

    let func = dev.get_func("bias_gelu_forward_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr)
            .arg(&x_ptr)
            .arg(&bias_ptr)
            .arg(&(n as i32))
            .arg(&(c as i32))
            .launch(launch_cfg(total as u32))
            .unwrap();
    }

    tape.record(TapeEntry {
        op: BackwardOp::BiasGelu, output_id: out_id,
        input_ids: smallvec![x, bias],
        saved: SavedContext::Tensors(smallvec![x, bias]),
    });
    out_id
}

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn bias_gelu_backward(
    grad: TensorId, saved: &SavedContext, store: &mut TensorStore,
) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensors(ids) = saved {
        let x_id = ids[0]; let bias_id = ids[1];
        let x_shape = store.shape(x_id).to_vec();
        let c = *x_shape.last().unwrap();
        let total = shape_size(&x_shape);
        let bias_shape = store.shape(bias_id).to_vec();

        let x_data = store.to_host(x_id);
        let bias_data = store.to_host(bias_id);
        let grad_data = store.to_host(grad);

        let mut dx = vec![0.0f32; total];
        let mut dbias = vec![0.0f32; c];

        for i in 0..total {
            let j = i % c;
            let val = x_data[i] + bias_data[j];
            let inner = 0.7978845608_f32 * (val + 0.044715_f32 * val * val * val);
            let th = inner.tanh();
            let sech2 = 1.0 - th * th;
            let d_inner = 0.7978845608_f32 * (1.0 + 3.0 * 0.044715_f32 * val * val);
            let dgelu = 0.5 * (1.0 + th) + 0.5 * val * sech2 * d_inner;
            let g = grad_data[i] * dgelu;
            dx[i] = g;
            dbias[j] += g;
        }

        vec![
            Some(store.from_vec(dx, &x_shape)),
            Some(store.from_vec(dbias, &bias_shape)),
        ]
    } else { vec![None, None] }
}

#[cfg(feature = "cuda")]
pub fn bias_gelu_backward(
    grad: TensorId, saved: &SavedContext, store: &mut TensorStore,
) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensors(ids) = saved {
        let x_id = ids[0]; let bias_id = ids[1];
        let x_shape = store.shape(x_id).to_vec();
        let c = *x_shape.last().unwrap();
        let total = shape_size(&x_shape);
        let n = total / c;
        let bias_shape = store.shape(bias_id).to_vec();

        let dev = GpuDevice::instance();
        let grad_ptr = store.dev_ptr(grad);
        let x_ptr = store.dev_ptr(x_id);
        let bias_ptr = store.dev_ptr(bias_id);

        let dx_id = store.zeros(&x_shape);
        let dx_ptr = store.dev_ptr(dx_id);
        let dbias_id = store.zeros(&bias_shape);
        let dbias_ptr = store.dev_ptr(dbias_id);

        let func = dev.get_func("bias_gelu_backward_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&dx_ptr)
                .arg(&dbias_ptr)
                .arg(&grad_ptr)
                .arg(&x_ptr)
                .arg(&bias_ptr)
                .arg(&(n as i32))
                .arg(&(c as i32))
                .launch(launch_cfg(total as u32))
                .unwrap();
        }

        vec![Some(dx_id), Some(dbias_id)]
    } else { vec![None, None] }
}
