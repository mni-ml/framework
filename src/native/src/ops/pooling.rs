use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore, shape_size};

#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchConfig, PushKernelArg};

#[cfg(feature = "cuda")]
fn launch_cfg(n: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

// =========================================================================
// AvgPool2d: input [N,C,H,W] -> [N,C,H_out,W_out]
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn avgpool2d_forward(
    input: TensorId, kh: usize, kw: usize,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let shape = store.shape(input).to_vec();
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let h_out = h / kh;
    let w_out = w / kw;
    let data = store.to_host(input);
    let mut out_data = vec![0.0f32; n * c * h_out * w_out];

    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0f32;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let ih = oh * kh + ky;
                            let iw = ow * kw + kx;
                            if ih < h && iw < w {
                                sum += data[ni * c * h * w + ci * h * w + ih * w + iw];
                            }
                        }
                    }
                    out_data[ni * c * h_out * w_out + ci * h_out * w_out + oh * w_out + ow] = sum / (kh * kw) as f32;
                }
            }
        }
    }

    let out = store.from_vec(out_data, &[n, c, h_out, w_out]);
    tape.record(TapeEntry {
        op: BackwardOp::AvgPool2d, output_id: out,
        input_ids: smallvec![input],
        saved: SavedContext::TensorAndShape(input, vec![kh, kw]),
    });
    out
}

#[cfg(feature = "cuda")]
pub fn avgpool2d_forward(
    input: TensorId, kh: usize, kw: usize,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let shape = store.shape(input).to_vec();
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let h_out = h / kh;
    let w_out = w / kw;
    let total = n * c * h_out * w_out;

    let out = store.zeros(&[n, c, h_out, w_out]);
    let dev = GpuDevice::instance();
    unsafe {
        dev.stream.launch_builder(dev.get_func("avgpool2d_forward_f32"))
            .arg(&store.dev_ptr(out)).arg(&store.dev_ptr(input))
            .arg(&(n as i32)).arg(&(c as i32)).arg(&(h as i32)).arg(&(w as i32))
            .arg(&(kh as i32)).arg(&(kw as i32)).arg(&(h_out as i32)).arg(&(w_out as i32))
            .launch(launch_cfg(total as u32)).unwrap();
    }
    tape.record(TapeEntry {
        op: BackwardOp::AvgPool2d, output_id: out,
        input_ids: smallvec![input],
        saved: SavedContext::TensorAndShape(input, vec![kh, kw]),
    });
    out
}

// =========================================================================
// AvgPool2d backward
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn avgpool2d_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorAndShape(inp, ks) = saved {
        let inp_shape = store.shape(*inp).to_vec();
        let (n, c, h, w) = (inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]);
        let (kh, kw) = (ks[0], ks[1]);
        let h_out = h / kh;
        let w_out = w / kw;
        let grad_data = store.to_host(grad);
        let inv = 1.0 / (kh * kw) as f32;
        let mut dinp = vec![0.0f32; n * c * h * w];
        for ni in 0..n {
            for ci in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let g = grad_data[ni * c * h_out * w_out + ci * h_out * w_out + oh * w_out + ow] * inv;
                        for ky in 0..kh {
                            for kx in 0..kw {
                                dinp[ni * c * h * w + ci * h * w + (oh * kh + ky) * w + ow * kw + kx] = g;
                            }
                        }
                    }
                }
            }
        }
        vec![Some(store.from_vec(dinp, &inp_shape))]
    } else { vec![None] }
}

#[cfg(feature = "cuda")]
pub fn avgpool2d_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorAndShape(inp, ks) = saved {
        let inp_shape = store.shape(*inp).to_vec();
        let (n, c, h, w) = (inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]);
        let (kh, kw) = (ks[0], ks[1]);
        let h_out = h / kh;
        let w_out = w / kw;
        let total = shape_size(&inp_shape);

        let dinp = store.zeros(&inp_shape);
        let dev = GpuDevice::instance();
        unsafe {
            dev.stream.launch_builder(dev.get_func("avgpool2d_backward_f32"))
                .arg(&store.dev_ptr(dinp)).arg(&store.dev_ptr(grad))
                .arg(&(n as i32)).arg(&(c as i32)).arg(&(h as i32)).arg(&(w as i32))
                .arg(&(kh as i32)).arg(&(kw as i32)).arg(&(h_out as i32)).arg(&(w_out as i32))
                .launch(launch_cfg(total as u32)).unwrap();
        }
        vec![Some(dinp)]
    } else { vec![None] }
}

// =========================================================================
// MaxPool2d: input [N,C,H,W] -> [N,C,H_out,W_out]
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn maxpool2d_forward(
    input: TensorId, kh: usize, kw: usize,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let shape = store.shape(input).to_vec();
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let h_out = h / kh;
    let w_out = w / kw;
    let data = store.to_host(input);
    let out_size = n * c * h_out * w_out;
    let mut out_data = vec![0.0f32; out_size];
    let mut argmax = vec![0usize; out_size];

    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let ih = oh * kh + ky;
                            let iw = ow * kw + kx;
                            if ih < h && iw < w {
                                let pos = ni * c * h * w + ci * h * w + ih * w + iw;
                                if data[pos] > max_val {
                                    max_val = data[pos];
                                    max_idx = pos;
                                }
                            }
                        }
                    }
                    let oidx = ni * c * h_out * w_out + ci * h_out * w_out + oh * w_out + ow;
                    out_data[oidx] = max_val;
                    argmax[oidx] = max_idx;
                }
            }
        }
    }

    let argmax_f32: Vec<f32> = argmax.iter().map(|&x| x as f32).collect();
    let argmax_id = store.from_vec(argmax_f32, &[out_size]);
    let out = store.from_vec(out_data, &[n, c, h_out, w_out]);
    tape.record(TapeEntry {
        op: BackwardOp::MaxPool2d, output_id: out,
        input_ids: smallvec![input],
        saved: SavedContext::TensorAndShape(argmax_id, shape),
    });
    out
}

#[cfg(feature = "cuda")]
pub fn maxpool2d_forward(
    input: TensorId, kh: usize, kw: usize,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let shape = store.shape(input).to_vec();
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let h_out = h / kh;
    let w_out = w / kw;
    let out_size = n * c * h_out * w_out;

    let out = store.zeros(&[n, c, h_out, w_out]);
    let argmax_f = store.zeros(&[out_size]);
    let dev = GpuDevice::instance();

    let argmax_i32: cudarc::driver::CudaSlice<i32> = dev.stream.alloc_zeros(out_size).unwrap();
    let argmax_ptr = dev.ptr(&argmax_i32);

    unsafe {
        dev.stream.launch_builder(dev.get_func("maxpool2d_forward_f32"))
            .arg(&store.dev_ptr(out)).arg(&argmax_ptr).arg(&store.dev_ptr(input))
            .arg(&(n as i32)).arg(&(c as i32)).arg(&(h as i32)).arg(&(w as i32))
            .arg(&(kh as i32)).arg(&(kw as i32)).arg(&(h_out as i32)).arg(&(w_out as i32))
            .launch(launch_cfg(out_size as u32)).unwrap();
    }

    let argmax_host: Vec<i32> = dev.stream.memcpy_dtov(&argmax_i32).unwrap();
    let argmax_f32: Vec<f32> = argmax_host.iter().map(|&x| x as f32).collect();
    let argmax_id = store.from_vec(argmax_f32, &[out_size]);

    tape.record(TapeEntry {
        op: BackwardOp::MaxPool2d, output_id: out,
        input_ids: smallvec![input],
        saved: SavedContext::TensorAndShape(argmax_id, shape),
    });
    out
}

// =========================================================================
// MaxPool2d backward
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn maxpool2d_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorAndShape(argmax_id, inp_shape) = saved {
        let argmax_data = store.to_host(*argmax_id);
        let grad_data = store.to_host(grad);
        let inp_size = shape_size(inp_shape);
        let mut dinp = vec![0.0f32; inp_size];
        for (i, &g) in grad_data.iter().enumerate() {
            let idx = argmax_data[i] as usize;
            if idx < inp_size { dinp[idx] += g; }
        }
        vec![Some(store.from_vec(dinp, inp_shape))]
    } else { vec![None] }
}

#[cfg(feature = "cuda")]
pub fn maxpool2d_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorAndShape(argmax_id, inp_shape) = saved {
        let argmax_data = store.to_host(*argmax_id);
        let grad_data = store.to_host(grad);
        let inp_size = shape_size(inp_shape);
        let mut dinp = vec![0.0f32; inp_size];
        for (i, &g) in grad_data.iter().enumerate() {
            let idx = argmax_data[i] as usize;
            if idx < inp_size { dinp[idx] += g; }
        }
        vec![Some(store.from_vec(dinp, inp_shape))]
    } else { vec![None] }
}

// =========================================================================
// tile: repeat tensor along dimensions
// =========================================================================

pub fn tile(input: TensorId, reps: &[usize], store: &mut TensorStore, _tape: &mut Tape) -> TensorId {
    let shape = store.shape(input).to_vec();
    let data = store.to_host(input);
    let ndim = shape.len();
    assert_eq!(reps.len(), ndim, "tile: reps must match tensor dimensions");

    let out_shape: Vec<usize> = shape.iter().zip(reps).map(|(&s, &r)| s * r).collect();
    let out_size = shape_size(&out_shape);
    let mut out_data = vec![0.0f32; out_size];

    let in_strides: Vec<usize> = {
        let mut s = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() { s[i] = s[i + 1] * shape[i + 1]; }
        s
    };
    let out_strides: Vec<usize> = {
        let mut s = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() { s[i] = s[i + 1] * out_shape[i + 1]; }
        s
    };

    for i in 0..out_size {
        let mut src_idx = 0;
        let mut rem = i;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            src_idx += (coord % shape[d]) * in_strides[d];
        }
        out_data[i] = data[src_idx];
    }

    store.from_vec(out_data, &out_shape)
}
