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
// Conv1d: input [N,C_in,L], weight [C_out,C_in,K] -> [N,C_out,L_out]
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn conv1d_forward(
    input: TensorId, weight: TensorId,
    stride: usize, padding: usize,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let inp_shape = store.shape(input).to_vec();
    let w_shape = store.shape(weight).to_vec();
    let (n, c_in, l) = (inp_shape[0], inp_shape[1], inp_shape[2]);
    let (c_out, _, k) = (w_shape[0], w_shape[1], w_shape[2]);
    let l_out = (l + 2 * padding - k) / stride + 1;

    let inp_data = store.to_host(input);
    let w_data = store.to_host(weight);
    let mut out_data = vec![0.0f32; n * c_out * l_out];

    for ni in 0..n {
        for co in 0..c_out {
            for ol in 0..l_out {
                let mut sum = 0.0f32;
                for ci in 0..c_in {
                    for ki in 0..k {
                        let il = (ol * stride) as isize - padding as isize + ki as isize;
                        if il >= 0 && (il as usize) < l {
                            sum += inp_data[ni * c_in * l + ci * l + il as usize]
                                 * w_data[co * c_in * k + ci * k + ki];
                        }
                    }
                }
                out_data[ni * c_out * l_out + co * l_out + ol] = sum;
            }
        }
    }

    let out = store.from_vec(out_data, &[n, c_out, l_out]);
    tape.record(TapeEntry {
        op: BackwardOp::Conv1d, output_id: out,
        input_ids: smallvec![input, weight],
        saved: SavedContext::TensorsAndShape(smallvec![input, weight], vec![stride, padding]),
    });
    out
}

#[cfg(feature = "cuda")]
pub fn conv1d_forward(
    input: TensorId, weight: TensorId,
    stride: usize, padding: usize,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let inp_shape = store.shape(input).to_vec();
    let w_shape = store.shape(weight).to_vec();
    let (n, c_in, l) = (inp_shape[0], inp_shape[1], inp_shape[2]);
    let (c_out, _, k) = (w_shape[0], w_shape[1], w_shape[2]);
    let l_out = (l + 2 * padding - k) / stride + 1;
    let total = n * c_out * l_out;

    let out = store.zeros(&[n, c_out, l_out]);
    let dev = GpuDevice::instance();
    unsafe {
        dev.stream.launch_builder(dev.get_func("conv1d_forward_f32"))
            .arg(&store.dev_ptr(out)).arg(&store.dev_ptr(input)).arg(&store.dev_ptr(weight))
            .arg(&(n as i32)).arg(&(c_in as i32)).arg(&(l as i32))
            .arg(&(c_out as i32)).arg(&(k as i32))
            .arg(&(stride as i32)).arg(&(padding as i32)).arg(&(l_out as i32))
            .launch(launch_cfg(total as u32)).unwrap();
    }
    tape.record(TapeEntry {
        op: BackwardOp::Conv1d, output_id: out,
        input_ids: smallvec![input, weight],
        saved: SavedContext::TensorsAndShape(smallvec![input, weight], vec![stride, padding]),
    });
    out
}

// =========================================================================
// Conv1d backward
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn conv1d_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorsAndShape(ids, params) = saved {
        let input = ids[0]; let weight = ids[1];
        let stride = params[0]; let padding = params[1];
        let inp_shape = store.shape(input).to_vec();
        let w_shape = store.shape(weight).to_vec();
        let grad_shape = store.shape(grad).to_vec();
        let (n, c_in, l) = (inp_shape[0], inp_shape[1], inp_shape[2]);
        let (c_out, _, k) = (w_shape[0], w_shape[1], w_shape[2]);
        let l_out = grad_shape[2];

        let inp_data = store.to_host(input);
        let w_data = store.to_host(weight);
        let grad_data = store.to_host(grad);

        let mut dinp = vec![0.0f32; n * c_in * l];
        let mut dw = vec![0.0f32; c_out * c_in * k];

        for ni in 0..n {
            for co in 0..c_out {
                for ol in 0..l_out {
                    let g = grad_data[ni * c_out * l_out + co * l_out + ol];
                    for ci in 0..c_in {
                        for ki in 0..k {
                            let il = (ol * stride) as isize - padding as isize + ki as isize;
                            if il >= 0 && (il as usize) < l {
                                dinp[ni * c_in * l + ci * l + il as usize] += g * w_data[co * c_in * k + ci * k + ki];
                                dw[co * c_in * k + ci * k + ki] += g * inp_data[ni * c_in * l + ci * l + il as usize];
                            }
                        }
                    }
                }
            }
        }

        vec![
            Some(store.from_vec(dinp, &inp_shape)),
            Some(store.from_vec(dw, &w_shape)),
        ]
    } else { vec![None, None] }
}

#[cfg(feature = "cuda")]
pub fn conv1d_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorsAndShape(ids, params) = saved {
        let input = ids[0]; let weight = ids[1];
        let stride = params[0]; let padding = params[1];
        let inp_shape = store.shape(input).to_vec();
        let w_shape = store.shape(weight).to_vec();
        let grad_shape = store.shape(grad).to_vec();
        let (n, c_in, l) = (inp_shape[0], inp_shape[1], inp_shape[2]);
        let (c_out, _, k) = (w_shape[0], w_shape[1], w_shape[2]);
        let l_out = grad_shape[2];

        let dinp = store.zeros(&inp_shape);
        let dw = store.zeros(&w_shape);
        let dev = GpuDevice::instance();
        unsafe {
            dev.stream.launch_builder(dev.get_func("conv1d_backward_input_f32"))
                .arg(&store.dev_ptr(dinp)).arg(&store.dev_ptr(grad)).arg(&store.dev_ptr(weight))
                .arg(&(n as i32)).arg(&(c_in as i32)).arg(&(l as i32))
                .arg(&(c_out as i32)).arg(&(k as i32))
                .arg(&(stride as i32)).arg(&(padding as i32)).arg(&(l_out as i32))
                .launch(launch_cfg(shape_size(&inp_shape) as u32)).unwrap();
            dev.stream.launch_builder(dev.get_func("conv1d_backward_weight_f32"))
                .arg(&store.dev_ptr(dw)).arg(&store.dev_ptr(grad)).arg(&store.dev_ptr(input))
                .arg(&(n as i32)).arg(&(c_in as i32)).arg(&(l as i32))
                .arg(&(c_out as i32)).arg(&(k as i32))
                .arg(&(stride as i32)).arg(&(padding as i32)).arg(&(l_out as i32))
                .launch(launch_cfg(shape_size(&w_shape) as u32)).unwrap();
        }
        vec![Some(dinp), Some(dw)]
    } else { vec![None, None] }
}

// =========================================================================
// Conv2d: input [N,C_in,H,W], weight [C_out,C_in,kH,kW] -> [N,C_out,H_out,W_out]
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn conv2d_forward(
    input: TensorId, weight: TensorId,
    stride: usize, padding: usize,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let inp_shape = store.shape(input).to_vec();
    let w_shape = store.shape(weight).to_vec();
    let (n, c_in, h, w) = (inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]);
    let (c_out, _, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;

    let inp_data = store.to_host(input);
    let w_data = store.to_host(weight);
    let mut out_data = vec![0.0f32; n * c_out * h_out * w_out];

    for ni in 0..n {
        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0f32;
                    for ci in 0..c_in {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let ih = (oh * stride) as isize - padding as isize + ky as isize;
                                let iw = (ow * stride) as isize - padding as isize + kx as isize;
                                if ih >= 0 && (ih as usize) < h && iw >= 0 && (iw as usize) < w {
                                    sum += inp_data[ni * c_in * h * w + ci * h * w + ih as usize * w + iw as usize]
                                         * w_data[co * c_in * kh * kw + ci * kh * kw + ky * kw + kx];
                                }
                            }
                        }
                    }
                    out_data[ni * c_out * h_out * w_out + co * h_out * w_out + oh * w_out + ow] = sum;
                }
            }
        }
    }

    let out = store.from_vec(out_data, &[n, c_out, h_out, w_out]);
    tape.record(TapeEntry {
        op: BackwardOp::Conv2d, output_id: out,
        input_ids: smallvec![input, weight],
        saved: SavedContext::TensorsAndShape(smallvec![input, weight], vec![stride, padding]),
    });
    out
}

#[cfg(feature = "cuda")]
pub fn conv2d_forward(
    input: TensorId, weight: TensorId,
    stride: usize, padding: usize,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let inp_shape = store.shape(input).to_vec();
    let w_shape = store.shape(weight).to_vec();
    let (n, c_in, h, w) = (inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]);
    let (c_out, _, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;
    let total = n * c_out * h_out * w_out;

    let out = store.zeros(&[n, c_out, h_out, w_out]);
    let dev = GpuDevice::instance();
    unsafe {
        dev.stream.launch_builder(dev.get_func("conv2d_forward_f32"))
            .arg(&store.dev_ptr(out)).arg(&store.dev_ptr(input)).arg(&store.dev_ptr(weight))
            .arg(&(n as i32)).arg(&(c_in as i32)).arg(&(h as i32)).arg(&(w as i32))
            .arg(&(c_out as i32)).arg(&(kh as i32)).arg(&(kw as i32))
            .arg(&(stride as i32)).arg(&(padding as i32)).arg(&(h_out as i32)).arg(&(w_out as i32))
            .launch(launch_cfg(total as u32)).unwrap();
    }
    tape.record(TapeEntry {
        op: BackwardOp::Conv2d, output_id: out,
        input_ids: smallvec![input, weight],
        saved: SavedContext::TensorsAndShape(smallvec![input, weight], vec![stride, padding]),
    });
    out
}

// =========================================================================
// Conv2d backward
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn conv2d_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorsAndShape(ids, params) = saved {
        let input = ids[0]; let weight = ids[1];
        let stride = params[0]; let padding = params[1];
        let inp_shape = store.shape(input).to_vec();
        let w_shape = store.shape(weight).to_vec();
        let grad_shape = store.shape(grad).to_vec();
        let (n, c_in, h, w) = (inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]);
        let (c_out, _, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
        let (h_out, w_out) = (grad_shape[2], grad_shape[3]);

        let inp_data = store.to_host(input);
        let w_data = store.to_host(weight);
        let grad_data = store.to_host(grad);

        let mut dinp = vec![0.0f32; n * c_in * h * w];
        let mut dw = vec![0.0f32; c_out * c_in * kh * kw];

        for ni in 0..n {
            for co in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let g = grad_data[ni * c_out * h_out * w_out + co * h_out * w_out + oh * w_out + ow];
                        for ci in 0..c_in {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let ih = (oh * stride) as isize - padding as isize + ky as isize;
                                    let iw_idx = (ow * stride) as isize - padding as isize + kx as isize;
                                    if ih >= 0 && (ih as usize) < h && iw_idx >= 0 && (iw_idx as usize) < w {
                                        dinp[ni * c_in * h * w + ci * h * w + ih as usize * w + iw_idx as usize] +=
                                            g * w_data[co * c_in * kh * kw + ci * kh * kw + ky * kw + kx];
                                        dw[co * c_in * kh * kw + ci * kh * kw + ky * kw + kx] +=
                                            g * inp_data[ni * c_in * h * w + ci * h * w + ih as usize * w + iw_idx as usize];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        vec![
            Some(store.from_vec(dinp, &inp_shape)),
            Some(store.from_vec(dw, &w_shape)),
        ]
    } else { vec![None, None] }
}

#[cfg(feature = "cuda")]
pub fn conv2d_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorsAndShape(ids, params) = saved {
        let input = ids[0]; let weight = ids[1];
        let stride = params[0]; let padding = params[1];
        let inp_shape = store.shape(input).to_vec();
        let w_shape = store.shape(weight).to_vec();
        let grad_shape = store.shape(grad).to_vec();
        let (n, c_in, h, w) = (inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]);
        let (c_out, _, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
        let (h_out, w_out) = (grad_shape[2], grad_shape[3]);

        let dinp = store.zeros(&inp_shape);
        let dw = store.zeros(&w_shape);
        let dev = GpuDevice::instance();
        unsafe {
            dev.stream.launch_builder(dev.get_func("conv2d_backward_input_f32"))
                .arg(&store.dev_ptr(dinp)).arg(&store.dev_ptr(grad)).arg(&store.dev_ptr(weight))
                .arg(&(n as i32)).arg(&(c_in as i32)).arg(&(h as i32)).arg(&(w as i32))
                .arg(&(c_out as i32)).arg(&(kh as i32)).arg(&(kw as i32))
                .arg(&(stride as i32)).arg(&(padding as i32)).arg(&(h_out as i32)).arg(&(w_out as i32))
                .launch(launch_cfg(shape_size(&inp_shape) as u32)).unwrap();
            dev.stream.launch_builder(dev.get_func("conv2d_backward_weight_f32"))
                .arg(&store.dev_ptr(dw)).arg(&store.dev_ptr(grad)).arg(&store.dev_ptr(input))
                .arg(&(n as i32)).arg(&(c_in as i32)).arg(&(h as i32)).arg(&(w as i32))
                .arg(&(c_out as i32)).arg(&(kh as i32)).arg(&(kw as i32))
                .arg(&(stride as i32)).arg(&(padding as i32)).arg(&(h_out as i32)).arg(&(w_out as i32))
                .launch(launch_cfg(shape_size(&w_shape) as u32)).unwrap();
        }
        vec![Some(dinp), Some(dw)]
    } else { vec![None, None] }
}
