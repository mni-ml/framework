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

#[cfg(any(feature = "cpu", feature = "webgpu"))]
fn gelu_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

#[cfg(any(feature = "cpu", feature = "webgpu"))]
fn gelu_grad_scalar(x: f32) -> f32 {
    let s = (2.0f32 / std::f32::consts::PI).sqrt();
    let inner = s * (x + 0.044715 * x * x * x);
    let tanh_inner = inner.tanh();
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let d_inner = s * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
}

// ---------------------------------------------------------------------------
// GELU forward
// ---------------------------------------------------------------------------

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn gelu_forward(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let data: Vec<f32> = store.to_host(a).iter().map(|&x| gelu_scalar(x)).collect();
    let shape = store.shape(a).to_vec();
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Gelu, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Tensor(a),
    });
    out
}

#[cfg(feature = "cuda")]
pub fn gelu_forward(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let shape = store.shape(a).to_vec();
    let n = shape_size(&shape);
    let out = store.zeros(&shape);
    let dev = GpuDevice::instance();
    let out_ptr = store.dev_ptr(out);
    let a_ptr = store.dev_ptr(a);
    let func = dev.get_func("gelu_forward_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr)
            .arg(&a_ptr)
            .arg(&(n as i32))
            .launch(launch_cfg(n as u32))
            .unwrap();
    }
    tape.record(TapeEntry {
        op: BackwardOp::Gelu, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Tensor(a),
    });
    out
}

// ---------------------------------------------------------------------------
// GELU backward
// ---------------------------------------------------------------------------

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn gelu_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensor(inp) = saved {
        let inp_data = store.to_host(*inp);
        let grad_data = store.to_host(grad);
        let data: Vec<f32> = grad_data.iter().zip(&inp_data)
            .map(|(g, &x)| g * gelu_grad_scalar(x)).collect();
        let shape = store.shape(grad).to_vec();
        vec![Some(store.from_vec(data, &shape))]
    } else { vec![None] }
}

#[cfg(feature = "cuda")]
pub fn gelu_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensor(inp) = saved {
        let shape = store.shape(grad).to_vec();
        let n = shape_size(&shape);
        let out = store.zeros(&shape);
        let dev = GpuDevice::instance();
        let out_ptr = store.dev_ptr(out);
        let grad_ptr = store.dev_ptr(grad);
        let inp_ptr = store.dev_ptr(*inp);
        let func = dev.get_func("gelu_backward_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&out_ptr)
                .arg(&grad_ptr)
                .arg(&inp_ptr)
                .arg(&(n as i32))
                .launch(launch_cfg(n as u32))
                .unwrap();
        }
        vec![Some(out)]
    } else { vec![None] }
}

// ---------------------------------------------------------------------------
// ReLU forward
// ---------------------------------------------------------------------------

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn relu_forward(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let data: Vec<f32> = store.to_host(a).iter().map(|&x| x.max(0.0)).collect();
    let shape = store.shape(a).to_vec();
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Relu, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Tensor(a),
    });
    out
}

#[cfg(feature = "cuda")]
pub fn relu_forward(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let shape = store.shape(a).to_vec();
    let n = shape_size(&shape);
    let out = store.zeros(&shape);
    let dev = GpuDevice::instance();
    let out_ptr = store.dev_ptr(out);
    let a_ptr = store.dev_ptr(a);
    let func = dev.get_func("relu_forward_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr)
            .arg(&a_ptr)
            .arg(&(n as i32))
            .launch(launch_cfg(n as u32))
            .unwrap();
    }
    tape.record(TapeEntry {
        op: BackwardOp::Relu, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Tensor(a),
    });
    out
}

// ---------------------------------------------------------------------------
// ReLU backward
// ---------------------------------------------------------------------------

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn relu_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensor(inp) = saved {
        let inp_data = store.to_host(*inp);
        let grad_data = store.to_host(grad);
        let data: Vec<f32> = grad_data.iter().zip(&inp_data)
            .map(|(g, &x)| if x > 0.0 { *g } else { 0.0 }).collect();
        let shape = store.shape(grad).to_vec();
        vec![Some(store.from_vec(data, &shape))]
    } else { vec![None] }
}

#[cfg(feature = "cuda")]
pub fn relu_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensor(inp) = saved {
        let shape = store.shape(grad).to_vec();
        let n = shape_size(&shape);
        let out = store.zeros(&shape);
        let dev = GpuDevice::instance();
        let out_ptr = store.dev_ptr(out);
        let grad_ptr = store.dev_ptr(grad);
        let inp_ptr = store.dev_ptr(*inp);
        let func = dev.get_func("relu_backward_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&out_ptr)
                .arg(&grad_ptr)
                .arg(&inp_ptr)
                .arg(&(n as i32))
                .launch(launch_cfg(n as u32))
                .unwrap();
        }
        vec![Some(out)]
    } else { vec![None] }
}

// ---------------------------------------------------------------------------
// Sigmoid forward
// ---------------------------------------------------------------------------

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn sigmoid_forward(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let data: Vec<f32> = store.to_host(a).iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    let shape = store.shape(a).to_vec();
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Sigmoid, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Tensor(out),
    });
    out
}

#[cfg(feature = "cuda")]
pub fn sigmoid_forward(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let shape = store.shape(a).to_vec();
    let n = shape_size(&shape);
    let out = store.zeros(&shape);
    let dev = GpuDevice::instance();
    unsafe {
        dev.stream.launch_builder(dev.get_func("sigmoid_forward_f32"))
            .arg(&store.dev_ptr(out))
            .arg(&store.dev_ptr(a))
            .arg(&(n as i32))
            .launch(launch_cfg(n as u32))
            .unwrap();
    }
    tape.record(TapeEntry {
        op: BackwardOp::Sigmoid, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Tensor(out),
    });
    out
}

// ---------------------------------------------------------------------------
// Sigmoid backward: dx = dy * out * (1 - out)
// ---------------------------------------------------------------------------

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn sigmoid_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensor(out) = saved {
        let out_data = store.to_host(*out);
        let grad_data = store.to_host(grad);
        let data: Vec<f32> = grad_data.iter().zip(&out_data)
            .map(|(g, &o)| g * o * (1.0 - o)).collect();
        let shape = store.shape(grad).to_vec();
        vec![Some(store.from_vec(data, &shape))]
    } else { vec![None] }
}

#[cfg(feature = "cuda")]
pub fn sigmoid_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensor(out) = saved {
        let shape = store.shape(grad).to_vec();
        let n = shape_size(&shape);
        let result = store.zeros(&shape);
        let dev = GpuDevice::instance();
        unsafe {
            dev.stream.launch_builder(dev.get_func("sigmoid_backward_f32"))
                .arg(&store.dev_ptr(result))
                .arg(&store.dev_ptr(grad))
                .arg(&store.dev_ptr(*out))
                .arg(&(n as i32))
                .launch(launch_cfg(n as u32))
                .unwrap();
        }
        vec![Some(result)]
    } else { vec![None] }
}
