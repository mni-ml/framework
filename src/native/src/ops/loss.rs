use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::ops::data::IntStore;
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
// Forward
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn cross_entropy(
    logits: TensorId, targets: &[usize],
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let shape = store.shape(logits).to_vec();
    let ndim = shape.len();
    let v = shape[ndim - 1];
    let n = shape_size(&shape) / v;
    let data = store.to_host(logits);

    let mut softmax_buf = vec![0.0f32; n * v];
    let mut loss_sum = 0.0f32;

    for i in 0..n {
        let off = i * v;
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..v {
            max_val = max_val.max(data[off + j]);
        }
        let mut sum = 0.0f32;
        for j in 0..v {
            let e = (data[off + j] - max_val).exp();
            softmax_buf[off + j] = e;
            sum += e;
        }
        for j in 0..v {
            softmax_buf[off + j] /= sum;
        }
        let target = targets[i];
        loss_sum += -softmax_buf[off + target].ln();
    }

    let loss = loss_sum / n as f32;
    let loss_id = store.from_vec(vec![loss], &[1]);
    let sm_id = store.from_vec(softmax_buf, &shape);

    tape.record(TapeEntry {
        op: BackwardOp::CrossEntropy, output_id: loss_id,
        input_ids: smallvec![logits],
        saved: SavedContext::Indices(targets.to_vec(), n, v, sm_id),
    });
    loss_id
}

#[cfg(feature = "cuda")]
pub fn cross_entropy(
    logits: TensorId, targets: &[usize],
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let shape = store.shape(logits).to_vec();
    let ndim = shape.len();
    let v = shape[ndim - 1];
    let n = shape_size(&shape) / v;

    let dev = GpuDevice::instance();
    let logits_ptr = store.dev_ptr(logits);

    let targets_i32: Vec<i32> = targets.iter().map(|&t| t as i32).collect();
    let targets_gpu = dev.stream.memcpy_stod(&targets_i32).unwrap();
    let targets_ptr = dev.ptr(&targets_gpu);

    let losses_id = store.zeros(&[n]);
    let losses_ptr = store.dev_ptr(losses_id);
    let sm_id = store.zeros(&shape);
    let sm_ptr = store.dev_ptr(sm_id);

    let func = dev.get_func("cross_entropy_forward_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&losses_ptr)
            .arg(&sm_ptr)
            .arg(&logits_ptr)
            .arg(&targets_ptr)
            .arg(&(n as i32))
            .arg(&(v as i32))
            .launch(launch_cfg(n as u32))
            .unwrap();
    }

    let loss_id = store.zeros(&[1]);
    let loss_ptr = store.dev_ptr(loss_id);
    let mean_func = dev.get_func("mean_along_dim_f32");
    unsafe {
        dev.stream.launch_builder(mean_func)
            .arg(&loss_ptr)
            .arg(&losses_ptr)
            .arg(&1i32)
            .arg(&(n as i32))
            .arg(&1i32)
            .arg(&1i32)
            .launch(launch_cfg(1))
            .unwrap();
    }

    tape.record(TapeEntry {
        op: BackwardOp::CrossEntropy, output_id: loss_id,
        input_ids: smallvec![logits],
        saved: SavedContext::Indices(targets.to_vec(), n, v, sm_id),
    });
    loss_id
}

// =========================================================================
// Backward
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn cross_entropy_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Indices(targets, n, v, sm_id) = saved {
        let grad_val = store.get_scalar(grad);
        let sm_data = store.to_host(*sm_id);
        let shape = store.shape(*sm_id).to_vec();

        let mut dlogits = sm_data.clone();
        for i in 0..*n {
            let off = i * v;
            dlogits[off + targets[i]] -= 1.0;
            let scale = grad_val / *n as f32;
            for j in 0..*v {
                dlogits[off + j] *= scale;
            }
        }
        vec![Some(store.from_vec(dlogits, &shape))]
    } else { vec![None] }
}

#[cfg(feature = "cuda")]
pub fn cross_entropy_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Indices(targets, n, v, sm_id) = saved {
        let dev = GpuDevice::instance();
        let grad_val = store.get_scalar(grad);
        let grad_scale = grad_val / *n as f32;
        let shape = store.shape(*sm_id).to_vec();

        let sm_ptr = store.dev_ptr(*sm_id);

        let targets_i32: Vec<i32> = targets.iter().map(|&t| t as i32).collect();
        let targets_gpu = dev.stream.memcpy_stod(&targets_i32).unwrap();
        let targets_ptr = dev.ptr(&targets_gpu);

        let dlogits_id = store.zeros(&shape);
        let dlogits_ptr = store.dev_ptr(dlogits_id);

        let func = dev.get_func("cross_entropy_backward_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&dlogits_ptr)
                .arg(&sm_ptr)
                .arg(&targets_ptr)
                .arg(&grad_scale)
                .arg(&(*n as i32))
                .arg(&(*v as i32))
                .launch(launch_cfg(*n as u32))
                .unwrap();
        }

        vec![Some(dlogits_id)]
    } else { vec![None] }
}

// =========================================================================
// GPU-index cross entropy (targets already on GPU via IntStore, zero PCIe)
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn cross_entropy_gpu(
    logits: TensorId, int_buf_id: usize,
    int_store: &IntStore, store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let shape = store.shape(logits).to_vec();
    let ndim = shape.len();
    let v = shape[ndim - 1];
    let n = shape_size(&shape) / v;
    let data = store.to_host(logits);
    let targets = &int_store.get(int_buf_id).data;

    let mut softmax_buf = vec![0.0f32; n * v];
    let mut loss_sum = 0.0f32;

    for i in 0..n {
        let off = i * v;
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..v { max_val = max_val.max(data[off + j]); }
        let mut sum = 0.0f32;
        for j in 0..v {
            let e = (data[off + j] - max_val).exp();
            softmax_buf[off + j] = e;
            sum += e;
        }
        for j in 0..v { softmax_buf[off + j] /= sum; }
        let target = targets[i] as usize;
        loss_sum += -softmax_buf[off + target].ln();
    }

    let loss = loss_sum / n as f32;
    let loss_id = store.from_vec(vec![loss], &[1]);
    let sm_id = store.from_vec(softmax_buf, &shape);

    tape.record(TapeEntry {
        op: BackwardOp::CrossEntropyGpu, output_id: loss_id,
        input_ids: smallvec![logits],
        saved: SavedContext::GpuIndices(int_buf_id, n, v, sm_id),
    });
    loss_id
}

#[cfg(feature = "cuda")]
pub fn cross_entropy_gpu(
    logits: TensorId, int_buf_id: usize,
    int_store: &IntStore, store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let shape = store.shape(logits).to_vec();
    let ndim = shape.len();
    let v = shape[ndim - 1];
    let n = shape_size(&shape) / v;

    let dev = GpuDevice::instance();
    let logits_ptr = store.dev_ptr(logits);
    let targets_ptr = dev.ptr(&int_store.get(int_buf_id).data);

    let losses_id = store.zeros(&[n]);
    let losses_ptr = store.dev_ptr(losses_id);
    let sm_id = store.zeros(&shape);
    let sm_ptr = store.dev_ptr(sm_id);

    let func = dev.get_func("cross_entropy_forward_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&losses_ptr)
            .arg(&sm_ptr)
            .arg(&logits_ptr)
            .arg(&targets_ptr)
            .arg(&(n as i32))
            .arg(&(v as i32))
            .launch(launch_cfg(n as u32))
            .unwrap();
    }

    let loss_id = store.zeros(&[1]);
    let loss_ptr = store.dev_ptr(loss_id);
    let mean_func = dev.get_func("mean_along_dim_f32");
    unsafe {
        dev.stream.launch_builder(mean_func)
            .arg(&loss_ptr)
            .arg(&losses_ptr)
            .arg(&1i32)
            .arg(&(n as i32))
            .arg(&1i32)
            .arg(&1i32)
            .launch(launch_cfg(1))
            .unwrap();
    }

    tape.record(TapeEntry {
        op: BackwardOp::CrossEntropyGpu, output_id: loss_id,
        input_ids: smallvec![logits],
        saved: SavedContext::GpuIndices(int_buf_id, n, v, sm_id),
    });
    loss_id
}

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn cross_entropy_backward_gpu(
    grad: TensorId, saved: &SavedContext,
    int_store: &IntStore, store: &mut TensorStore,
) -> Vec<Option<TensorId>> {
    if let SavedContext::GpuIndices(int_buf_id, n, v, sm_id) = saved {
        let grad_val = store.get_scalar(grad);
        let sm_data = store.to_host(*sm_id);
        let shape = store.shape(*sm_id).to_vec();
        let targets = &int_store.get(*int_buf_id).data;

        let mut dlogits = sm_data.clone();
        for i in 0..*n {
            let off = i * v;
            dlogits[off + targets[i] as usize] -= 1.0;
            let scale = grad_val / *n as f32;
            for j in 0..*v { dlogits[off + j] *= scale; }
        }
        vec![Some(store.from_vec(dlogits, &shape))]
    } else { vec![None] }
}

#[cfg(feature = "cuda")]
pub fn cross_entropy_backward_gpu(
    grad: TensorId, saved: &SavedContext,
    int_store: &IntStore, store: &mut TensorStore,
) -> Vec<Option<TensorId>> {
    if let SavedContext::GpuIndices(int_buf_id, n, v, sm_id) = saved {
        let dev = GpuDevice::instance();
        let grad_val = store.get_scalar(grad);
        let grad_scale = grad_val / *n as f32;
        let shape = store.shape(*sm_id).to_vec();

        let sm_ptr = store.dev_ptr(*sm_id);
        let targets_ptr = dev.ptr(&int_store.get(*int_buf_id).data);

        let dlogits_id = store.zeros(&shape);
        let dlogits_ptr = store.dev_ptr(dlogits_id);

        let func = dev.get_func("cross_entropy_backward_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&dlogits_ptr)
                .arg(&sm_ptr)
                .arg(&targets_ptr)
                .arg(&grad_scale)
                .arg(&(*n as i32))
                .arg(&(*v as i32))
                .launch(launch_cfg(*n as u32))
                .unwrap();
        }

        vec![Some(dlogits_id)]
    } else { vec![None] }
}
