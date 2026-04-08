use smallvec::smallvec;
use rand::Rng;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore};

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
pub fn dropout_forward(
    x: TensorId, rate: f32, training: bool,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    if !training || rate == 0.0 {
        return x;
    }

    let x_data = store.to_host(x);
    let shape = store.shape(x).to_vec();
    let size = x_data.len();
    let scale = 1.0 / (1.0 - rate);

    let mut rng = rand::thread_rng();
    let mut mask = vec![0.0f32; size];
    let mut out = vec![0.0f32; size];

    for i in 0..size {
        let keep = rng.gen::<f32>() >= rate;
        mask[i] = if keep { 1.0 } else { 0.0 };
        out[i] = x_data[i] * mask[i] * scale;
    }

    let mask_id = store.from_vec(mask, &shape);
    let out_id = store.from_vec(out, &shape);

    tape.record(TapeEntry {
        op: BackwardOp::Dropout, output_id: out_id,
        input_ids: smallvec![x],
        saved: SavedContext::DropoutMask(mask_id, rate),
    });
    out_id
}

#[cfg(feature = "cuda")]
pub fn dropout_forward(
    x: TensorId, rate: f32, training: bool,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    if !training || rate == 0.0 {
        return x;
    }

    let shape = store.shape(x).to_vec();
    let size = store.size(x);
    let scale = 1.0 / (1.0 - rate);

    let mut rng = rand::thread_rng();
    let mask_host: Vec<f32> = (0..size)
        .map(|_| if rng.gen::<f32>() >= rate { 1.0 } else { 0.0 })
        .collect();
    let mask_id = store.from_slice(&mask_host, &shape);

    let dev = GpuDevice::instance();
    let x_ptr = store.dev_ptr(x);
    let mask_ptr = store.dev_ptr(mask_id);

    let out_id = store.zeros(&shape);
    let out_ptr = store.dev_ptr(out_id);

    let func = dev.get_func("dropout_apply_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr)
            .arg(&x_ptr)
            .arg(&mask_ptr)
            .arg(&scale)
            .arg(&(size as i32))
            .launch(launch_cfg(size as u32))
            .unwrap();
    }

    tape.record(TapeEntry {
        op: BackwardOp::Dropout, output_id: out_id,
        input_ids: smallvec![x],
        saved: SavedContext::DropoutMask(mask_id, rate),
    });
    out_id
}

// =========================================================================
// Backward
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn dropout_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::DropoutMask(mask_id, rate) = saved {
        let grad_data = store.to_host(grad);
        let mask_data = store.to_host(*mask_id);
        let shape = store.shape(grad).to_vec();
        let scale = 1.0 / (1.0 - rate);

        let out: Vec<f32> = grad_data.iter().zip(&mask_data)
            .map(|(g, m)| g * m * scale).collect();
        vec![Some(store.from_vec(out, &shape))]
    } else { vec![None] }
}

#[cfg(feature = "cuda")]
pub fn dropout_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::DropoutMask(mask_id, rate) = saved {
        let shape = store.shape(grad).to_vec();
        let size = store.size(grad);
        let scale = 1.0 / (1.0 - rate);

        let dev = GpuDevice::instance();
        let grad_ptr = store.dev_ptr(grad);
        let mask_ptr = store.dev_ptr(*mask_id);

        let dx_id = store.zeros(&shape);
        let dx_ptr = store.dev_ptr(dx_id);

        let func = dev.get_func("dropout_backward_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&dx_ptr)
                .arg(&grad_ptr)
                .arg(&mask_ptr)
                .arg(&scale)
                .arg(&(size as i32))
                .launch(launch_cfg(size as u32))
                .unwrap();
        }

        vec![Some(dx_id)]
    } else { vec![None] }
}
