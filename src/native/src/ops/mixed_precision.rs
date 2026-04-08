use crate::tensor::{TensorId, TensorStore};

#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

#[cfg(feature = "cuda")]
fn launch_cfg(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Scale all gradients by inv_scale for GradScaler unscale step.
#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn scale_grads(param_ids: &[TensorId], inv_scale: f32, store: &mut TensorStore) -> bool {
    let mut found_inf = false;
    for &pid in param_ids {
        if let Some(grad_id) = store.get(pid).grad {
            let data = store.data_mut(grad_id);
            for v in data.iter_mut() {
                *v *= inv_scale;
                if v.is_infinite() || v.is_nan() {
                    found_inf = true;
                }
            }
        }
    }
    found_inf
}

#[cfg(feature = "cuda")]
pub fn scale_grads(param_ids: &[TensorId], inv_scale: f32, store: &mut TensorStore) -> bool {
    let dev = GpuDevice::instance();

    for &pid in param_ids {
        if let Some(grad_id) = store.get(pid).grad {
            let size = store.size(grad_id);
            let grad_ptr = store.dev_ptr(grad_id);
            let func = dev.get_func("scale_f32");
            unsafe {
                dev.stream.launch_builder(func)
                    .arg(&grad_ptr)
                    .arg(&inv_scale)
                    .arg(&(size as i32))
                    .launch(launch_cfg(size as u32))
                    .unwrap();
            }
        }
    }

    // Check for inf/nan
    let result_buf: CudaSlice<f32> = dev.stream.alloc_zeros(1).unwrap();
    let result_ptr = dev.ptr(&result_buf);

    for &pid in param_ids {
        if let Some(grad_id) = store.get(pid).grad {
            let size = store.size(grad_id);
            let grad_ptr = store.dev_ptr(grad_id);
            let func = dev.get_func("check_inf_nan_f32");
            unsafe {
                dev.stream.launch_builder(func)
                    .arg(&result_ptr)
                    .arg(&grad_ptr)
                    .arg(&(size as i32))
                    .launch(launch_cfg(size as u32))
                    .unwrap();
            }
        }
    }

    let result: Vec<f32> = dev.stream.memcpy_dtov(&result_buf).unwrap();
    result[0] > 0.0
}
