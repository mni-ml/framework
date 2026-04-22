//! Gradient utility ops: in-place clip, global L2 norm-squared.
//!
//! `grad_norm_sq` single-passes `Σ gradᵢ²` via `grad_norm_sq_atomic` —
//! same pattern as `ops::reduce::sum_all`, matching the CUDA C++ SIMT
//! backend.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{shape_size, TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::PartitionMut;
use cutile::tile_kernel::TileKernel;

const CANDIDATE_BLOCKS: [usize; 9] = [256, 128, 64, 32, 16, 8, 4, 2, 1];

fn pick_block(n: usize) -> usize {
    for &b in &CANDIDATE_BLOCKS {
        if n % b == 0 {
            return b;
        }
    }
    1
}

/// Elements-per-block for the single-pass atomic norm-squared reduction.
/// Matches the CUDA C++ SIMT backend's block size.
const NORM_SQ_BLOCK: usize = 1024;

/// In-place `grad *= scale`.  Returns the same `grad` id for convenience.
pub fn grad_clip(store: &mut TensorStore, grad: TensorId, scale: f32) -> TensorId {
    let shape = store.shape(grad).to_vec();
    let size = shape_size(&shape);
    let block = pick_block(size);
    let rt = runtime();
    {
        let gt = store.tensor_mut(grad);
        let _ = kernels::grad_clip(gt.partition([block]), scale)
            .sync_on(&rt.stream)
            .expect("grad_clip kernel");
    }
    grad
}

/// Returns a freshly-allocated `[1]` scalar tensor with `Σ gradᵢ²`.
pub fn grad_norm_sq(store: &mut TensorStore, grad: TensorId) -> TensorId {
    let n = shape_size(&store.shape(grad).to_vec());
    let rt = runtime();

    if n == 0 {
        return store.from_slice(&[0.0f32], &[1]);
    }

    let result = api::zeros::<f32>(&[1])
        .sync_on(&rt.stream)
        .expect("alloc result");
    let result_ptr = result.device_pointer();
    let grid = n.div_ceil(NORM_SQ_BLOCK) as u32;
    {
        let gt = store.tensor(grad);
        let gv = gt.view(&[n]).expect("view grad");
        unsafe {
            let _ = kernels::grad_norm_sq_atomic(result_ptr, &gv)
                .grid((grid, 1, 1))
                .generics(vec![NORM_SQ_BLOCK.to_string()])
                .sync_on(&rt.stream)
                .expect("grad_norm_sq_atomic kernel");
        }
    }
    store.insert_tensor(result, vec![1])
}
