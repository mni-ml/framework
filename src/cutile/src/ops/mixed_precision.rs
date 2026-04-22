//! Mixed-precision ops.
//!
//! `scale_f32` is in-place.  `check_inf_nan_f32` writes 1.0 into a 1-element
//! flag tensor if any element of `data` is non-finite, otherwise leaves it
//! at the caller-zeroed default.  bf16 conversions need `Tensor<u16>`
//! support, which this f32-only TensorStore doesn't expose; `f32_to_bf16`
//! and `bf16_to_f32` will be wired once a multi-dtype store lands (TODO).

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{shape_size, TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::PartitionMut;
use cutile::tensor::Reshape;
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

const NAN_BLOCK: usize = 256;

/// In-place `data *= scale`.  Returns the same `data` id for convenience.
pub fn scale(store: &mut TensorStore, data: TensorId, scale_v: f32) -> TensorId {
    let shape = store.shape(data).to_vec();
    let size = shape_size(&shape);
    let block = pick_block(size);
    let rt = runtime();
    {
        let dt = store.tensor_mut(data);
        let _ = kernels::scale_f32(dt.partition([block]), scale_v)
            .sync_on(&rt.stream)
            .expect("scale_f32 kernel");
    }
    data
}

/// Returns a `[1]` flag tensor: `1.0` if any element of `data` is non-finite
/// (`isinf` or `isnan`), otherwise `0.0`.  Internally fans out across
/// `ceil(n / NAN_BLOCK)` blocks, each atomically umax-ing into the same
/// flag — safe because the bit pattern of `1.0f32` (`0x3F800000u`) is
/// strictly greater (under unsigned compare) than `0.0f32` (`0`).
pub fn check_inf_nan(store: &mut TensorStore, data: TensorId) -> TensorId {
    let n = shape_size(&store.shape(data).to_vec());
    let rt = runtime();
    let flag = api::zeros::<f32>(&[1])
        .sync_on(&rt.stream)
        .expect("alloc flag");

    if n == 0 {
        return store.insert_tensor(flag, vec![1]);
    }

    let grid = n.div_ceil(NAN_BLOCK) as u32;
    let flag_ptr = flag.device_pointer();
    {
        let dt = store.tensor(data);
        let dv = dt.view(&[n]).expect("view data");
        unsafe {
            let _ = kernels::check_inf_nan_f32(flag_ptr, &dv)
                .grid((grid, 1, 1))
                .generics(vec![NAN_BLOCK.to_string()])
                .sync_on(&rt.stream)
                .expect("check_inf_nan_f32 kernel");
        }
    }
    let logical = flag.reshape(&[1]).expect("reshape");
    store.insert_tensor(logical, vec![1])
}
