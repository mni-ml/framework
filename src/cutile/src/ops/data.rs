//! Data-loader ops: device-side `sample_batch` matching `kernels::sample_batch`.
//!
//! `Dataset` owns a device-side `Tensor<i32>` (full token stream).
//! `sample_batch` takes a host `&[i32]` of per-element start offsets, uploads
//! them, and emits two `[B, BLOCK]` device tensors for inputs and targets
//! (targets are inputs shifted by +1, matching the CUDA reference).
//!
//! Inputs and targets stay as `Tensor<i32>` in `SampledBatch`; the f32
//! `TensorStore` doesn't hold them.  Callers pass them straight to
//! cross-entropy / embedding which take device i32 tensors directly.

use crate::device::runtime;
use crate::kernels;
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;

const SAMPLE_BLOCK_CANDIDATES: [usize; 6] = [1024, 512, 256, 128, 64, 32];

fn pick_sample_block(block: usize) -> usize {
    for &b in &SAMPLE_BLOCK_CANDIDATES {
        if block % b == 0 {
            return b;
        }
    }
    1
}

/// Owned device-side i32 dataset — load once, sample many batches from it.
pub struct Dataset {
    pub data: Tensor<i32>,
    pub len: usize,
}

impl Dataset {
    pub fn from_host(host: &[i32]) -> Self {
        let rt = runtime();
        let arc = Arc::new(host.to_vec());
        let data = api::copy_host_vec_to_device(&arc)
            .sync_on(&rt.stream)
            .expect("dataset h2d");
        Self {
            data,
            len: host.len(),
        }
    }
}

pub struct SampledBatch {
    pub inputs: Tensor<i32>,
    pub targets: Tensor<i32>,
    pub batch_size: usize,
    pub block_size: usize,
}

/// Gather `[batch_size, block_size]` token windows from `dataset` at
/// host-supplied per-batch start offsets.  `block_size` must equal
/// `inputs.shape()[1]` and is baked in as a const generic; we pick the
/// largest divisor from `[1024, 512, 256, 128, 64, 32]`.
///
/// Caller must guarantee that every offset is in `[0, dataset.len - block_size - 1]`
/// — `sample_batch` reads `block_size + 1` tokens per element, so the last
/// valid offset is `len - block_size - 1` (target shift).
pub fn sample_batch(
    dataset: &Dataset,
    offsets: &[i32],
    block_size: usize,
) -> SampledBatch {
    let batch_size = offsets.len();
    assert!(batch_size > 0, "sample_batch: empty offsets");
    assert!(block_size > 0, "sample_batch: zero block_size");
    let block_const = pick_sample_block(block_size);
    assert_eq!(
        block_size, block_const,
        "sample_batch: block_size {block_size} not in {SAMPLE_BLOCK_CANDIDATES:?}"
    );

    let rt = runtime();
    let off_arc = Arc::new(offsets.to_vec());
    let off_tensor = api::copy_host_vec_to_device(&off_arc)
        .sync_on(&rt.stream)
        .expect("offsets h2d");

    let mut inputs = api::zeros::<i32>(&[batch_size, block_size])
        .sync_on(&rt.stream)
        .expect("alloc inputs");
    let mut targets = api::zeros::<i32>(&[batch_size, block_size])
        .sync_on(&rt.stream)
        .expect("alloc targets");

    let dataset_ptr = dataset.data.device_pointer();
    {
        let off_view = off_tensor.view(&[batch_size]).expect("view offsets");
        unsafe {
            let _ = kernels::sample_batch(
                (&mut inputs).partition([1, block_size]),
                (&mut targets).partition([1, block_size]),
                dataset_ptr,
                &off_view,
            )
            .generics(vec![block_size.to_string()])
            .sync_on(&rt.stream)
            .expect("sample_batch kernel");
        }
    }

    SampledBatch {
        inputs,
        targets,
        batch_size,
        block_size,
    }
}
