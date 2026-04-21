//! Global reductions implemented as a classic CUB-style two-pass (really
//! N-pass) reduction on top of cuTile.
//!
//! Each pass runs `kernels::sum_block::<BLOCK>`, which reduces a BLOCK-sized
//! chunk of its input down to one scalar written to the corresponding slot
//! of a `partials` buffer.  We ping-pong partials back into the kernel until
//! one element remains.  BLOCK is a const generic picked per-pass so the
//! largest candidate from `CANDIDATE_BLOCKS` that divides the current size
//! wins; if the remainder ever becomes indivisible by any of those (e.g. a
//! stray prime factor), we finish the tail on the host so correctness
//! doesn't depend on power-of-two inputs.
//!
//! Ideally the tail would either pad with the reduction identity or call a
//! Thrust/CUB C-API binding; that's not in scope here.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{shape_size, TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Reshape, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};

const CANDIDATE_BLOCKS: [usize; 9] = [256, 128, 64, 32, 16, 8, 4, 2, 1];

/// Largest block size from `CANDIDATE_BLOCKS` (strictly > 1) that divides `n`.
/// Returns `None` if no non-trivial divisor exists (i.e. further reduction
/// needs a host fallback).
fn pick_reduce_block(n: usize) -> Option<usize> {
    for &b in &CANDIDATE_BLOCKS {
        if b >= 2 && b <= n && n % b == 0 {
            return Some(b);
        }
    }
    None
}

/// Sum over all elements.  Returns a new scalar tensor with shape `[1]`.
pub fn sum_all(store: &mut TensorStore, a: TensorId) -> TensorId {
    let n = shape_size(&store.shape(a).to_vec());
    let rt = runtime();

    if n <= 1 {
        // Degenerate: just copy through as a 1-element tensor.
        let host = store.to_host(a);
        let total: f32 = host.iter().copied().sum();
        return store.from_slice(&[total], &[1]);
    }

    // Seed the pipeline with an owned 1D copy of the input so we can ping-pong
    // partials independent of the store.
    let mut current: Tensor<f32> = store
        .tensor(a)
        .dup()
        .sync_on(&rt.stream)
        .expect("dup input")
        .reshape(&[n])
        .expect("reshape input to 1D");
    let mut size = n;

    while size > 1 {
        let block = match pick_reduce_block(size) {
            Some(b) => b,
            None => {
                // Indivisible tail — finish on the host.
                let host = current
                    .dup()
                    .to_host_vec()
                    .sync_on(&rt.stream)
                    .expect("d2h tail");
                let total: f32 = host.iter().copied().sum();
                return store.from_slice(&[total], &[1]);
            }
        };
        let nblocks = size / block;
        let mut partials = api::zeros::<f32>(&[nblocks])
            .sync_on(&rt.stream)
            .expect("alloc partials");
        {
            let xv = current.view(&[size]).expect("view current");
            let _ = kernels::sum_block((&mut partials).partition([1]), &xv)
                .generics(vec![block.to_string()])
                .sync_on(&rt.stream)
                .expect("sum_block kernel");
        }
        current = partials;
        size = nblocks;
    }

    // size == 1; reshape to the framework's scalar convention.
    let one = current.reshape(&[1]).expect("reshape to [1]");
    store.insert_tensor(one, vec![1])
}

/// Mean over all elements.  Returns a new scalar tensor with shape `[1]`.
pub fn mean_all(store: &mut TensorStore, a: TensorId) -> TensorId {
    let n = shape_size(&store.shape(a).to_vec()).max(1);
    let sum_id = sum_all(store, a);
    let scaled = crate::ops::elementwise::mul_scalar(store, sum_id, 1.0 / n as f32);
    store.free(sum_id);
    scaled
}
