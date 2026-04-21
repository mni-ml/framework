//! Global reductions implemented as a CUB-style strict two-pass reduction
//! on top of cuTile.
//!
//! Pass 1 (multi-block): launches `sum_block::<PASS1_BLOCK>` with a grid of
//! `ceil(n / PASS1_BLOCK)` blocks.  Each block does an in-tile `reduce_sum`
//! and writes one scalar into a partials buffer — O(n) → O(n / PASS1_BLOCK).
//!
//! Pass 2 (single block): launches the same `sum_block` kernel with grid = 1
//! and a BLOCK const large enough to cover all of pass 1's partials.  One
//! block does an in-tile `reduce_sum` over the whole partials buffer and
//! writes the final scalar — O(tiles) → 1.
//!
//! The last block in pass 1 and the sole block in pass 2 are the usual
//! "partial tile" case: when `n` (resp. `nblocks1`) is not a multiple of
//! the tile size, `Tensor::partition()` returns a **zero-padded** view
//! (`make_partition_view_padded(.., "zero", ..)` in cuTile), so out-of-range
//! tile lanes load 0.0f32 — the identity for sum — and the reduction result
//! is unaffected.  No host tail, no divisibility constraints on n.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{shape_size, TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Reshape, Tensor};
use cutile::tile_kernel::TileKernel;

/// Elements-per-block in pass 1.  Pass 1 launches `ceil(n / PASS1_BLOCK)`
/// blocks, each of which reduces a `PASS1_BLOCK`-sized tile to one scalar.
/// With `PASS1_BLOCK = 2048` and a pass-2 tile cap of 4096, this handles
/// inputs up to ~8M elements in strictly two passes; above that we'd need
/// to either bump `PASS1_BLOCK` (larger tiles cost shared memory) or extend
/// `FINAL_BLOCKS`.
const PASS1_BLOCK: usize = 2048;

/// Candidate pass-2 tile sizes (powers of 2).  Pass 2 is a single-block
/// launch with BLOCK = the smallest candidate ≥ pass-1's partials count.
/// Zero-padded loads cover the slack when `nblocks1` isn't a power of 2.
const FINAL_BLOCKS: [usize; 12] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

fn pick_final_block(n: usize) -> usize {
    for &b in &FINAL_BLOCKS {
        if b >= n {
            return b;
        }
    }
    panic!(
        "reduce size {n} exceeds max pass-2 tile {}",
        FINAL_BLOCKS[FINAL_BLOCKS.len() - 1]
    )
}

/// Sum over all elements.  Returns a new scalar tensor with shape `[1]`.
pub fn sum_all(store: &mut TensorStore, a: TensorId) -> TensorId {
    let n = shape_size(&store.shape(a).to_vec());
    let rt = runtime();

    if n == 0 {
        return store.from_slice(&[0.0f32], &[1]);
    }
    if n == 1 {
        let host = store.to_host(a);
        return store.from_slice(&host, &[1]);
    }

    // Flatten the input to 1D; this also detaches it from the store so the
    // kernel driver owns its own buffer.
    let input_1d: Tensor<f32> = store
        .tensor(a)
        .dup()
        .sync_on(&rt.stream)
        .expect("dup input")
        .reshape(&[n])
        .expect("reshape input to 1D");

    let max_final = *FINAL_BLOCKS.last().unwrap();

    // Small inputs fit in a single pass — one block, zero-padded tile.
    if n <= max_final {
        return launch_final_reduce(store, &input_1d, n);
    }

    // Pass 1: n -> nblocks1.  `ceil(n / PASS1_BLOCK)` blocks, each writes
    // one partial.  Last block's tile is zero-padded for a non-divisible n.
    let nblocks1 = n.div_ceil(PASS1_BLOCK);
    let mut partials = api::zeros::<f32>(&[nblocks1])
        .sync_on(&rt.stream)
        .expect("alloc partials");
    {
        let xv = input_1d.view(&[n]).expect("view input");
        let _ = kernels::sum_block((&mut partials).partition([1]), &xv)
            .generics(vec![PASS1_BLOCK.to_string()])
            .sync_on(&rt.stream)
            .expect("sum_block pass 1");
    }

    // Pass 2: single-block reduction of the partials to one scalar.
    launch_final_reduce(store, &partials, nblocks1)
}

/// Launch `sum_block` with grid = 1 on the given 1D tensor of length `len`,
/// picking a tile size that covers `len` (zero-padding any slack).  Returns
/// a new scalar tensor of shape `[1]` registered in the store.
fn launch_final_reduce(store: &mut TensorStore, src: &Tensor<f32>, len: usize) -> TensorId {
    let rt = runtime();
    let block = pick_final_block(len);
    let mut result = api::zeros::<f32>(&[1])
        .sync_on(&rt.stream)
        .expect("alloc result");
    {
        let sv = src.view(&[len]).expect("view src");
        let _ = kernels::sum_block((&mut result).partition([1]), &sv)
            .generics(vec![block.to_string()])
            .sync_on(&rt.stream)
            .expect("sum_block final");
    }
    store.insert_tensor(result, vec![1])
}

/// Mean over all elements.  Returns a new scalar tensor with shape `[1]`.
pub fn mean_all(store: &mut TensorStore, a: TensorId) -> TensorId {
    let n = shape_size(&store.shape(a).to_vec()).max(1);
    let sum_id = sum_all(store, a);
    let scaled = crate::ops::elementwise::mul_scalar(store, sum_id, 1.0 / n as f32);
    store.free(sum_id);
    scaled
}
