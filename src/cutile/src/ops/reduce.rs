//! Reduction ops on top of the cuTile reduce kernels.
//!
//! * **Global sum / mean** — `sum_all`, `mean_all` drive the single-pass
//!   atomic `sum_atomic` kernel described below.
//! * **Along-dim sum / mean / max** — `sum_along_dim` & friends collapse a
//!   single axis.  The kernels (`*_along_last`) only reduce the last axis;
//!   the ops layer permutes arbitrary axes to the last position via the
//!   `permute` kernel.
//! * **Broadcast** — `sum_broadcast` expands a `[*, 1]`-shaped reduction
//!   back across a new last dim, matching `sum_broadcast_f32` in the CUDA
//!   backend.
//!
//! Global reductions follow the CUDA C++ SIMT backend's pattern:
//! single-pass atomic.  We launch `sum_atomic::<BLOCK>` with a grid of
//! `ceil(n / BLOCK)` blocks; each block does an in-tile `reduce_sum` and
//! `atomic_rmw_tko "addf"`s the partial into a zero-initialised scalar
//! output buffer.  Tail handling is automatic — cuTile's `partition()`
//! returns a zero-padded view, so out-of-range lanes contribute the
//! additive identity `0.0f32` and the reduction stays correct for any
//! `n` with no divisibility constraints.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{shape_size, TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Reshape, Tensor};
use cutile::tile_kernel::TileKernel;

/// Elements-per-block for the single-pass atomic reduction.  Matches the
/// CUDA C++ SIMT backend's block size.
const SUM_BLOCK: usize = 1024;

/// Sum over all elements.  Returns a new scalar tensor with shape `[1]`.
pub fn sum_all(store: &mut TensorStore, a: TensorId) -> TensorId {
    let n = shape_size(&store.shape(a).to_vec());
    let rt = runtime();

    if n == 0 {
        return store.from_slice(&[0.0f32], &[1]);
    }

    let input_1d: Tensor<f32> = store
        .tensor(a)
        .dup()
        .sync_on(&rt.stream)
        .expect("dup input")
        .reshape(&[n])
        .expect("reshape input to 1D");

    let result = api::zeros::<f32>(&[1])
        .sync_on(&rt.stream)
        .expect("alloc result");
    let result_ptr = result.device_pointer();
    let grid = n.div_ceil(SUM_BLOCK) as u32;
    {
        let xv = input_1d.view(&[n]).expect("view input");
        unsafe {
            let _ = kernels::sum_atomic(result_ptr, &xv)
                .grid((grid, 1, 1))
                .generics(vec![SUM_BLOCK.to_string()])
                .sync_on(&rt.stream)
                .expect("sum_atomic kernel");
        }
    }
    store.insert_tensor(result, vec![1])
}

/// Mean over all elements.
pub fn mean_all(store: &mut TensorStore, a: TensorId) -> TensorId {
    let n = shape_size(&store.shape(a).to_vec()).max(1);
    let sum_id = sum_all(store, a);
    let scaled = crate::ops::elementwise::mul_scalar(store, sum_id, 1.0 / n as f32);
    store.free(sum_id);
    scaled
}

// ---------------------------------------------------------------------------
// Along-dim reductions.
//
// Input shape is viewed as `[outer, dim, inner]`.  For `dim == last` (inner
// = 1) the kernels reduce directly.  For an arbitrary axis, the ops layer
// permutes the target axis to the last before launching, mirroring the CUDA
// backend's fast path.
// ---------------------------------------------------------------------------

const ROW_CANDIDATES: [usize; 6] = [32, 16, 8, 4, 2, 1];

fn pick_row_block(n: usize) -> usize {
    for &b in &ROW_CANDIDATES {
        if n % b == 0 {
            return b;
        }
    }
    1
}

/// Axis normalization: `dim < 0` counts from the end.
fn normalize_axis(dim: i32, rank: usize) -> usize {
    if dim < 0 {
        (rank as i32 + dim) as usize
    } else {
        dim as usize
    }
}

/// Builds a permutation that moves `axis` to the end.
fn perm_axis_to_last(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank).filter(|&i| i != axis).collect();
    perm.push(axis);
    perm
}

/// Builds the inverse of `perm_axis_to_last` — moves the last back to `axis`.
fn perm_last_to_axis(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank - 1).collect();
    perm.insert(axis, rank - 1);
    perm
}

enum ReduceKind {
    Sum,
    Mean,
    Max,
}

fn along_last(
    store: &mut TensorStore,
    a: TensorId,
    kind: ReduceKind,
    keepdim: bool,
) -> TensorId {
    let shape = store.shape(a).to_vec();
    let rank = shape.len();
    assert!(rank >= 1, "reduce: rank must be >= 1");
    let dim = *shape.last().unwrap();
    let outer: usize = shape[..rank - 1].iter().product();
    let bm = pick_row_block(outer.max(1));
    let rt = runtime();

    let mut out_flat = api::zeros::<f32>(&[outer])
        .sync_on(&rt.stream)
        .expect("alloc");
    {
        let xt = store.tensor(a);
        let xv = xt.view(&[outer, dim]).expect("view x");
        let gen = vec![bm.to_string(), dim.to_string()];
        match kind {
            ReduceKind::Sum => {
                let _ = kernels::sum_along_last((&mut out_flat).partition([bm]), &xv)
                    .generics(gen)
                    .sync_on(&rt.stream)
                    .expect("sum_along_last");
            }
            ReduceKind::Mean => {
                let _ = kernels::mean_along_last((&mut out_flat).partition([bm]), &xv)
                    .generics(gen)
                    .sync_on(&rt.stream)
                    .expect("mean_along_last");
            }
            ReduceKind::Max => {
                let _ = kernels::max_along_last((&mut out_flat).partition([bm]), &xv)
                    .generics(gen)
                    .sync_on(&rt.stream)
                    .expect("max_along_last");
            }
        }
    }

    // Final shape: outer shape with an optional trailing 1 for keepdim.
    let mut out_shape: Vec<usize> = shape[..rank - 1].to_vec();
    if keepdim {
        out_shape.push(1);
    }
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let logical = out_flat.reshape(&out_shape).expect("reshape");
    store.insert_tensor(logical, out_shape)
}

fn along_dim_impl(
    store: &mut TensorStore,
    a: TensorId,
    dim: i32,
    keepdim: bool,
    kind: ReduceKind,
) -> TensorId {
    let shape = store.shape(a).to_vec();
    let rank = shape.len();
    let axis = normalize_axis(dim, rank);
    assert!(axis < rank, "reduce: dim {dim} out of range for rank {rank}");

    if axis == rank - 1 {
        return along_last(store, a, kind, keepdim);
    }

    let perm = perm_axis_to_last(rank, axis);
    let permuted = crate::ops::elementwise::permute(store, a, &perm);
    let reduced = along_last(store, permuted, kind, keepdim);
    store.free(permuted);

    if keepdim {
        // permuted shape: shape[..axis] ++ shape[axis+1..] ++ [shape[axis]]
        // after keepdim reduce: shape[..axis] ++ shape[axis+1..] ++ [1]
        // we want: shape[..axis] ++ [1] ++ shape[axis+1..]
        let inv = perm_last_to_axis(rank, axis);
        let unpermuted = crate::ops::elementwise::permute(store, reduced, &inv);
        store.free(reduced);
        unpermuted
    } else {
        reduced
    }
}

pub fn sum_along_dim(
    store: &mut TensorStore,
    a: TensorId,
    dim: i32,
    keepdim: bool,
) -> TensorId {
    along_dim_impl(store, a, dim, keepdim, ReduceKind::Sum)
}

pub fn mean_along_dim(
    store: &mut TensorStore,
    a: TensorId,
    dim: i32,
    keepdim: bool,
) -> TensorId {
    along_dim_impl(store, a, dim, keepdim, ReduceKind::Mean)
}

pub fn max_along_dim(
    store: &mut TensorStore,
    a: TensorId,
    dim: i32,
    keepdim: bool,
) -> TensorId {
    along_dim_impl(store, a, dim, keepdim, ReduceKind::Max)
}

/// `sum_broadcast(x, dim, size)` — broadcast `x` across a new axis of length
/// `size` inserted at position `dim`.  Used as the forward half of
/// sum-backward (dy → dx = broadcast(dy) to x's original shape).
pub fn sum_broadcast(
    store: &mut TensorStore,
    x: TensorId,
    dim: i32,
    size: usize,
) -> TensorId {
    let shape = store.shape(x).to_vec();
    let rank = shape.len();
    let axis = normalize_axis(dim, rank + 1);
    assert!(axis <= rank, "sum_broadcast: dim out of range");

    if axis == rank {
        // Expanding the last axis directly — invoke broadcast_last.
        let outer: usize = shape.iter().product::<usize>().max(1);
        let bm = pick_row_block(outer);
        let rt = runtime();
        let mut out = api::zeros::<f32>(&[outer, size])
            .sync_on(&rt.stream)
            .expect("alloc");
        {
            let xt = store.tensor(x);
            let xv = xt.view(&[outer]).expect("view x");
            let _ = kernels::broadcast_last((&mut out).partition([bm, size]), &xv)
                .generics(vec![bm.to_string(), size.to_string()])
                .sync_on(&rt.stream)
                .expect("broadcast_last kernel");
        }
        let mut final_shape: Vec<usize> = shape;
        final_shape.push(size);
        let logical = out.reshape(&final_shape).expect("reshape");
        return store.insert_tensor(logical, final_shape);
    }

    // General case: broadcast to end, then permute the new axis into place.
    let intermediate = sum_broadcast(store, x, rank as i32, size);
    let new_rank = rank + 1;
    let inv = perm_last_to_axis(new_rank, axis);
    let result = crate::ops::elementwise::permute(store, intermediate, &inv);
    store.free(intermediate);
    result
}
