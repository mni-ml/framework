//! Softmax forward + backward ops.
//!
//! Kernels reduce along the last axis only; for arbitrary `dim` the ops
//! layer permutes the reduction axis to the end, invokes the kernel, then
//! permutes back.  Matches the `(outer, dim, inner)` fast path in the CUDA
//! backend, where the `dim == -1` case is linear in memory.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{shape_size, TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Reshape};
use cutile::tile_kernel::TileKernel;

const ROW_CANDIDATES: [usize; 6] = [32, 16, 8, 4, 2, 1];

fn pick_row_block(n: usize) -> usize {
    for &b in &ROW_CANDIDATES {
        if n % b == 0 {
            return b;
        }
    }
    1
}

fn normalize_axis(dim: i32, rank: usize) -> usize {
    if dim < 0 {
        (rank as i32 + dim) as usize
    } else {
        dim as usize
    }
}

fn perm_axis_to_last(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank).filter(|&i| i != axis).collect();
    perm.push(axis);
    perm
}

fn perm_last_to_axis(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank - 1).collect();
    perm.insert(axis, rank - 1);
    perm
}

fn softmax_last(store: &mut TensorStore, a: TensorId) -> TensorId {
    let shape = store.shape(a).to_vec();
    let rank = shape.len();
    assert!(rank >= 1, "softmax: rank must be >= 1");
    let dim = *shape.last().unwrap();
    let outer: usize = shape[..rank - 1].iter().product::<usize>().max(1);
    let bm = pick_row_block(outer);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[outer, dim])
        .sync_on(&rt.stream)
        .expect("alloc");
    {
        let xt = store.tensor(a);
        let xv = xt.view(&[outer, dim]).expect("view x");
        let _ = kernels::softmax_forward((&mut out).partition([bm, dim]), &xv)
            .generics(vec![bm.to_string(), dim.to_string()])
            .sync_on(&rt.stream)
            .expect("softmax_forward kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

fn softmax_last_backward(
    store: &mut TensorStore,
    dy: TensorId,
    out: TensorId,
) -> TensorId {
    let shape = store.shape(out).to_vec();
    assert_eq!(store.shape(dy), shape, "softmax_backward: shape mismatch");
    let rank = shape.len();
    let dim = *shape.last().unwrap();
    let outer: usize = shape[..rank - 1].iter().product::<usize>().max(1);
    let bm = pick_row_block(outer);
    let rt = runtime();
    let mut dx = api::zeros::<f32>(&[outer, dim])
        .sync_on(&rt.stream)
        .expect("alloc");
    {
        let dyt = store.tensor(dy);
        let ot = store.tensor(out);
        let dyv = dyt.view(&[outer, dim]).expect("view dy");
        let ov = ot.view(&[outer, dim]).expect("view out");
        let _ = kernels::softmax_backward((&mut dx).partition([bm, dim]), &dyv, &ov)
            .generics(vec![bm.to_string(), dim.to_string()])
            .sync_on(&rt.stream)
            .expect("softmax_backward kernel");
    }
    let logical = dx.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

pub fn softmax_forward(store: &mut TensorStore, a: TensorId, dim: i32) -> TensorId {
    let shape = store.shape(a).to_vec();
    let rank = shape.len();
    let axis = normalize_axis(dim, rank);
    assert!(axis < rank, "softmax: dim out of range");

    if axis == rank - 1 {
        return softmax_last(store, a);
    }
    let perm = perm_axis_to_last(rank, axis);
    let permuted = crate::ops::elementwise::permute(store, a, &perm);
    let softmaxed = softmax_last(store, permuted);
    store.free(permuted);
    let inv = perm_last_to_axis(rank, axis);
    let unpermuted = crate::ops::elementwise::permute(store, softmaxed, &inv);
    store.free(softmaxed);
    unpermuted
}

pub fn softmax_backward(
    store: &mut TensorStore,
    dy: TensorId,
    out: TensorId,
    dim: i32,
) -> TensorId {
    let shape = store.shape(out).to_vec();
    let rank = shape.len();
    let axis = normalize_axis(dim, rank);
    assert!(axis < rank, "softmax_backward: dim out of range");

    if axis == rank - 1 {
        return softmax_last_backward(store, dy, out);
    }
    let perm = perm_axis_to_last(rank, axis);
    let dy_p = crate::ops::elementwise::permute(store, dy, &perm);
    let out_p = crate::ops::elementwise::permute(store, out, &perm);
    let dx_p = softmax_last_backward(store, dy_p, out_p);
    store.free(dy_p);
    store.free(out_p);
    let inv = perm_last_to_axis(rank, axis);
    let dx = crate::ops::elementwise::permute(store, dx_p, &inv);
    store.free(dx_p);
    dx
}

// Silence unused-import warnings if this module is compiled without any
// caller that uses `shape_size`.
#[allow(dead_code)]
const _: fn(&[usize]) -> usize = shape_size;
