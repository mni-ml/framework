//! Elementwise ops backed by cuTile kernels.
//!
//! All ops flatten the inputs/outputs to 1D and launch a 1D tiled kernel.
//! The block size is chosen to divide the element count so the kernel never
//! writes out of bounds.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{shape_size, TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Reshape};

const CANDIDATE_BLOCKS: [usize; 9] = [256, 128, 64, 32, 16, 8, 4, 2, 1];

/// Largest block size from `CANDIDATE_BLOCKS` that divides `n`.
fn pick_block(n: usize) -> usize {
    for &b in &CANDIDATE_BLOCKS {
        if n % b == 0 {
            return b;
        }
    }
    1
}

pub fn add(store: &mut TensorStore, a: TensorId, b: TensorId) -> TensorId {
    let shape = store.shape(a).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(b), size, "add: shape mismatch");
    let block = pick_block(size);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let at = store.tensor(a);
        let bt = store.tensor(b);
        let av = at.view(&[size]).expect("view a");
        let bv = bt.view(&[size]).expect("view b");
        let _ = kernels::add((&mut out).partition([block]), &av, &bv)
            .sync_on(&rt.stream)
            .expect("add kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

pub fn sub(store: &mut TensorStore, a: TensorId, b: TensorId) -> TensorId {
    let shape = store.shape(a).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(b), size, "sub: shape mismatch");
    let block = pick_block(size);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let at = store.tensor(a);
        let bt = store.tensor(b);
        let av = at.view(&[size]).expect("view a");
        let bv = bt.view(&[size]).expect("view b");
        let _ = kernels::sub((&mut out).partition([block]), &av, &bv)
            .sync_on(&rt.stream)
            .expect("sub kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

pub fn mul(store: &mut TensorStore, a: TensorId, b: TensorId) -> TensorId {
    let shape = store.shape(a).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(b), size, "mul: shape mismatch");
    let block = pick_block(size);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let at = store.tensor(a);
        let bt = store.tensor(b);
        let av = at.view(&[size]).expect("view a");
        let bv = bt.view(&[size]).expect("view b");
        let _ = kernels::mul((&mut out).partition([block]), &av, &bv)
            .sync_on(&rt.stream)
            .expect("mul kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

pub fn div(store: &mut TensorStore, a: TensorId, b: TensorId) -> TensorId {
    let shape = store.shape(a).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(b), size, "div: shape mismatch");
    let block = pick_block(size);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let at = store.tensor(a);
        let bt = store.tensor(b);
        let av = at.view(&[size]).expect("view a");
        let bv = bt.view(&[size]).expect("view b");
        let _ = kernels::div((&mut out).partition([block]), &av, &bv)
            .sync_on(&rt.stream)
            .expect("div kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

pub fn neg(store: &mut TensorStore, a: TensorId) -> TensorId {
    let shape = store.shape(a).to_vec();
    let size = shape_size(&shape);
    let block = pick_block(size);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let at = store.tensor(a);
        let av = at.view(&[size]).expect("view a");
        let _ = kernels::neg((&mut out).partition([block]), &av)
            .sync_on(&rt.stream)
            .expect("neg kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

pub fn mul_scalar(store: &mut TensorStore, a: TensorId, s: f32) -> TensorId {
    let shape = store.shape(a).to_vec();
    let size = shape_size(&shape);
    let block = pick_block(size);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let at = store.tensor(a);
        let av = at.view(&[size]).expect("view a");
        let _ = kernels::mul_scalar((&mut out).partition([block]), &av, s)
            .sync_on(&rt.stream)
            .expect("mul_scalar kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

/// Fused z = a*x + y.
pub fn saxpy(store: &mut TensorStore, a_scalar: f32, x: TensorId, y: TensorId) -> TensorId {
    let shape = store.shape(x).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(y), size, "saxpy: shape mismatch");
    let block = pick_block(size);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let xt = store.tensor(x);
        let yt = store.tensor(y);
        let xv = xt.view(&[size]).expect("view x");
        let yv = yt.view(&[size]).expect("view y");
        let _ = kernels::saxpy((&mut out).partition([block]), a_scalar, &xv, &yv)
            .sync_on(&rt.stream)
            .expect("saxpy kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}
