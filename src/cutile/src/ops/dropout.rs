//! Dropout forward + backward.
//!
//! The mask is supplied by the caller as a tensor of 0.0 / 1.0 (typically
//! produced via a host-side bernoulli sample uploaded as a tensor).  The
//! kernel applies `out = x * mask * scale` where `scale = 1 / (1 - p)`.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{shape_size, TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Reshape};

const CANDIDATE_BLOCKS: [usize; 9] = [256, 128, 64, 32, 16, 8, 4, 2, 1];

fn pick_block(n: usize) -> usize {
    for &b in &CANDIDATE_BLOCKS {
        if n % b == 0 {
            return b;
        }
    }
    1
}

pub fn dropout_apply(
    store: &mut TensorStore,
    x: TensorId,
    mask: TensorId,
    p: f32,
) -> TensorId {
    let shape = store.shape(x).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(mask), size, "dropout: mask shape mismatch");
    let block = pick_block(size);
    let scale = 1.0f32 / (1.0f32 - p);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let xt = store.tensor(x);
        let mt = store.tensor(mask);
        let xv = xt.view(&[size]).expect("view x");
        let mv = mt.view(&[size]).expect("view mask");
        let _ = kernels::dropout_apply((&mut out).partition([block]), &xv, &mv, scale)
            .sync_on(&rt.stream)
            .expect("dropout_apply kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

pub fn dropout_backward(
    store: &mut TensorStore,
    dy: TensorId,
    mask: TensorId,
    p: f32,
) -> TensorId {
    let shape = store.shape(dy).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(mask), size, "dropout_backward: mask shape mismatch");
    let block = pick_block(size);
    let scale = 1.0f32 / (1.0f32 - p);
    let rt = runtime();
    let mut dx = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let dyt = store.tensor(dy);
        let mt = store.tensor(mask);
        let dyv = dyt.view(&[size]).expect("view dy");
        let mv = mt.view(&[size]).expect("view mask");
        let _ = kernels::dropout_backward((&mut dx).partition([block]), &dyv, &mv, scale)
            .sync_on(&rt.stream)
            .expect("dropout_backward kernel");
    }
    let logical = dx.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}
