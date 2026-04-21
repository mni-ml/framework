//! Activation ops backed by cuTile kernels.

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

pub fn relu(store: &mut TensorStore, a: TensorId) -> TensorId {
    let shape = store.shape(a).to_vec();
    let size = shape_size(&shape);
    let block = pick_block(size);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let at = store.tensor(a);
        let av = at.view(&[size]).expect("view a");
        let _ = kernels::relu((&mut out).partition([block]), &av)
            .sync_on(&rt.stream)
            .expect("relu kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

/// dx = grad * (x > 0 ? 1 : 0).  Returns a freshly allocated `dx` tensor.
pub fn relu_backward(store: &mut TensorStore, x: TensorId, grad: TensorId) -> TensorId {
    let shape = store.shape(x).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(grad), size, "relu_backward: shape mismatch");
    let block = pick_block(size);
    let rt = runtime();
    let mut dx = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let xt = store.tensor(x);
        let gt = store.tensor(grad);
        let xv = xt.view(&[size]).expect("view x");
        let gv = gt.view(&[size]).expect("view grad");
        let _ = kernels::relu_backward((&mut dx).partition([block]), &xv, &gv)
            .sync_on(&rt.stream)
            .expect("relu_backward kernel");
    }
    let logical = dx.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}
