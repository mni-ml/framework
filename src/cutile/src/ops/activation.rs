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

macro_rules! unary_activation {
    ($name:ident, $kernel:ident, $label:literal) => {
        pub fn $name(store: &mut TensorStore, a: TensorId) -> TensorId {
            let shape = store.shape(a).to_vec();
            let size = shape_size(&shape);
            let block = pick_block(size);
            let rt = runtime();
            let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
            {
                let at = store.tensor(a);
                let av = at.view(&[size]).expect("view a");
                let _ = kernels::$kernel((&mut out).partition([block]), &av)
                    .sync_on(&rt.stream)
                    .expect(concat!($label, " kernel"));
            }
            let logical = out.reshape(&shape).expect("reshape");
            store.insert_tensor(logical, shape)
        }
    };
}

unary_activation!(relu, relu, "relu");
unary_activation!(gelu, gelu_forward, "gelu");
unary_activation!(sigmoid, sigmoid_forward, "sigmoid");

/// dx = (x > 0) ? grad : 0.  Kernel arg order: `(dx, x, grad)`.
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

/// dx = dy · gelu'(x).  Kernel arg order: `(dx, dy, x)`.
pub fn gelu_backward(store: &mut TensorStore, dy: TensorId, x: TensorId) -> TensorId {
    let shape = store.shape(x).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(dy), size, "gelu_backward: shape mismatch");
    let block = pick_block(size);
    let rt = runtime();
    let mut dx = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let dyt = store.tensor(dy);
        let xt = store.tensor(x);
        let dyv = dyt.view(&[size]).expect("view dy");
        let xv = xt.view(&[size]).expect("view x");
        let _ = kernels::gelu_backward((&mut dx).partition([block]), &dyv, &xv)
            .sync_on(&rt.stream)
            .expect("gelu_backward kernel");
    }
    let logical = dx.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

/// dx = dy · σ(x) · (1 - σ(x)) — uses the saved forward `out` (not the input).
/// Kernel arg order: `(dx, dy, out)`.
pub fn sigmoid_backward(store: &mut TensorStore, dy: TensorId, out: TensorId) -> TensorId {
    let shape = store.shape(out).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(dy), size, "sigmoid_backward: shape mismatch");
    let block = pick_block(size);
    let rt = runtime();
    let mut dx = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let dyt = store.tensor(dy);
        let ot = store.tensor(out);
        let dyv = dyt.view(&[size]).expect("view dy");
        let ov = ot.view(&[size]).expect("view out");
        let _ = kernels::sigmoid_backward((&mut dx).partition([block]), &dyv, &ov)
            .sync_on(&rt.stream)
            .expect("sigmoid_backward kernel");
    }
    let logical = dx.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}
