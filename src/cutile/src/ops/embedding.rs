//! Embedding forward + backward.
//!
//! `D` (embedding dim) is a const generic on the kernel side, so callers
//! must use a fixed embedding width per model.  Indices are taken as
//! `&[i32]` from the host and uploaded to a temporary `Tensor<i32>` —
//! they don't live in the f32 `TensorStore`.
//!
//! Backward zero-inits `dweight` and atomically scatters per-token
//! contributions; if the same index appears multiple times in `indices`,
//! the kernel handles the contention through `atomic_rmw_tko "addf"`.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::PartitionMut;
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;

/// `out[t, :] = weight[indices[t], :]`.  `weight` is `[V, D]`, `indices`
/// has length `T`; result is `[T, D]`.
pub fn embedding_forward(
    store: &mut TensorStore,
    weight: TensorId,
    indices: &[i32],
) -> TensorId {
    let w_shape = store.shape(weight).to_vec();
    assert_eq!(w_shape.len(), 2, "embedding: weight must be [V, D]");
    let (v, d) = (w_shape[0], w_shape[1]);
    let t = indices.len();

    let rt = runtime();
    let idx_arc = Arc::new(indices.to_vec());
    let idx_tensor = api::copy_host_vec_to_device(&idx_arc)
        .sync_on(&rt.stream)
        .expect("indices h2d");
    let mut out = api::zeros::<f32>(&[t, d])
        .sync_on(&rt.stream)
        .expect("alloc out");
    {
        let wt = store.tensor(weight);
        let wv = wt.view(&[v, d]).expect("view weight");
        let iv = idx_tensor.view(&[t]).expect("view indices");
        let _ = kernels::embedding_forward((&mut out).partition([1, d]), &wv, &iv)
            .generics(vec![d.to_string()])
            .sync_on(&rt.stream)
            .expect("embedding_forward kernel");
    }
    store.insert_tensor(out, vec![t, d])
}

/// Returns `dweight` of shape `[V, D]`, scattered from `dout` `[T, D]`
/// via the same `indices` used in the forward pass.
pub fn embedding_backward(
    store: &mut TensorStore,
    dout: TensorId,
    indices: &[i32],
    vocab: usize,
) -> TensorId {
    let dout_shape = store.shape(dout).to_vec();
    assert_eq!(dout_shape.len(), 2, "embedding_bw: dout must be [T, D]");
    let (t, d) = (dout_shape[0], dout_shape[1]);
    assert_eq!(t, indices.len(), "embedding_bw: T mismatch");

    let rt = runtime();
    let idx_arc = Arc::new(indices.to_vec());
    let idx_tensor = api::copy_host_vec_to_device(&idx_arc)
        .sync_on(&rt.stream)
        .expect("indices h2d");
    let dweight = api::zeros::<f32>(&[vocab, d])
        .sync_on(&rt.stream)
        .expect("alloc dweight");
    let dw_ptr = dweight.device_pointer();
    {
        let dt = store.tensor(dout);
        let dv = dt.view(&[t, d]).expect("view dout");
        let iv = idx_tensor.view(&[t]).expect("view indices");
        unsafe {
            let _ = kernels::embedding_backward(dw_ptr, &dv, &iv)
                .grid((t as u32, 1, 1))
                .generics(vec![d.to_string()])
                .sync_on(&rt.stream)
                .expect("embedding_backward kernel");
        }
    }
    store.insert_tensor(dweight, vec![vocab, d])
}
