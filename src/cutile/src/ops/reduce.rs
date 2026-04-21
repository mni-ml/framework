//! Reductions — implemented on the host to keep the backend focused on the
//! tiled compute kernels (elementwise + GEMM).  In real usage these would be
//! upgraded to cuTile reduce kernels (see `cutile-examples/softmax.rs` for a
//! reference using `reduce_sum(tile, dim)`), but doing so requires a
//! per-rank family of kernels which is out of scope for this first
//! demonstration backend.

use crate::tensor::{TensorId, TensorStore};

pub fn sum_all(store: &mut TensorStore, a: TensorId) -> TensorId {
    let data = store.to_host(a);
    let total: f32 = data.iter().copied().sum();
    store.from_slice(&[total], &[1])
}

pub fn mean_all(store: &mut TensorStore, a: TensorId) -> TensorId {
    let data = store.to_host(a);
    let n = data.len().max(1) as f32;
    let total: f32 = data.iter().copied().sum();
    store.from_slice(&[total / n], &[1])
}
