//! Cross-entropy forward + backward.
//!
//! `V` (vocab) and `BM` (rows per block) are const generics on the kernel.
//! `V` is baked from the runtime logits shape (callers must use a fixed
//! vocab per model — same as embedding `D`); `BM` picks the largest divisor
//! of `N` from `[16, 8, 4, 2, 1]`.
//!
//! Forward writes both per-row loss and the softmax probabilities — the
//! latter is fed straight back into `cross_entropy_backward`, so the
//! backward kernel doesn't recompute softmax.
//!
//! Targets are taken as `&Tensor<i32>` (e.g. from `data::sample_batch`) so
//! the i32-typed buffer doesn't need to round-trip through the f32 store.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::TileKernel;

const BM_CANDIDATES: [usize; 5] = [16, 8, 4, 2, 1];

fn pick_bm(n: usize) -> usize {
    for &b in &BM_CANDIDATES {
        if n % b == 0 {
            return b;
        }
    }
    1
}

pub struct CrossEntropyForward {
    /// `[N]` per-row negative-log-likelihood losses.
    pub losses: TensorId,
    /// `[N, V]` softmax probabilities (consumed by `cross_entropy_backward`).
    pub softmax: TensorId,
}

/// `losses[r] = -log(softmax(logits[r])[targets[r]])`,
/// `softmax[r, :] = softmax(logits[r])`.
pub fn cross_entropy_forward(
    store: &mut TensorStore,
    logits: TensorId,
    targets: &Tensor<i32>,
) -> CrossEntropyForward {
    let logits_shape = store.shape(logits).to_vec();
    assert_eq!(logits_shape.len(), 2, "cross_entropy: logits must be [N, V]");
    let (n, v) = (logits_shape[0], logits_shape[1]);
    assert_eq!(targets.size(), n, "cross_entropy: targets length mismatch");
    let bm = pick_bm(n);

    let rt = runtime();
    let mut losses = api::zeros::<f32>(&[n])
        .sync_on(&rt.stream)
        .expect("alloc losses");
    let mut softmax = api::zeros::<f32>(&[n, v])
        .sync_on(&rt.stream)
        .expect("alloc softmax");
    {
        let lt = store.tensor(logits);
        let lv = lt.view(&[n, v]).expect("view logits");
        let tv = targets.view(&[n]).expect("view targets");
        let _ = kernels::cross_entropy_forward(
            (&mut losses).partition([bm]),
            (&mut softmax).partition([bm, v]),
            &lv,
            &tv,
        )
        .generics(vec![bm.to_string(), v.to_string()])
        .sync_on(&rt.stream)
        .expect("cross_entropy_forward kernel");
    }
    let losses_id = store.insert_tensor(losses, vec![n]);
    let softmax_id = store.insert_tensor(softmax, vec![n, v]);
    CrossEntropyForward {
        losses: losses_id,
        softmax: softmax_id,
    }
}

/// `dlogits[r, j] = (softmax[r, j] - [j == targets[r]]) * grad_scale`.
/// `softmax` must be the buffer returned by `cross_entropy_forward`.
pub fn cross_entropy_backward(
    store: &mut TensorStore,
    softmax: TensorId,
    targets: &Tensor<i32>,
    grad_scale: f32,
) -> TensorId {
    let sm_shape = store.shape(softmax).to_vec();
    assert_eq!(sm_shape.len(), 2, "cross_entropy_bw: softmax must be [N, V]");
    let (n, v) = (sm_shape[0], sm_shape[1]);
    assert_eq!(targets.size(), n, "cross_entropy_bw: targets length mismatch");
    let bm = pick_bm(n);

    let rt = runtime();
    let mut dlogits = api::zeros::<f32>(&[n, v])
        .sync_on(&rt.stream)
        .expect("alloc dlogits");
    {
        let st = store.tensor(softmax);
        let sv = st.view(&[n, v]).expect("view softmax");
        let tv = targets.view(&[n]).expect("view targets");
        let _ = kernels::cross_entropy_backward(
            (&mut dlogits).partition([bm, v]),
            &sv,
            &tv,
            grad_scale,
        )
        .generics(vec![bm.to_string(), v.to_string()])
        .sync_on(&rt.stream)
        .expect("cross_entropy_backward kernel");
    }
    store.insert_tensor(dlogits, vec![n, v])
}
