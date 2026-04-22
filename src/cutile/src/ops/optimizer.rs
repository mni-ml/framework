//! AdamW step.
//!
//! Updates `param`, `m`, `v` in place from the supplied `grad`.  The host
//! computes `bc1 = 1 - β₁^t` and `bc2 = 1 - β₂^t` and passes them as
//! scalars; `t` is the AdamW step counter.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{shape_size, TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
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

#[allow(clippy::too_many_arguments)]
pub fn adamw_step(
    store: &mut TensorStore,
    param: TensorId,
    m: TensorId,
    v: TensorId,
    grad: TensorId,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bc1: f32,
    bc2: f32,
) {
    let shape = store.shape(param).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(m), size, "adamw: m shape mismatch");
    assert_eq!(store.size(v), size, "adamw: v shape mismatch");
    assert_eq!(store.size(grad), size, "adamw: grad shape mismatch");

    let block = pick_block(size);
    let rt = runtime();

    // Dup the grad to an owned tensor so we don't need a live immutable
    // borrow of the store while taking three concurrent mutable borrows.
    let grad_owned = store
        .tensor(grad)
        .dup()
        .sync_on(&rt.stream)
        .expect("dup grad")
        .reshape(&[size])
        .expect("reshape grad");

    // We need &mut to param/m/v simultaneously — split through the
    // underlying Vec so the borrows are disjoint.
    let param_t;
    let m_t;
    let v_t;
    {
        // Sort by index so we can split_at_mut deterministically.
        let mut indices = [(param, 0usize), (m, 1usize), (v, 2usize)];
        indices.sort_by_key(|&(id, _)| id);
        assert!(
            indices[0].0 != indices[1].0 && indices[1].0 != indices[2].0,
            "adamw: param, m, v must be distinct tensor ids"
        );
        let tensors = &mut store.tensors;
        let max_id = indices[2].0;
        assert!(max_id < tensors.len(), "tensor id out of range");
        // Bind the three slots via successive split_at_mut.
        let (lo, hi) = tensors.split_at_mut(indices[1].0);
        let (mid, rest) = hi.split_at_mut(indices[2].0 - indices[1].0);
        let slot_lo = lo[indices[0].0].as_mut().expect("tensor freed");
        let slot_mid = mid[0].as_mut().expect("tensor freed");
        let slot_hi = rest[0].as_mut().expect("tensor freed");
        // Map back to original (param, m, v) order.
        let mut slots: [Option<&mut crate::tensor::CuTileTensor>; 3] = [None, None, None];
        slots[indices[0].1] = Some(slot_lo);
        slots[indices[1].1] = Some(slot_mid);
        slots[indices[2].1] = Some(slot_hi);
        let [Some(p), Some(mm), Some(vv)] = slots else {
            unreachable!("disjoint slots populated above")
        };
        param_t = &mut p.tensor;
        m_t = &mut mm.tensor;
        v_t = &mut vv.tensor;

        let _ = kernels::adamw_step(
            param_t.partition([block]),
            m_t.partition([block]),
            v_t.partition([block]),
            &grad_owned,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bc1,
            bc2,
        )
        .sync_on(&rt.stream)
        .expect("adamw_step kernel");
    }
}
