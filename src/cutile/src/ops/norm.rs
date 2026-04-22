//! LayerNorm forward + backward.
//!
//! Forward calls `layernorm_forward` per row, producing `(out, mean, rstd)`.
//! `mean` and `rstd` are kept alive as `[N]` tensors so the backward pass can
//! recover the normalized activations without re-computing them.
//!
//! Backward runs `layernorm_backward_input`, which emits `dx` plus per-row
//! `dgamma`/`dbeta` partials of shape `[N, C]`.  The ops layer folds those
//! partials along `N` with `sum_along_first` — a two-pass reduction, no
//! atomics — to produce the final `[C]` weight gradients, matching the
//! CUDA version's numerics without `atomicAdd`.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Reshape};
use cutile::tile_kernel::TileKernel;

/// Single row per block — the kernel is `layernorm_forward<C>` with a fixed
/// row tile of 1.  Choose the C candidate at launch to match `x`'s last dim.
pub struct LayerNormOutput {
    pub out: TensorId,
    pub mean: TensorId,
    pub rstd: TensorId,
}

fn flatten_nc(shape: &[usize]) -> (usize, usize) {
    assert!(shape.len() >= 2, "layernorm: input must be rank >= 2");
    let c = *shape.last().unwrap();
    let n: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
    (n, c)
}

pub fn layernorm_forward(
    store: &mut TensorStore,
    x: TensorId,
    gamma: TensorId,
    beta: TensorId,
    eps: f32,
) -> LayerNormOutput {
    let shape = store.shape(x).to_vec();
    let (n, c) = flatten_nc(&shape);
    assert_eq!(store.size(gamma), c, "layernorm: gamma must be [C]");
    assert_eq!(store.size(beta), c, "layernorm: beta must be [C]");
    let rt = runtime();

    let mut out = api::zeros::<f32>(&[n, c]).sync_on(&rt.stream).expect("alloc out");
    let mut mean = api::zeros::<f32>(&[n]).sync_on(&rt.stream).expect("alloc mean");
    let mut rstd = api::zeros::<f32>(&[n]).sync_on(&rt.stream).expect("alloc rstd");
    {
        let xt = store.tensor(x);
        let gt = store.tensor(gamma);
        let bt = store.tensor(beta);
        let xv = xt.view(&[n, c]).expect("view x");
        let gv = gt.view(&[c]).expect("view gamma");
        let bv = bt.view(&[c]).expect("view beta");
        let _ = kernels::layernorm_forward(
            (&mut out).partition([1, c]),
            (&mut mean).partition([1]),
            (&mut rstd).partition([1]),
            &xv,
            &gv,
            &bv,
            eps,
            1.0f32 / c as f32,
        )
        .generics(vec![c.to_string()])
        .sync_on(&rt.stream)
        .expect("layernorm_forward kernel");
    }

    let out_logical = out.reshape(&shape).expect("reshape out");
    let out_id = store.insert_tensor(out_logical, shape);
    let mean_id = store.insert_tensor(mean, vec![n]);
    let rstd_id = store.insert_tensor(rstd, vec![n]);
    LayerNormOutput {
        out: out_id,
        mean: mean_id,
        rstd: rstd_id,
    }
}

/// Returns `(dx, dgamma, dbeta)`.  `mean` and `rstd` are the forward pass's
/// per-row statistics.
pub struct LayerNormBackward {
    pub dx: TensorId,
    pub dgamma: TensorId,
    pub dbeta: TensorId,
}

pub fn layernorm_backward(
    store: &mut TensorStore,
    dy: TensorId,
    x: TensorId,
    mean: TensorId,
    rstd: TensorId,
    gamma: TensorId,
) -> LayerNormBackward {
    let shape = store.shape(x).to_vec();
    let (n, c) = flatten_nc(&shape);
    assert_eq!(store.shape(dy), shape, "layernorm_backward: dy shape mismatch");
    assert_eq!(store.size(mean), n, "layernorm_backward: mean must be [N]");
    assert_eq!(store.size(rstd), n, "layernorm_backward: rstd must be [N]");
    assert_eq!(store.size(gamma), c, "layernorm_backward: gamma must be [C]");
    let rt = runtime();

    let mut dx = api::zeros::<f32>(&[n, c]).sync_on(&rt.stream).expect("alloc dx");
    let mut dgamma_partial = api::zeros::<f32>(&[n, c])
        .sync_on(&rt.stream)
        .expect("alloc dgamma_partial");
    let mut dbeta_partial = api::zeros::<f32>(&[n, c])
        .sync_on(&rt.stream)
        .expect("alloc dbeta_partial");
    {
        let dyt = store.tensor(dy);
        let xt = store.tensor(x);
        let mt = store.tensor(mean);
        let rt_t = store.tensor(rstd);
        let gt = store.tensor(gamma);
        let dyv = dyt.view(&[n, c]).expect("view dy");
        let xv = xt.view(&[n, c]).expect("view x");
        let mv = mt.view(&[n]).expect("view mean");
        let rv = rt_t.view(&[n]).expect("view rstd");
        let gv = gt.view(&[c]).expect("view gamma");
        let _ = kernels::layernorm_backward_input(
            (&mut dx).partition([1, c]),
            (&mut dgamma_partial).partition([1, c]),
            (&mut dbeta_partial).partition([1, c]),
            &dyv,
            &xv,
            &mv,
            &rv,
            &gv,
            1.0f32 / c as f32,
        )
        .generics(vec![c.to_string()])
        .sync_on(&rt.stream)
        .expect("layernorm_backward_input kernel");
    }

    // Reduce per-row partials along N to get [C] weight grads.
    let dg_id = store.insert_tensor(dgamma_partial, vec![n, c]);
    let db_id = store.insert_tensor(dbeta_partial, vec![n, c]);
    let dgamma = reduce_along_first(store, dg_id, n, c);
    let dbeta = reduce_along_first(store, db_id, n, c);
    store.free(dg_id);
    store.free(db_id);

    let dx_logical = dx.reshape(&shape).expect("reshape dx");
    let dx_id = store.insert_tensor(dx_logical, shape);
    LayerNormBackward {
        dx: dx_id,
        dgamma,
        dbeta,
    }
}

const COL_CANDIDATES: [usize; 6] = [64, 32, 16, 8, 4, 1];

fn pick_col_block(c: usize) -> usize {
    for &b in &COL_CANDIDATES {
        if c % b == 0 {
            return b;
        }
    }
    1
}

const ROW_CANDIDATES_2D: [usize; 6] = [32, 16, 8, 4, 2, 1];

fn pick_rows_block(n: usize) -> usize {
    for &b in &ROW_CANDIDATES_2D {
        if n % b == 0 {
            return b;
        }
    }
    1
}

/// Reduce `src : [N, C]` → `[C]` by summing along the N axis.
/// Uses `sum_along_first<ROWS, BN>` tiled so multiple row blocks accumulate
/// into the same `[BN]` output tile via successive launches if N > ROWS.
/// For simplicity we pick ROWS to divide N exactly and launch once;
/// otherwise fall back to a host loop.  N ≤ ROWS is the common batched
/// LayerNorm case (N is batch*seq, typically 128 or 256).
fn reduce_along_first(
    store: &mut TensorStore,
    src: TensorId,
    n: usize,
    c: usize,
) -> TensorId {
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[c]).sync_on(&rt.stream).expect("alloc");
    let rows = pick_rows_block(n);
    let bn = pick_col_block(c);
    if rows == n {
        let st = store.tensor(src);
        let sv = st.view(&[n, c]).expect("view src");
        let _ = kernels::sum_along_first((&mut out).partition([bn]), &sv)
            .generics(vec![rows.to_string(), bn.to_string()])
            .sync_on(&rt.stream)
            .expect("sum_along_first kernel");
    } else {
        // Fall back: host materialize + sum.  Correct for any N but slow;
        // in practice N is small (≤ a few thousand).
        let flat = store.to_host(src);
        let mut acc = vec![0.0f32; c];
        for r in 0..n {
            for j in 0..c {
                acc[j] += flat[r * c + j];
            }
        }
        drop(out);
        return store.from_slice(&acc, &[c]);
    }
    store.insert_tensor(out, vec![c])
}
