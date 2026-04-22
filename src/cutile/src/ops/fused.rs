//! Fused ops: residual+layernorm, bias+gelu (forward + backward).
//!
//! These keep the addition / bias broadcast in registers so the row-reduce
//! and elementwise passes don't make extra GMEM trips.  The backward of
//! `bias_gelu` emits a per-element `dbias_partial`; the ops layer reduces it
//! along N with a separate `sum_along_first` launch (no atomics).

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Reshape};
use cutile::tile_kernel::TileKernel;

const ROW_CANDIDATES: [usize; 6] = [32, 16, 8, 4, 2, 1];
const COL_CANDIDATES: [usize; 6] = [64, 32, 16, 8, 4, 1];

fn pick_row_block(n: usize) -> usize {
    for &b in &ROW_CANDIDATES {
        if n % b == 0 {
            return b;
        }
    }
    1
}

fn pick_col_block(c: usize) -> usize {
    for &b in &COL_CANDIDATES {
        if c % b == 0 {
            return b;
        }
    }
    1
}

fn flatten_nc(shape: &[usize]) -> (usize, usize) {
    assert!(shape.len() >= 2, "fused 2D op needs rank >= 2 input");
    let c = *shape.last().unwrap();
    let n: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
    (n, c)
}

pub struct ResidualLayerNormOutput {
    pub out: TensorId,
    pub residual: TensorId,
    pub mean: TensorId,
    pub rstd: TensorId,
}

pub fn residual_layernorm_forward(
    store: &mut TensorStore,
    x: TensorId,
    residual: TensorId,
    gamma: TensorId,
    beta: TensorId,
    eps: f32,
) -> ResidualLayerNormOutput {
    let shape = store.shape(x).to_vec();
    let (n, c) = flatten_nc(&shape);
    assert_eq!(store.shape(residual), shape, "residual shape mismatch");
    assert_eq!(store.size(gamma), c, "gamma must be [C]");
    assert_eq!(store.size(beta), c, "beta must be [C]");
    let rt = runtime();

    let mut out = api::zeros::<f32>(&[n, c]).sync_on(&rt.stream).expect("alloc out");
    let mut r = api::zeros::<f32>(&[n, c]).sync_on(&rt.stream).expect("alloc r");
    let mut mean = api::zeros::<f32>(&[n]).sync_on(&rt.stream).expect("alloc mean");
    let mut rstd = api::zeros::<f32>(&[n]).sync_on(&rt.stream).expect("alloc rstd");
    {
        let xt = store.tensor(x);
        let res_t = store.tensor(residual);
        let gt = store.tensor(gamma);
        let bt = store.tensor(beta);
        let xv = xt.view(&[n, c]).expect("view x");
        let rv = res_t.view(&[n, c]).expect("view residual");
        let gv = gt.view(&[c]).expect("view gamma");
        let bv = bt.view(&[c]).expect("view beta");
        let _ = kernels::residual_layernorm_forward(
            (&mut out).partition([1, c]),
            (&mut r).partition([1, c]),
            (&mut mean).partition([1]),
            (&mut rstd).partition([1]),
            &xv,
            &rv,
            &gv,
            &bv,
            eps,
            1.0f32 / c as f32,
        )
        .generics(vec![c.to_string()])
        .sync_on(&rt.stream)
        .expect("residual_layernorm_forward kernel");
    }
    let out_logical = out.reshape(&shape).expect("reshape out");
    let r_logical = r.reshape(&shape).expect("reshape r");
    let out_id = store.insert_tensor(out_logical, shape.clone());
    let res_id = store.insert_tensor(r_logical, shape);
    let mean_id = store.insert_tensor(mean, vec![n]);
    let rstd_id = store.insert_tensor(rstd, vec![n]);
    ResidualLayerNormOutput {
        out: out_id,
        residual: res_id,
        mean: mean_id,
        rstd: rstd_id,
    }
}

/// Fused `out[i, j] = gelu(x[i, j] + bias[j])`.
pub fn bias_gelu_forward(store: &mut TensorStore, x: TensorId, bias: TensorId) -> TensorId {
    let shape = store.shape(x).to_vec();
    let (n, c) = flatten_nc(&shape);
    assert_eq!(store.size(bias), c, "bias_gelu: bias must be [C]");
    let bn = pick_row_block(n);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[n, c]).sync_on(&rt.stream).expect("alloc");
    {
        let xt = store.tensor(x);
        let bt = store.tensor(bias);
        let xv = xt.view(&[n, c]).expect("view x");
        let bv = bt.view(&[c]).expect("view bias");
        let _ = kernels::bias_gelu_forward((&mut out).partition([bn, c]), &xv, &bv)
            .generics(vec![bn.to_string(), c.to_string()])
            .sync_on(&rt.stream)
            .expect("bias_gelu_forward kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

pub struct BiasGeluBackward {
    pub dx: TensorId,
    pub dbias: TensorId,
}

/// Backward of `bias_gelu_forward`.  `dx` and per-element `dbias_partial`
/// share the same compiled value; the partial is reduced along N to give
/// the final `[C]` `dbias`.
pub fn bias_gelu_backward(
    store: &mut TensorStore,
    grad: TensorId,
    x: TensorId,
    bias: TensorId,
) -> BiasGeluBackward {
    let shape = store.shape(x).to_vec();
    let (n, c) = flatten_nc(&shape);
    assert_eq!(store.shape(grad), shape, "bias_gelu_backward: grad shape mismatch");
    assert_eq!(store.size(bias), c, "bias_gelu_backward: bias must be [C]");
    let bn = pick_row_block(n);
    let rt = runtime();
    let mut dx = api::zeros::<f32>(&[n, c]).sync_on(&rt.stream).expect("alloc dx");
    let mut dbias_partial = api::zeros::<f32>(&[n, c])
        .sync_on(&rt.stream)
        .expect("alloc dbias_partial");
    {
        let gt = store.tensor(grad);
        let xt = store.tensor(x);
        let bt = store.tensor(bias);
        let gv = gt.view(&[n, c]).expect("view grad");
        let xv = xt.view(&[n, c]).expect("view x");
        let bv = bt.view(&[c]).expect("view bias");
        let _ = kernels::bias_gelu_backward(
            (&mut dx).partition([bn, c]),
            (&mut dbias_partial).partition([bn, c]),
            &gv,
            &xv,
            &bv,
        )
        .generics(vec![bn.to_string(), c.to_string()])
        .sync_on(&rt.stream)
        .expect("bias_gelu_backward kernel");
    }

    let dbias_partial_id = store.insert_tensor(dbias_partial, vec![n, c]);
    let dbias = reduce_along_first(store, dbias_partial_id, n, c);
    store.free(dbias_partial_id);

    let dx_logical = dx.reshape(&shape).expect("reshape dx");
    let dx_id = store.insert_tensor(dx_logical, shape);
    BiasGeluBackward {
        dx: dx_id,
        dbias,
    }
}

fn reduce_along_first(store: &mut TensorStore, src: TensorId, n: usize, c: usize) -> TensorId {
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[c]).sync_on(&rt.stream).expect("alloc");
    let bn = pick_col_block(c);
    let rows_candidates = [32usize, 16, 8, 4, 2, 1];
    let mut rows = 1usize;
    for &b in &rows_candidates {
        if n % b == 0 {
            rows = b;
            break;
        }
    }
    if rows == n {
        let st = store.tensor(src);
        let sv = st.view(&[n, c]).expect("view src");
        let _ = kernels::sum_along_first((&mut out).partition([bn]), &sv)
            .generics(vec![rows.to_string(), bn.to_string()])
            .sync_on(&rt.stream)
            .expect("sum_along_first kernel");
        store.insert_tensor(out, vec![c])
    } else {
        let flat = store.to_host(src);
        let mut acc = vec![0.0f32; c];
        for r in 0..n {
            for j in 0..c {
                acc[j] += flat[r * c + j];
            }
        }
        drop(out);
        store.from_slice(&acc, &[c])
    }
}
