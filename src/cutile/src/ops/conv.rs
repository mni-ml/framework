//! Conv1D / Conv2D ops — port of `native/kernels/conv.cu` callers.
//!
//! Each forward pass allocates a fresh output tensor and dispatches the
//! corresponding fused cuTile kernel.  Both backward passes (`backward_input`
//! and `backward_weight`) return new tensors as well — the CUDA ops layer
//! does the same and we don't try to fuse `dinp + dweight` into one launch.
//!
//! `C_in`, `C_out`, and `K` (or `KH`, `KW`) are baked into the kernel as
//! const generics.  `stride` and `padding` are runtime ints, so a model
//! that varies the spatial kernel keeps each (CI, K) shape in PTX cache
//! exactly once.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Reshape};
use cutile::tile_kernel::TileKernel;

/// `BL` for `conv1d_backward_weight` — inner reduction tile along `L_out`.
const BL_CONV1D: usize = 32;
/// `BW` for `conv2d_backward_weight` — inner reduction tile along `W_out`.
const BW_CONV2D: usize = 32;

fn out_dim(in_dim: usize, k: usize, stride: usize, padding: usize) -> usize {
    (in_dim + 2 * padding).saturating_sub(k) / stride + 1
}

/// cuTile tile shapes must have power-of-two dims.  Conv reduction loops
/// run over `CI`, `K`, `KH`, `KW` etc., which in real models are commonly
/// 3 / 5 / 7 — we round up to the next pow2 and let the kernel mask the
/// tail.  `next_pow2(0)` is forced to 1 to keep cuTile happy.
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n.next_power_of_two()
    }
}

// ---------------------------------------------------------------------------
// Conv1D
// ---------------------------------------------------------------------------

/// `out[N, C_out, L_out] = Σ(ci, k) inp[N, ci, l*stride - pad + k] · weight[co, ci, k]`.
pub fn conv1d_forward(
    store: &mut TensorStore,
    inp: TensorId,
    weight: TensorId,
    stride: usize,
    padding: usize,
) -> TensorId {
    let inp_shape = store.shape(inp).to_vec();
    let w_shape = store.shape(weight).to_vec();
    assert_eq!(inp_shape.len(), 3, "conv1d: expected NCL input");
    assert_eq!(w_shape.len(), 3, "conv1d: expected [C_out, C_in, K] weight");
    let (n, c_in, l_in) = (inp_shape[0], inp_shape[1], inp_shape[2]);
    let (c_out, w_ci, k) = (w_shape[0], w_shape[1], w_shape[2]);
    assert_eq!(w_ci, c_in, "conv1d: weight C_in mismatch");
    let l_out = out_dim(l_in, k, stride, padding);

    let rt = runtime();
    let mut out = api::zeros::<f32>(&[n, c_out, l_out])
        .sync_on(&rt.stream)
        .expect("alloc out");
    {
        let it = store.tensor(inp);
        let wt = store.tensor(weight);
        let inp_ptr = it.device_pointer();
        let weight_ptr = wt.device_pointer();
        unsafe {
            let ci_p = next_pow2(c_in);
            let k_p = next_pow2(k);
            let _ = kernels::conv1d_forward(
                (&mut out).partition([1, 1, 1]),
                inp_ptr,
                weight_ptr,
                l_in as i32,
                stride as i32,
                padding as i32,
            )
            .generics(vec![
                c_in.to_string(),
                k.to_string(),
                ci_p.to_string(),
                k_p.to_string(),
            ])
            .sync_on(&rt.stream)
            .expect("conv1d_forward kernel");
        }
    }
    store.insert_tensor(out, vec![n, c_out, l_out])
}

/// Returns `dinp` of shape `[N, C_in, L_in]`, given `dout` `[N, C_out, L_out]`
/// and the original `weight` `[C_out, C_in, K]` (needed to re-compute the
/// strided gather).
pub fn conv1d_backward_input(
    store: &mut TensorStore,
    dout: TensorId,
    weight: TensorId,
    l_in: usize,
    stride: usize,
    padding: usize,
) -> TensorId {
    let dout_shape = store.shape(dout).to_vec();
    let w_shape = store.shape(weight).to_vec();
    assert_eq!(dout_shape.len(), 3, "conv1d_bwi: expected NCL dout");
    assert_eq!(w_shape.len(), 3, "conv1d_bwi: expected [C_out, C_in, K] weight");
    let (n, c_out, l_out) = (dout_shape[0], dout_shape[1], dout_shape[2]);
    let (w_co, c_in, k) = (w_shape[0], w_shape[1], w_shape[2]);
    assert_eq!(w_co, c_out, "conv1d_bwi: weight C_out mismatch");

    let rt = runtime();
    let mut dinp = api::zeros::<f32>(&[n, c_in, l_in])
        .sync_on(&rt.stream)
        .expect("alloc dinp");
    {
        let dt = store.tensor(dout);
        let wt = store.tensor(weight);
        let dout_ptr = dt.device_pointer();
        let weight_ptr = wt.device_pointer();
        unsafe {
            let _ = kernels::conv1d_backward_input(
                (&mut dinp).partition([1, 1, 1]),
                dout_ptr,
                weight_ptr,
                c_in as i32,
                l_out as i32,
                stride as i32,
                padding as i32,
            )
            .generics(vec![c_out.to_string(), k.to_string()])
            .sync_on(&rt.stream)
            .expect("conv1d_backward_input kernel");
        }
    }
    store.insert_tensor(dinp, vec![n, c_in, l_in])
}

/// Returns `dweight` of shape `[C_out, C_in, K]`.
pub fn conv1d_backward_weight(
    store: &mut TensorStore,
    dout: TensorId,
    inp: TensorId,
    k: usize,
    stride: usize,
    padding: usize,
) -> TensorId {
    let dout_shape = store.shape(dout).to_vec();
    let inp_shape = store.shape(inp).to_vec();
    assert_eq!(dout_shape.len(), 3, "conv1d_bww: expected NCL dout");
    assert_eq!(inp_shape.len(), 3, "conv1d_bww: expected NCL inp");
    let (n, c_out, l_out) = (dout_shape[0], dout_shape[1], dout_shape[2]);
    let (n2, c_in, l_in) = (inp_shape[0], inp_shape[1], inp_shape[2]);
    assert_eq!(n, n2, "conv1d_bww: N mismatch");

    let rt = runtime();
    let mut dweight = api::zeros::<f32>(&[c_out, c_in, k])
        .sync_on(&rt.stream)
        .expect("alloc dweight");
    {
        let dt = store.tensor(dout);
        let it = store.tensor(inp);
        let dout_ptr = dt.device_pointer();
        let inp_ptr = it.device_pointer();
        unsafe {
            let _ = kernels::conv1d_backward_weight(
                (&mut dweight).partition([1, 1, 1]),
                dout_ptr,
                inp_ptr,
                n as i32,
                c_in as i32,
                c_out as i32,
                l_in as i32,
                l_out as i32,
                stride as i32,
                padding as i32,
            )
            .generics(vec![BL_CONV1D.to_string()])
            .sync_on(&rt.stream)
            .expect("conv1d_backward_weight kernel");
        }
    }
    store.insert_tensor(dweight, vec![c_out, c_in, k])
}

// ---------------------------------------------------------------------------
// Conv2D
// ---------------------------------------------------------------------------

/// `out[N, C_out, H_out, W_out]`.  Grid is flattened as `(N*C_out, H_out, W_out)`
/// inside the kernel — the partition tile shape `[1,1,1,1]` over a 4D
/// output yields exactly that grid.
pub fn conv2d_forward(
    store: &mut TensorStore,
    inp: TensorId,
    weight: TensorId,
    stride: usize,
    padding: usize,
) -> TensorId {
    let inp_shape = store.shape(inp).to_vec();
    let w_shape = store.shape(weight).to_vec();
    assert_eq!(inp_shape.len(), 4, "conv2d: expected NCHW input");
    assert_eq!(w_shape.len(), 4, "conv2d: expected [C_out, C_in, KH, KW] weight");
    let (n, c_in, h_in, w_in) = (inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]);
    let (c_out, w_ci, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
    assert_eq!(w_ci, c_in, "conv2d: weight C_in mismatch");
    let h_out = out_dim(h_in, kh, stride, padding);
    let w_out = out_dim(w_in, kw, stride, padding);

    let rt = runtime();
    let nco = n * c_out;
    let mut out_flat = api::zeros::<f32>(&[nco, h_out, w_out])
        .sync_on(&rt.stream)
        .expect("alloc out");
    {
        let it = store.tensor(inp);
        let wt = store.tensor(weight);
        let inp_ptr = it.device_pointer();
        let weight_ptr = wt.device_pointer();
        unsafe {
            let ci_p = next_pow2(c_in);
            let kh_p = next_pow2(kh);
            let kw_p = next_pow2(kw);
            let _ = kernels::conv2d_forward(
                (&mut out_flat).partition([1, 1, 1]),
                inp_ptr,
                weight_ptr,
                c_out as i32,
                h_in as i32,
                w_in as i32,
                stride as i32,
                padding as i32,
            )
            .generics(vec![
                c_in.to_string(),
                kh.to_string(),
                kw.to_string(),
                ci_p.to_string(),
                kh_p.to_string(),
                kw_p.to_string(),
            ])
            .sync_on(&rt.stream)
            .expect("conv2d_forward kernel");
        }
    }
    let out = out_flat
        .reshape(&[n, c_out, h_out, w_out])
        .expect("reshape out");
    store.insert_tensor(out, vec![n, c_out, h_out, w_out])
}

/// Returns `dinp` of shape `[N, C_in, H, W]`.  Caller supplies `(h_in, w_in)`
/// so we don't have to invert the convolution arithmetic.
pub fn conv2d_backward_input(
    store: &mut TensorStore,
    dout: TensorId,
    weight: TensorId,
    h_in: usize,
    w_in: usize,
    stride: usize,
    padding: usize,
) -> TensorId {
    let dout_shape = store.shape(dout).to_vec();
    let w_shape = store.shape(weight).to_vec();
    assert_eq!(dout_shape.len(), 4, "conv2d_bwi: expected NCHW dout");
    assert_eq!(w_shape.len(), 4, "conv2d_bwi: expected [C_out, C_in, KH, KW] weight");
    let (n, c_out, h_out, w_out) = (
        dout_shape[0],
        dout_shape[1],
        dout_shape[2],
        dout_shape[3],
    );
    let (w_co, c_in, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
    assert_eq!(w_co, c_out, "conv2d_bwi: weight C_out mismatch");

    let rt = runtime();
    let nci = n * c_in;
    let mut dinp_flat = api::zeros::<f32>(&[nci, h_in, w_in])
        .sync_on(&rt.stream)
        .expect("alloc dinp");
    {
        let dt = store.tensor(dout);
        let wt = store.tensor(weight);
        let dout_ptr = dt.device_pointer();
        let weight_ptr = wt.device_pointer();
        unsafe {
            let _ = kernels::conv2d_backward_input(
                (&mut dinp_flat).partition([1, 1, 1]),
                dout_ptr,
                weight_ptr,
                c_in as i32,
                h_out as i32,
                w_out as i32,
                stride as i32,
                padding as i32,
            )
            .generics(vec![c_out.to_string(), kh.to_string(), kw.to_string()])
            .sync_on(&rt.stream)
            .expect("conv2d_backward_input kernel");
        }
    }
    let dinp = dinp_flat
        .reshape(&[n, c_in, h_in, w_in])
        .expect("reshape dinp");
    store.insert_tensor(dinp, vec![n, c_in, h_in, w_in])
}

/// Returns `dweight` of shape `[C_out, C_in, KH, KW]`.
pub fn conv2d_backward_weight(
    store: &mut TensorStore,
    dout: TensorId,
    inp: TensorId,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> TensorId {
    let dout_shape = store.shape(dout).to_vec();
    let inp_shape = store.shape(inp).to_vec();
    assert_eq!(dout_shape.len(), 4, "conv2d_bww: expected NCHW dout");
    assert_eq!(inp_shape.len(), 4, "conv2d_bww: expected NCHW inp");
    let (n, c_out, h_out, w_out) = (
        dout_shape[0],
        dout_shape[1],
        dout_shape[2],
        dout_shape[3],
    );
    let (n2, c_in, h_in, w_in) = (inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]);
    assert_eq!(n, n2, "conv2d_bww: N mismatch");

    let rt = runtime();
    let coci = c_out * c_in;
    let mut dweight_flat = api::zeros::<f32>(&[coci, kh, kw])
        .sync_on(&rt.stream)
        .expect("alloc dweight");
    {
        let dt = store.tensor(dout);
        let it = store.tensor(inp);
        let dout_ptr = dt.device_pointer();
        let inp_ptr = it.device_pointer();
        unsafe {
            let _ = kernels::conv2d_backward_weight(
                (&mut dweight_flat).partition([1, 1, 1]),
                dout_ptr,
                inp_ptr,
                n as i32,
                c_in as i32,
                c_out as i32,
                h_in as i32,
                w_in as i32,
                h_out as i32,
                w_out as i32,
                stride as i32,
                padding as i32,
            )
            .generics(vec![BW_CONV2D.to_string()])
            .sync_on(&rt.stream)
            .expect("conv2d_backward_weight kernel");
        }
    }
    let dweight = dweight_flat
        .reshape(&[c_out, c_in, kh, kw])
        .expect("reshape dweight");
    store.insert_tensor(dweight, vec![c_out, c_in, kh, kw])
}
