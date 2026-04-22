//! 2D pooling ops.
//!
//! Avg-pool forward + backward fully wired through the f32 TensorStore.
//!
//! Max-pool forward additionally produces a within-window `argmax` index
//! per output element (`kh*KW + kw`) so the backward pass can scatter
//! gradients back to the right input cell.  Because the f32 TensorStore
//! doesn't currently hold `i32` tensors, the argmax buffer is held as an
//! owned `Tensor<i32>` inside `MaxPoolState` and passed back into the
//! backward op directly — no host round-trip.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Reshape, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};

/// Average-pool 2D forward: `out[n, c, oh, ow] = mean(inp[n, c, oh*KH..oh*KH+KH, ow*KW..ow*KW+KW])`.
///
/// Requires `H % KH == 0` and `W % KW == 0` (matches the CUDA kernel's
/// implicit assumption — out-of-window lanes load zero so a partial bottom
/// or right border would otherwise underweight the average).
pub fn avgpool2d_forward(
    store: &mut TensorStore,
    inp: TensorId,
    kh: usize,
    kw: usize,
) -> TensorId {
    let shape = store.shape(inp).to_vec();
    assert_eq!(shape.len(), 4, "avgpool2d: expected NCHW input");
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    assert_eq!(h % kh, 0, "avgpool2d: H ({h}) not divisible by KH ({kh})");
    assert_eq!(w % kw, 0, "avgpool2d: W ({w}) not divisible by KW ({kw})");
    let oh = h / kh;
    let ow = w / kw;

    let rt = runtime();
    let nc = n * c;
    let mut out_flat = api::zeros::<f32>(&[nc, oh, ow])
        .sync_on(&rt.stream)
        .expect("alloc out");
    {
        let it = store.tensor(inp);
        let iv = it.view(&[nc, h, w]).expect("view inp");
        let inv_kh_kw = 1.0f32 / (kh * kw) as f32;
        let _ = kernels::avgpool2d_forward(
            (&mut out_flat).partition([1, 1, 1]),
            &iv,
            inv_kh_kw,
        )
        .generics(vec![kh.to_string(), kw.to_string()])
        .sync_on(&rt.stream)
        .expect("avgpool2d_forward kernel");
    }
    let out = out_flat.reshape(&[n, c, oh, ow]).expect("reshape out");
    store.insert_tensor(out, vec![n, c, oh, ow])
}

/// Average-pool 2D backward.  Each input element receives `dout[oh, ow] / (KH·KW)`.
pub fn avgpool2d_backward(
    store: &mut TensorStore,
    dout: TensorId,
    kh: usize,
    kw: usize,
) -> TensorId {
    let dout_shape = store.shape(dout).to_vec();
    assert_eq!(dout_shape.len(), 4, "avgpool2d_backward: expected NCHW dout");
    let (n, c, oh, ow) = (
        dout_shape[0],
        dout_shape[1],
        dout_shape[2],
        dout_shape[3],
    );
    let h = oh * kh;
    let w = ow * kw;
    let rt = runtime();
    let nc = n * c;
    let mut dinp_flat = api::zeros::<f32>(&[nc, h, w])
        .sync_on(&rt.stream)
        .expect("alloc dinp");
    {
        let dt = store.tensor(dout);
        let dv = dt.view(&[nc, oh, ow]).expect("view dout");
        let inv_kh_kw = 1.0f32 / (kh * kw) as f32;
        let _ = kernels::avgpool2d_backward(
            (&mut dinp_flat).partition([1, kh, kw]),
            &dv,
            inv_kh_kw,
        )
        .generics(vec![kh.to_string(), kw.to_string()])
        .sync_on(&rt.stream)
        .expect("avgpool2d_backward kernel");
    }
    let dinp = dinp_flat.reshape(&[n, c, h, w]).expect("reshape dinp");
    store.insert_tensor(dinp, vec![n, c, h, w])
}

/// State held between max-pool forward and backward — owns the `[N, C, OH, OW]`
/// argmax buffer in i32 form (each element encodes `kh*KW + kw`, 0..KH*KW).
/// Backward consumes the state and scatters into a fresh `dinp` buffer.
pub struct MaxPoolState {
    pub out: TensorId,
    /// Owned i32 argmax tensor (within-window position).
    argmax: Tensor<i32>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
}

pub fn maxpool2d_forward(
    store: &mut TensorStore,
    inp: TensorId,
    kh: usize,
    kw: usize,
) -> MaxPoolState {
    let shape = store.shape(inp).to_vec();
    assert_eq!(shape.len(), 4, "maxpool2d: expected NCHW input");
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    assert_eq!(h % kh, 0, "maxpool2d: H ({h}) not divisible by KH ({kh})");
    assert_eq!(w % kw, 0, "maxpool2d: W ({w}) not divisible by KW ({kw})");
    let oh = h / kh;
    let ow = w / kw;

    let rt = runtime();
    let nc = n * c;
    let mut out_flat = api::zeros::<f32>(&[nc, oh, ow])
        .sync_on(&rt.stream)
        .expect("alloc out");
    let mut argmax = api::zeros::<i32>(&[nc, oh, ow])
        .sync_on(&rt.stream)
        .expect("alloc argmax");
    {
        let it = store.tensor(inp);
        let iv = it.view(&[nc, h, w]).expect("view inp");
        let _ = kernels::maxpool2d_forward(
            (&mut out_flat).partition([1, 1, 1]),
            (&mut argmax).partition([1, 1, 1]),
            &iv,
        )
        .generics(vec![kh.to_string(), kw.to_string()])
        .sync_on(&rt.stream)
        .expect("maxpool2d_forward kernel");
    }
    let out = out_flat.reshape(&[n, c, oh, ow]).expect("reshape out");
    let out_id = store.insert_tensor(out, vec![n, c, oh, ow]);
    MaxPoolState {
        out: out_id,
        argmax,
        n,
        c,
        h,
        w,
        oh,
        ow,
        kh,
        kw,
    }
}

const SCATTER_BLOCK: usize = 256;

/// Backward: scatter `dout[i]` into `dinp[global_argmax(i)]` via atomic add.
/// We first translate the within-window argmax (`kh*KW + kw`) to a global
/// flat index `n*CHW + c*HW + (oh*KH+kh)*W + (ow*KW+kw)` on the host,
/// upload it as an i32 tensor, and dispatch the scatter kernel.  The
/// host-side translation is a no-op for the kernel arithmetic; doing it on
/// device would require shape constants the maxpool kernel doesn't carry.
pub fn maxpool2d_backward(
    store: &mut TensorStore,
    state: &MaxPoolState,
    dout: TensorId,
) -> TensorId {
    let dout_shape = store.shape(dout).to_vec();
    assert_eq!(
        dout_shape,
        vec![state.n, state.c, state.oh, state.ow],
        "maxpool2d_backward: dout shape must match forward output"
    );
    let rt = runtime();

    // Pull the argmax to host so we can rewrite within-window → global flat.
    let am_host: Vec<i32> = state
        .argmax
        .dup()
        .to_host_vec()
        .sync_on(&rt.stream)
        .expect("argmax → host");
    let mut global = vec![0i32; am_host.len()];
    let chw = (state.c * state.h * state.w) as i32;
    let hw = (state.h * state.w) as i32;
    let w_i = state.w as i32;
    let kw_i = state.kw as i32;
    let kh_i = state.kh as i32;
    let _ = kh_i; // unused; kept for symmetry.
    for n in 0..state.n {
        for c in 0..state.c {
            for oh in 0..state.oh {
                for ow in 0..state.ow {
                    let i = ((n * state.c + c) * state.oh + oh) * state.ow + ow;
                    let am = am_host[i];
                    let kh = am / kw_i;
                    let kw = am % kw_i;
                    let n_i = n as i32;
                    let c_i = c as i32;
                    let oh_i = oh as i32;
                    let ow_i = ow as i32;
                    let g = n_i * chw
                        + c_i * hw
                        + (oh_i * (state.h as i32 / state.oh as i32) + kh) * w_i
                        + (ow_i * (state.w as i32 / state.ow as i32) + kw);
                    global[i] = g;
                }
            }
        }
    }
    let global_arc = std::sync::Arc::new(global);
    let argmax_global = api::copy_host_vec_to_device(&global_arc)
        .sync_on(&rt.stream)
        .expect("argmax_global h2d");

    let dinp = api::zeros::<f32>(&[state.n, state.c, state.h, state.w])
        .sync_on(&rt.stream)
        .expect("alloc dinp");

    let n_total = state.n * state.c * state.oh * state.ow;
    let block = if n_total % SCATTER_BLOCK == 0 {
        SCATTER_BLOCK
    } else {
        // Pick the largest power-of-two ≤ n_total that divides it.
        let mut b = SCATTER_BLOCK;
        while b > 1 && n_total % b != 0 {
            b /= 2;
        }
        b
    };
    let grid = n_total.div_ceil(block) as u32;

    let dinp_ptr = dinp.device_pointer();
    {
        let dt = store.tensor(dout);
        let dv = dt.view(&[n_total]).expect("view dout flat");
        let amv = argmax_global.view(&[n_total]).expect("view argmax flat");
        unsafe {
            let _ = kernels::maxpool2d_backward(dinp_ptr, &dv, &amv)
                .grid((grid, 1, 1))
                .generics(vec![block.to_string()])
                .sync_on(&rt.stream)
                .expect("maxpool2d_backward kernel");
        }
    }

    store.insert_tensor(dinp, vec![state.n, state.c, state.h, state.w])
}
