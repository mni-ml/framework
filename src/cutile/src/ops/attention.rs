//! FlashAttention-2 forward + backward — wrapper around
//! `kernels::flash_attention_{forward,backward}`.
//!
//! All tensors live as `[BH, S, D]` (with `BH = batch * heads` already
//! collapsed by the caller).  `D` (head dim) is a const generic, so the
//! caller must specify a head dim that the kernel was compiled for —
//! we cap that to a small set of common dims (32, 64, 96, 128).
//!
//! `BM` (query tile) and `BN` (KV tile) are picked here based on `S`:
//! both default to 64 if `S` divides cleanly, otherwise we fall back to
//! the largest tile that divides.  The kernel itself handles tail
//! masking on the inner KV loop, but we keep `BM` a divisor of `S` so
//! every grid block has a full row tile.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::PartitionMut;
use cutile::tile_kernel::TileKernel;

const BM_CANDIDATES: [usize; 4] = [64, 32, 16, 8];

fn pick_bm(s: usize) -> usize {
    for &bm in &BM_CANDIDATES {
        if s % bm == 0 {
            return bm;
        }
    }
    1
}

/// Pick `BN` (key/value tile) that divides `s` so the partition load doesn't
/// over-read the K/V tensor.  Without this, `S=16, BN=64` would make the kernel
/// load 64 rows into the tile when only 16 are valid — the OOB lanes get
/// garbage that softmax then folds into `l_i`, throwing off the result.
fn pick_bn(s: usize) -> usize {
    pick_bm(s)
}

fn assert_supported_d(d: usize) {
    assert!(
        matches!(d, 32 | 64 | 96 | 128),
        "flash_attention: head dim D={d} not supported (expected 32, 64, 96, or 128)"
    );
}

pub struct FlashAttentionForward {
    pub out: TensorId,
    /// `[BH, S]` log-sum-exp from the forward pass — required by backward.
    pub lse: TensorId,
}

/// Forward pass.  `causal == true` applies a strict-upper-triangular mask
/// (`col > row → -inf` before softmax), matching the CUDA `col < row + 1`
/// semantics.
pub fn flash_attention_forward(
    store: &mut TensorStore,
    q: TensorId,
    k: TensorId,
    v: TensorId,
    scale: f32,
    causal: bool,
) -> FlashAttentionForward {
    let q_shape = store.shape(q).to_vec();
    let k_shape = store.shape(k).to_vec();
    let v_shape = store.shape(v).to_vec();
    assert_eq!(q_shape.len(), 3, "flash_attn: expected [BH, S, D] q");
    assert_eq!(q_shape, k_shape, "flash_attn: q/k shape mismatch");
    assert_eq!(q_shape, v_shape, "flash_attn: q/v shape mismatch");
    let (bh, s, d) = (q_shape[0], q_shape[1], q_shape[2]);
    assert_supported_d(d);
    let bm = pick_bm(s);
    let bn = pick_bn(s);

    let rt = runtime();
    let mut out = api::zeros::<f32>(&[bh, s, d])
        .sync_on(&rt.stream)
        .expect("alloc out");
    let mut lse = api::zeros::<f32>(&[bh, s])
        .sync_on(&rt.stream)
        .expect("alloc lse");
    {
        let qt = store.tensor(q);
        let kt = store.tensor(k);
        let vt = store.tensor(v);
        let qv = qt.view(&[bh, s, d]).expect("view q");
        let kv = kt.view(&[bh, s, d]).expect("view k");
        let vv = vt.view(&[bh, s, d]).expect("view v");
        let _ = kernels::flash_attention_forward(
            (&mut out).partition([1, bm, d]),
            (&mut lse).partition([1, bm]),
            &qv,
            &kv,
            &vv,
            scale,
            if causal { 1 } else { 0 },
        )
        .generics(vec![bm.to_string(), bn.to_string(), d.to_string()])
        .sync_on(&rt.stream)
        .expect("flash_attention_forward kernel");
    }
    let out_id = store.insert_tensor(out, vec![bh, s, d]);
    let lse_id = store.insert_tensor(lse, vec![bh, s]);
    FlashAttentionForward {
        out: out_id,
        lse: lse_id,
    }
}

pub struct FlashAttentionBackward {
    pub dq: TensorId,
    pub dk: TensorId,
    pub dv: TensorId,
}

/// Backward pass — `dq` is computed directly per-row-tile, `dk` and `dv`
/// are zero-initialized and atomically accumulated across query tiles.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward(
    store: &mut TensorStore,
    dout: TensorId,
    q: TensorId,
    k: TensorId,
    v: TensorId,
    out: TensorId,
    lse: TensorId,
    scale: f32,
    causal: bool,
) -> FlashAttentionBackward {
    let q_shape = store.shape(q).to_vec();
    assert_eq!(q_shape.len(), 3, "flash_attn_bw: expected [BH, S, D] q");
    let (bh, s, d) = (q_shape[0], q_shape[1], q_shape[2]);
    assert_supported_d(d);
    assert_eq!(store.shape(k), &[bh, s, d], "flash_attn_bw: k shape mismatch");
    assert_eq!(store.shape(v), &[bh, s, d], "flash_attn_bw: v shape mismatch");
    assert_eq!(
        store.shape(out),
        &[bh, s, d],
        "flash_attn_bw: out shape mismatch"
    );
    assert_eq!(
        store.shape(dout),
        &[bh, s, d],
        "flash_attn_bw: dout shape mismatch"
    );
    assert_eq!(store.shape(lse), &[bh, s], "flash_attn_bw: lse shape mismatch");

    let bm = pick_bm(s);
    let bn = pick_bn(s);
    let rt = runtime();

    let mut dq = api::zeros::<f32>(&[bh, s, d])
        .sync_on(&rt.stream)
        .expect("alloc dq");
    let dk = api::zeros::<f32>(&[bh, s, d])
        .sync_on(&rt.stream)
        .expect("alloc dk");
    let dv = api::zeros::<f32>(&[bh, s, d])
        .sync_on(&rt.stream)
        .expect("alloc dv");

    let dk_ptr = dk.device_pointer();
    let dv_ptr = dv.device_pointer();
    {
        let dt = store.tensor(dout);
        let qt = store.tensor(q);
        let kt = store.tensor(k);
        let vt = store.tensor(v);
        let ot = store.tensor(out);
        let lt = store.tensor(lse);
        let dv_view = dt.view(&[bh, s, d]).expect("view dout");
        let qv = qt.view(&[bh, s, d]).expect("view q");
        let kv = kt.view(&[bh, s, d]).expect("view k");
        let vv = vt.view(&[bh, s, d]).expect("view v");
        let ov = ot.view(&[bh, s, d]).expect("view out");
        let lv = lt.view(&[bh, s]).expect("view lse");
        unsafe {
            let _ = kernels::flash_attention_backward(
                (&mut dq).partition([1, bm, d]),
                dk_ptr,
                dv_ptr,
                &dv_view,
                &qv,
                &kv,
                &vv,
                &ov,
                &lv,
                scale,
                if causal { 1 } else { 0 },
            )
            .generics(vec![bm.to_string(), bn.to_string(), d.to_string()])
            .sync_on(&rt.stream)
            .expect("flash_attention_backward kernel");
        }
    }
    let dq_id = store.insert_tensor(dq, vec![bh, s, d]);
    let dk_id = store.insert_tensor(dk, vec![bh, s, d]);
    let dv_id = store.insert_tensor(dv, vec![bh, s, d]);
    FlashAttentionBackward {
        dq: dq_id,
        dk: dk_id,
        dv: dv_id,
    }
}
