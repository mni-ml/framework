//! Row-wise symmetric i8 quantization for the KV cache.
//!
//! Values are quantized to i8 with one f32 scale per row, computed as
//! `max(|row|) / 127` (floored at 1e-8).  Quantize and dequantize are
//! exposed as separate ops; callers typically quantize on cache write
//! and dequantize on cache read.
//!
//! The i8 buffer doesn't fit in the f32 `TensorStore`, so it lives in
//! `QuantizedRows` as an owned `Tensor<i8>` alongside the per-row
//! `Tensor<f32>` scale buffer.

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

/// Owned i8 quantized buffer + per-row f32 scales.
pub struct QuantizedRows {
    pub data: Tensor<i8>,
    pub scales: Tensor<f32>,
    pub n: usize,
    pub d: usize,
}

/// Quantize an `[N, D]` f32 tensor row-wise to i8.  `D` is baked as a
/// const generic so this only needs to be PTX-compiled once per head dim.
pub fn quantize_rows(store: &TensorStore, input: TensorId) -> QuantizedRows {
    let shape = store.shape(input).to_vec();
    assert_eq!(shape.len(), 2, "quantize_rows: input must be [N, D]");
    let (n, d) = (shape[0], shape[1]);
    let bm = pick_bm(n);

    let rt = runtime();
    let mut scales = api::zeros::<f32>(&[n])
        .sync_on(&rt.stream)
        .expect("alloc scales");
    let mut data = api::zeros::<i8>(&[n, d])
        .sync_on(&rt.stream)
        .expect("alloc qdata");
    {
        let it = store.tensor(input);
        let iv = it.view(&[n, d]).expect("view input");
        // compute scales: 1 row per block, [1] tile for scale.
        let _ = kernels::compute_rowwise_scale((&mut scales).partition([1]), &iv)
            .generics(vec![d.to_string()])
            .sync_on(&rt.stream)
            .expect("compute_rowwise_scale kernel");
        // quantize: BM rows per block.
        let sv = scales.view(&[n]).expect("view scales");
        let _ = kernels::quantize_rowwise_i8((&mut data).partition([bm, d]), &iv, &sv)
            .generics(vec![bm.to_string(), d.to_string()])
            .sync_on(&rt.stream)
            .expect("quantize_rowwise_i8 kernel");
    }
    QuantizedRows { data, scales, n, d }
}

/// Dequantize back to f32; result is registered in the f32 `TensorStore`.
pub fn dequantize_rows(store: &mut TensorStore, q: &QuantizedRows) -> TensorId {
    let bm = pick_bm(q.n);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[q.n, q.d])
        .sync_on(&rt.stream)
        .expect("alloc out");
    {
        let iv = q.data.view(&[q.n, q.d]).expect("view qdata");
        let sv = q.scales.view(&[q.n]).expect("view scales");
        let _ = kernels::dequantize_rowwise_i8((&mut out).partition([bm, q.d]), &iv, &sv)
            .generics(vec![bm.to_string(), q.d.to_string()])
            .sync_on(&rt.stream)
            .expect("dequantize_rowwise_i8 kernel");
    }
    store.insert_tensor(out, vec![q.n, q.d])
}
