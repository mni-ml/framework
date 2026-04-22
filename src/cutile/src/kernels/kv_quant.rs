//! KV-cache quantization cuTile kernels — port of `native/kernels/kv_quant.cu`.
//!
//! Row-wise symmetric i8 quantization used by the KV cache:
//!
//! - `compute_rowwise_scale`: per row `scale = max(|row|) / 127`,
//!   floored at `1e-8` so we never divide by zero at quantize time.
//! - `quantize_rowwise_i8`: per element `out = clamp(round(x / scale), ±127)`.
//! - `dequantize_rowwise_i8`: per element `out = (f32)q * scale[row]`.

#[cutile::module]
pub mod kv_quant_kernels {
    use cutile::core::*;

    /// Per row (one block per row):
    ///
    /// ```text
    ///   scales[r] = max(max(|input[r, :]|) / 127, 1e-8)
    /// ```
    #[cutile::entry()]
    pub fn compute_rowwise_scale<const D: i32>(
        scales: &mut Tensor<f32, { [1] }>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let in_part: Partition<f32, { [1, D] }> = input.partition(const_shape![1, D]);
        let tx: Tile<f32, { [1, D] }> = in_part.load([pid.0, 0i32]);
        let abs_tx: Tile<f32, { [1, D] }> = absf(tx);
        let row_max: Tile<f32, { [1] }> = reduce_max(abs_tx, 1i32);

        let inv_127: f32 = 1.0f32 / 127.0f32;
        let inv_127_t: Tile<f32, { [1] }> = inv_127.broadcast(const_shape![1]);
        let divided: Tile<f32, { [1] }> = row_max * inv_127_t;
        let eps: Tile<f32, { [1] }> = constant(1.0e-8f32, const_shape![1]);
        let clamped: Tile<f32, { [1] }> = max_tile(divided, eps);
        scales.store(clamped);
    }

    /// Per row tile `[BM, D]`:
    ///
    /// ```text
    ///   q = clamp(round(input[r, j] / scales[r]), -127, 127)
    ///   output[r, j] = (i8)q
    /// ```
    ///
    /// Round is `floor(x + 0.5)` for positive / `-floor(-x + 0.5)` for
    /// negative — matching CUDA's `roundf` semantics for the values that
    /// matter after quantization clamping.  Symmetric around zero here.
    #[cutile::entry()]
    pub fn quantize_rowwise_i8<const BM: i32, const D: i32>(
        output: &mut Tensor<i8, { [BM, D] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        scales: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let in_part: Partition<f32, { [BM, D] }> = input.partition(const_shape![BM, D]);
        let sc_part: Partition<f32, { [BM] }> = scales.partition(const_shape![BM]);
        let tx: Tile<f32, { [BM, D] }> = in_part.load([pid.0, 0i32]);
        let ts: Tile<f32, { [BM] }> = sc_part.load([pid.0]);
        let ts_b: Tile<f32, { [BM, D] }> = ts
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, D]);

        let scaled: Tile<f32, { [BM, D] }> = tx / ts_b;

        // Round-to-nearest via `floor(|x| + 0.5) * sign(x)`.
        let zero: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);
        let half: Tile<f32, { [BM, D] }> = constant(0.5f32, const_shape![BM, D]);
        let abs_scaled: Tile<f32, { [BM, D] }> = absf(scaled);
        let rounded_abs: Tile<f32, { [BM, D] }> = floor(abs_scaled + half);
        let is_neg: Tile<bool, { [BM, D] }> = lt_tile(scaled, zero);
        let neg_rounded: Tile<f32, { [BM, D] }> = zero - rounded_abs;
        let rounded: Tile<f32, { [BM, D] }> = select(is_neg, neg_rounded, rounded_abs);

        let lo: Tile<f32, { [BM, D] }> = constant(-127.0f32, const_shape![BM, D]);
        let hi: Tile<f32, { [BM, D] }> = constant(127.0f32, const_shape![BM, D]);
        let clamped: Tile<f32, { [BM, D] }> = max_tile(min_tile(rounded, hi), lo);
        let q: Tile<i8, { [BM, D] }> = convert_tile(clamped);
        output.store(q);
    }

    /// Per row tile `[BM, D]`:
    ///
    /// ```text
    ///   output[r, j] = (f32)input[r, j] * scales[r]
    /// ```
    #[cutile::entry()]
    pub fn dequantize_rowwise_i8<const BM: i32, const D: i32>(
        output: &mut Tensor<f32, { [BM, D] }>,
        input: &Tensor<i8, { [-1, -1] }>,
        scales: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let in_part: Partition<i8, { [BM, D] }> = input.partition(const_shape![BM, D]);
        let sc_part: Partition<f32, { [BM] }> = scales.partition(const_shape![BM]);
        let tx_i8: Tile<i8, { [BM, D] }> = in_part.load([pid.0, 0i32]);
        let tx_f: Tile<f32, { [BM, D] }> = convert_tile(tx_i8);
        let ts: Tile<f32, { [BM] }> = sc_part.load([pid.0]);
        let ts_b: Tile<f32, { [BM, D] }> = ts
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, D]);
        output.store(tx_f * ts_b);
    }
}

pub use kv_quant_kernels::{compute_rowwise_scale, dequantize_rowwise_i8, quantize_rowwise_i8};
