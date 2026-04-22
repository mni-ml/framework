//! Mixed-precision cuTile kernels — ports of `native/kernels/mixed_precision.cu`.
//!
//! bf16 = upper 16 bits of f32 (1 sign + 8 exponent + 7 mantissa).
//! `f32_to_bf16` uses round-to-nearest-even via bitcast.
//! `check_inf_nan_f32` atomically sets a 1-element f32 flag to 1.0 if any
//! element of `data` is non-finite — mirrors the CUDA `isinf(x) || isnan(x)`
//! check (both are captured by "all exponent bits set").

#[cutile::module]
pub mod mixed_precision_kernels {
    use cutile::core::*;

    /// `data[i] *= scale` in place.  Matches `scale_f32` in the CUDA backend.
    #[cutile::entry()]
    pub fn scale_f32<const S: [i32; 1]>(data: &mut Tensor<f32, S>, scale: f32) {
        let t = load_tile_mut(data);
        let s: Tile<f32, S> = scale.broadcast(data.shape());
        data.store(t * s);
    }

    /// `out[i] = bf16(x[i])` via round-to-nearest-even on the
    /// upper 16 bits.  Output tensor is `u16` reinterpretable as bf16.
    #[cutile::entry()]
    pub fn f32_to_bf16<const S: [i32; 1]>(
        out: &mut Tensor<u16, S>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, out);
        let bits: Tile<u32, S> = bitcast(tx);
        let bias: Tile<u32, S> = constant(0x7FFFu32, out.shape());
        let one: Tile<u32, S> = constant(1u32, out.shape());
        let sixteen: Tile<u32, S> = constant(16u32, out.shape());
        // Round-to-nearest-even: (f + 0x7FFF + ((f >> 16) & 1)) >> 16
        let high: Tile<u32, S> = shri(bits, sixteen);
        let lsb: Tile<u32, S> = andi(high, one);
        let rounded: Tile<u32, S> = shri(bits + bias + lsb, sixteen);
        let out_tile: Tile<u16, S> = trunci(rounded);
        out.store(out_tile);
    }

    /// `out[i] = f32((u32(x[i])) << 16)`, i.e. zero-extend the bf16
    /// mantissa and bitcast.
    #[cutile::entry()]
    pub fn bf16_to_f32<const S: [i32; 1]>(
        out: &mut Tensor<f32, S>,
        x: &Tensor<u16, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, out);
        let ext: Tile<u32, S> = exti(tx);
        let sixteen: Tile<u32, S> = constant(16u32, out.shape());
        let shifted: Tile<u32, S> = shli(ext, sixteen);
        let f: Tile<f32, S> = bitcast(shifted);
        out.store(f);
    }

    /// Per tile `[BM]`:
    ///
    /// ```text
    ///   non_finite[i] = (bitcast<u32>(data[i]) & 0x7F800000u) == 0x7F800000u
    ///   any           = reduce_max(non_finite)
    ///   if any: atomically set result[0] = 1.0f
    /// ```
    ///
    /// The atomic is implemented as an `umax` on the bit-pattern of
    /// `result[0]` viewed as `u32`: `0.0f = 0x00000000u`, `1.0f =
    /// 0x3F800000u`, both positive in signed interpretation, so `umax`
    /// is idempotent and monotonic — multiple blocks racing to set 1.0
    /// is safe.  The caller pre-zeroes `result[0]`.
    ///
    /// Subnormals and zeros have a zero exponent field, so they don't
    /// match — matching CUDA's `isinf(x) || isnan(x)` semantics
    /// (which also exclude subnormals).
    #[cutile::entry()]
    pub unsafe fn check_inf_nan_f32<const BM: i32>(
        result_ptr: *mut f32,
        data: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<f32, { [BM] }> = data.partition(const_shape![BM]);
        let tile: Tile<f32, { [BM] }> = part.load([pid.0]);

        let bits: Tile<u32, { [BM] }> = bitcast(tile);
        let exp_mask: Tile<u32, { [BM] }> = constant(0x7F800000u32, const_shape![BM]);
        let exp_bits: Tile<u32, { [BM] }> = andi(bits, exp_mask);
        let non_finite: Tile<bool, { [BM] }> = eq_tile(exp_bits, exp_mask);

        // Encode: non_finite → 1.0f bit-pattern, finite → 0.0f bit-pattern.
        let one_bits: Tile<u32, { [BM] }> = constant(0x3F800000u32, const_shape![BM]);
        let zero_bits: Tile<u32, { [BM] }> = constant(0u32, const_shape![BM]);
        let flag_bits: Tile<u32, { [BM] }> = select(non_finite, one_bits, zero_bits);

        // Reduce to a single u32: 0 if all finite, 0x3F800000 if any non-finite.
        let any_bits_0: Tile<u32, { [] }> = reduce_max(flag_bits, 0i32);
        let any_bits: Tile<u32, { [1] }> = any_bits_0.reshape(const_shape![1]);

        // Atomic umax into &result[0] viewed as u32.
        let base_f: PointerTile<*mut f32, { [] }> = pointer_to_tile(result_ptr);
        let base_u: PointerTile<*mut u32, { [] }> = ptr_to_ptr(base_f);
        let base_u_1: PointerTile<*mut u32, { [1] }> = base_u.reshape(const_shape![1]);

        let (_old, _tok): (Tile<u32, { [1] }>, Token) = atomic_rmw_tko(
            base_u_1,
            any_bits,
            "umax",
            "relaxed",
            "device",
            None,
            None,
        );
    }
}

pub use mixed_precision_kernels::{bf16_to_f32, check_inf_nan_f32, f32_to_bf16, scale_f32};
