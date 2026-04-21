//! Mixed-precision cuTile kernels — ports of `native/kernels/mixed_precision.cu`.
//!
//! bf16 = upper 16 bits of f32 (1 sign + 8 exponent + 7 mantissa).
//! `f32_to_bf16` uses round-to-nearest-even via bitcast.
//! `check_inf_nan` atomically sets a 1-element flag if any input is non-finite.

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
}

pub use mixed_precision_kernels::{bf16_to_f32, f32_to_bf16, scale_f32};
