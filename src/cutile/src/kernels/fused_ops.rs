//! Fused ops cuTile kernels — port of `native/kernels/fused_ops.cu`.
//!
//! - `residual_layernorm_forward` fuses `x + residual` with layernorm so
//!   the sum lives in registers for the row-reduce passes — one GMEM
//!   round-trip instead of three.
//! - `bias_gelu_forward` / `bias_gelu_backward` fuse `x + bias[j]` with
//!   the tanh-approx GELU (and its derivative).  The backward pass emits a
//!   per-element `dbias_partial` tile that the ops layer reduces along
//!   dim-0 with `sum_along_dim` to produce the final `dbias[j]` — matching
//!   the CUDA version's result without the per-thread `atomicAdd`.

#[cutile::module]
pub mod fused_ops_kernels {
    use cutile::core::*;

    /// Per row (one block per row):
    ///
    /// ```text
    ///   r     = x + residual                (saved for backward)
    ///   μ, rσ = layernorm_stats(r)
    ///   out   = γ · (r - μ) · rσ + β
    /// ```
    #[cutile::entry()]
    #[allow(clippy::too_many_arguments)]
    pub fn residual_layernorm_forward<const C: i32>(
        out: &mut Tensor<f32, { [1, C] }>,
        residual_out: &mut Tensor<f32, { [1, C] }>,
        mean_out: &mut Tensor<f32, { [1] }>,
        rstd_out: &mut Tensor<f32, { [1] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        residual: &Tensor<f32, { [-1, -1] }>,
        gamma: &Tensor<f32, { [C] }>,
        beta: &Tensor<f32, { [C] }>,
        eps: f32,
        inv_c: f32,
    ) {
        let tx: Tile<f32, { [1, C] }> = load_tile_like_2d(x, out);
        let tr: Tile<f32, { [1, C] }> = load_tile_like_2d(residual, out);
        let r: Tile<f32, { [1, C] }> = tx + tr;
        residual_out.store(r);

        let sum_row: Tile<f32, { [1] }> = reduce_sum(r, 1i32);
        let inv_c_1: Tile<f32, { [1] }> = inv_c.broadcast(const_shape![1]);
        let mean: Tile<f32, { [1] }> = sum_row * inv_c_1;
        mean_out.store(mean);

        let mean_b: Tile<f32, { [1, C] }> =
            mean.reshape(const_shape![1, 1]).broadcast(const_shape![1, C]);
        let diff: Tile<f32, { [1, C] }> = r - mean_b;
        let sqd: Tile<f32, { [1, C] }> = diff * diff;
        let sum_sq: Tile<f32, { [1] }> = reduce_sum(sqd, 1i32);
        let var: Tile<f32, { [1] }> = sum_sq * inv_c_1;
        let eps_t: Tile<f32, { [1] }> = eps.broadcast(const_shape![1]);
        let rstd: Tile<f32, { [1] }> = rsqrt(var + eps_t, ftz::Disabled);
        rstd_out.store(rstd);

        let rstd_b: Tile<f32, { [1, C] }> =
            rstd.reshape(const_shape![1, 1]).broadcast(const_shape![1, C]);

        let gamma_part: Partition<f32, { [C] }> = gamma.partition(const_shape![C]);
        let beta_part: Partition<f32, { [C] }> = beta.partition(const_shape![C]);
        let tg: Tile<f32, { [C] }> = gamma_part.load([0i32]);
        let tb: Tile<f32, { [C] }> = beta_part.load([0i32]);
        let tg2: Tile<f32, { [1, C] }> =
            tg.reshape(const_shape![1, C]).broadcast(const_shape![1, C]);
        let tb2: Tile<f32, { [1, C] }> =
            tb.reshape(const_shape![1, C]).broadcast(const_shape![1, C]);

        out.store(tg2 * diff * rstd_b + tb2);
    }

    /// Fused `out = gelu(x + bias[j])` with `x: [N, C]`, `bias: [C]`.
    /// GELU is the tanh approximation: `0.5·v·(1 + tanh(√(2/π)·(v + 0.044715·v³)))`.
    #[cutile::entry()]
    pub fn bias_gelu_forward<const BN: i32, const C: i32>(
        out: &mut Tensor<f32, { [BN, C] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        bias: &Tensor<f32, { [C] }>,
    ) {
        let tx: Tile<f32, { [BN, C] }> = load_tile_like_2d(x, out);
        let bias_part: Partition<f32, { [C] }> = bias.partition(const_shape![C]);
        let tb: Tile<f32, { [C] }> = bias_part.load([0i32]);
        let tb2: Tile<f32, { [BN, C] }> =
            tb.reshape(const_shape![1, C]).broadcast(const_shape![BN, C]);
        let v: Tile<f32, { [BN, C] }> = tx + tb2;

        let half: Tile<f32, { [BN, C] }> = constant(0.5f32, const_shape![BN, C]);
        let one: Tile<f32, { [BN, C] }> = constant(1.0f32, const_shape![BN, C]);
        let k0: Tile<f32, { [BN, C] }> = constant(0.7978845608028654f32, const_shape![BN, C]);
        let k1: Tile<f32, { [BN, C] }> = constant(0.044715f32, const_shape![BN, C]);
        let v2: Tile<f32, { [BN, C] }> = v * v;
        let v3: Tile<f32, { [BN, C] }> = v2 * v;
        let inner: Tile<f32, { [BN, C] }> = k0 * (v + k1 * v3);
        let th: Tile<f32, { [BN, C] }> = tanh(inner);
        out.store(half * v * (one + th));
    }

    /// Fused backward for `bias_gelu_forward`:
    ///
    /// ```text
    ///   v    = x + bias[j]
    ///   φ    = tanh(√(2/π)·(v + 0.044715·v³))
    ///   ∂v/∂x = 1
    ///   gelu'(v) = 0.5·(1 + φ) + 0.5·v·(1 - φ²)·√(2/π)·(1 + 3·0.044715·v²)
    ///   dx[i, j] = grad[i, j] · gelu'(v)
    ///   dbias_partial[i, j] = dx[i, j]     (reduced along N by the ops layer)
    /// ```
    #[cutile::entry()]
    pub fn bias_gelu_backward<const BN: i32, const C: i32>(
        dx: &mut Tensor<f32, { [BN, C] }>,
        dbias_partial: &mut Tensor<f32, { [BN, C] }>,
        grad: &Tensor<f32, { [-1, -1] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        bias: &Tensor<f32, { [C] }>,
    ) {
        let tg: Tile<f32, { [BN, C] }> = load_tile_like_2d(grad, dx);
        let tx: Tile<f32, { [BN, C] }> = load_tile_like_2d(x, dx);
        let bias_part: Partition<f32, { [C] }> = bias.partition(const_shape![C]);
        let tb: Tile<f32, { [C] }> = bias_part.load([0i32]);
        let tb2: Tile<f32, { [BN, C] }> =
            tb.reshape(const_shape![1, C]).broadcast(const_shape![BN, C]);
        let v: Tile<f32, { [BN, C] }> = tx + tb2;

        let half: Tile<f32, { [BN, C] }> = constant(0.5f32, const_shape![BN, C]);
        let one: Tile<f32, { [BN, C] }> = constant(1.0f32, const_shape![BN, C]);
        let three: Tile<f32, { [BN, C] }> = constant(3.0f32, const_shape![BN, C]);
        let k0: Tile<f32, { [BN, C] }> = constant(0.7978845608028654f32, const_shape![BN, C]);
        let k1: Tile<f32, { [BN, C] }> = constant(0.044715f32, const_shape![BN, C]);

        let v2: Tile<f32, { [BN, C] }> = v * v;
        let v3: Tile<f32, { [BN, C] }> = v2 * v;
        let inner: Tile<f32, { [BN, C] }> = k0 * (v + k1 * v3);
        let th: Tile<f32, { [BN, C] }> = tanh(inner);
        let sech2: Tile<f32, { [BN, C] }> = one - th * th;
        let d_inner: Tile<f32, { [BN, C] }> = k0 * (one + three * k1 * v2);
        let dgelu: Tile<f32, { [BN, C] }> = half * (one + th) + half * v * sech2 * d_inner;
        let out: Tile<f32, { [BN, C] }> = tg * dgelu;
        dx.store(out);
        dbias_partial.store(out);
    }
}

pub use fused_ops_kernels::{bias_gelu_backward, bias_gelu_forward, residual_layernorm_forward};
