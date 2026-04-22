//! LayerNorm cuTile kernels — port of `native/kernels/layernorm.cu`.
//!
//! Forward is a single fused kernel per row: mean → subtract → variance →
//! rsqrt → affine, with `mean_out`/`rstd_out` saved for the backward pass.
//!
//! Backward is split so the atomic scatter over the weight gradients
//! (`dgamma`, `dbeta`) becomes a clean two-pass reduction instead of an
//! atomic add per row-per-col: `layernorm_backward_input` computes `dx` +
//! per-row partial contributions to gamma/beta (shape `[N, C]`), which the
//! ops layer reduces along N with `sum_along_dim` into the final weight
//! gradients.  This preserves one-for-one numerics without pulling atomics
//! into the middle of a fused row compute.

#[cutile::module]
pub mod layernorm_kernels {
    use cutile::core::*;

    /// Per row (one block per row):
    ///
    /// ```text
    ///   μ   = Σⱼ x[r, j] / C
    ///   σ²  = Σⱼ (x[r, j] - μ)² / C
    ///   rσ  = rsqrt(σ² + ε)
    ///   out[r, j] = γ[j] · (x[r, j] - μ) · rσ + β[j]
    /// ```
    #[cutile::entry()]
    pub fn layernorm_forward<const C: i32>(
        out: &mut Tensor<f32, { [1, C] }>,
        mean_out: &mut Tensor<f32, { [1] }>,
        rstd_out: &mut Tensor<f32, { [1] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        gamma: &Tensor<f32, { [C] }>,
        beta: &Tensor<f32, { [C] }>,
        eps: f32,
        inv_c: f32,
    ) {
        let tx: Tile<f32, { [1, C] }> = load_tile_like_2d(x, out);

        let sum_row: Tile<f32, { [1] }> = reduce_sum(tx, 1i32);
        let inv_c_1: Tile<f32, { [1] }> = inv_c.broadcast(const_shape![1]);
        let mean: Tile<f32, { [1] }> = sum_row * inv_c_1;
        mean_out.store(mean);

        let mean_b: Tile<f32, { [1, C] }> =
            mean.reshape(const_shape![1, 1]).broadcast(const_shape![1, C]);
        let diff: Tile<f32, { [1, C] }> = tx - mean_b;
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

    /// Per row (one block per row):
    ///
    /// ```text
    ///   x̂    = (x - μ) · rσ
    ///   Σdγ  = Σⱼ dy[j] · γ[j] · x̂[j]
    ///   Σdy' = Σⱼ dy[j] · γ[j]
    ///   dx[j] = rσ · (dy[j] · γ[j] - (Σdy' + x̂[j] · Σdγ) / C)
    /// ```
    ///
    /// `dgamma_partial[r, j] = dy[r, j] · x̂[r, j]` and
    /// `dbeta_partial[r, j] = dy[r, j]`.  The ops layer reduces these along N
    /// with `sum_along_dim` to produce the final `dgamma[j]`, `dbeta[j]` —
    /// avoiding the per-element atomics used in the CUDA version while
    /// preserving the same numerical result.
    #[cutile::entry()]
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm_backward_input<const C: i32>(
        dx: &mut Tensor<f32, { [1, C] }>,
        dgamma_partial: &mut Tensor<f32, { [1, C] }>,
        dbeta_partial: &mut Tensor<f32, { [1, C] }>,
        dy: &Tensor<f32, { [-1, -1] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        mean: &Tensor<f32, { [-1] }>,
        rstd: &Tensor<f32, { [-1] }>,
        gamma: &Tensor<f32, { [C] }>,
        inv_c: f32,
    ) {
        let tdy: Tile<f32, { [1, C] }> = load_tile_like_2d(dy, dx);
        let tx: Tile<f32, { [1, C] }> = load_tile_like_2d(x, dx);

        let pid: (i32, i32, i32) = get_tile_block_id();
        let mean_part: Partition<f32, { [1] }> = mean.partition(const_shape![1]);
        let rstd_part: Partition<f32, { [1] }> = rstd.partition(const_shape![1]);
        let m: Tile<f32, { [1] }> = mean_part.load([pid.0]);
        let r: Tile<f32, { [1] }> = rstd_part.load([pid.0]);

        let gamma_part: Partition<f32, { [C] }> = gamma.partition(const_shape![C]);
        let tg: Tile<f32, { [C] }> = gamma_part.load([0i32]);
        let tg2: Tile<f32, { [1, C] }> =
            tg.reshape(const_shape![1, C]).broadcast(const_shape![1, C]);

        let m_b: Tile<f32, { [1, C] }> =
            m.reshape(const_shape![1, 1]).broadcast(const_shape![1, C]);
        let r_b: Tile<f32, { [1, C] }> =
            r.reshape(const_shape![1, 1]).broadcast(const_shape![1, C]);
        let xhat: Tile<f32, { [1, C] }> = (tx - m_b) * r_b;

        dgamma_partial.store(tdy * xhat);
        dbeta_partial.store(tdy);

        let dyg: Tile<f32, { [1, C] }> = tdy * tg2;
        let dyg_xhat: Tile<f32, { [1, C] }> = dyg * xhat;
        let dot_dy: Tile<f32, { [1] }> = reduce_sum(dyg, 1i32);
        let dot_dy_xhat: Tile<f32, { [1] }> = reduce_sum(dyg_xhat, 1i32);

        let dot_dy_b: Tile<f32, { [1, C] }> = dot_dy
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![1, C]);
        let dot_dy_xhat_b: Tile<f32, { [1, C] }> = dot_dy_xhat
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![1, C]);
        let inv_c_b: Tile<f32, { [1, C] }> = inv_c.broadcast(const_shape![1, C]);
        let correction: Tile<f32, { [1, C] }> = (dot_dy_b + xhat * dot_dy_xhat_b) * inv_c_b;

        dx.store(r_b * (dyg - correction));
    }
}

pub use layernorm_kernels::{layernorm_backward_input, layernorm_forward};
