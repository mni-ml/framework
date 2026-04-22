//! AdamW cuTile kernel — port of `native/kernels/adamw.cu`.
//!
//! Single fused pointwise pass that updates `m`, `v`, and `param` in place.

#[cutile::module]
pub mod adamw_kernels {
    use cutile::core::*;

    /// One AdamW step per parameter element:
    ///
    /// ```text
    ///   mᵢ = β₁·m + (1-β₁)·g
    ///   vᵢ = β₂·v + (1-β₂)·g²
    ///   m̂ = mᵢ / bc₁     v̂ = vᵢ / bc₂
    ///   p ← p·(1 - lr·wd) - lr·m̂ / (√v̂ + ε)
    /// ```
    ///
    /// `bc1` = `1 - β₁^t`, `bc2` = `1 - β₂^t` — computed host-side each step.
    #[cutile::entry()]
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step<const S: [i32; 1]>(
        param: &mut Tensor<f32, S>,
        m: &mut Tensor<f32, S>,
        v: &mut Tensor<f32, S>,
        grad: &Tensor<f32, { [-1] }>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
    ) {
        let tp = load_tile_mut(param);
        let tm = load_tile_mut(m);
        let tv = load_tile_mut(v);
        let tg = load_tile_like_1d(grad, param);

        let s_one_mb1: f32 = 1.0f32 - beta1;
        let s_one_mb2: f32 = 1.0f32 - beta2;
        let s_inv_bc1: f32 = 1.0f32 / bc1;
        let s_inv_bc2: f32 = 1.0f32 / bc2;
        let s_wd_lr: f32 = 1.0f32 - lr * weight_decay;

        let b1: Tile<f32, S> = beta1.broadcast(param.shape());
        let b2: Tile<f32, S> = beta2.broadcast(param.shape());
        let one_mb1: Tile<f32, S> = s_one_mb1.broadcast(param.shape());
        let one_mb2: Tile<f32, S> = s_one_mb2.broadcast(param.shape());
        let eps_t: Tile<f32, S> = eps.broadcast(param.shape());
        let inv_bc1: Tile<f32, S> = s_inv_bc1.broadcast(param.shape());
        let inv_bc2: Tile<f32, S> = s_inv_bc2.broadcast(param.shape());
        let wd_lr: Tile<f32, S> = s_wd_lr.broadcast(param.shape());
        let lr_t: Tile<f32, S> = lr.broadcast(param.shape());

        let new_m: Tile<f32, S> = b1 * tm + one_mb1 * tg;
        let new_v: Tile<f32, S> = b2 * tv + one_mb2 * tg * tg;
        m.store(new_m);
        v.store(new_v);

        let m_hat: Tile<f32, S> = new_m * inv_bc1;
        let v_hat: Tile<f32, S> = new_v * inv_bc2;
        let sv: Tile<f32, S> = sqrt(v_hat, rounding::NegativeInf, ftz::Disabled);
        let denom: Tile<f32, S> = sv + eps_t;
        param.store(tp * wd_lr - lr_t * m_hat / denom);
    }
}

pub use adamw_kernels::adamw_step;
