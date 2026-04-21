//! Gradient utility cuTile kernels — ports of `native/kernels/grad_util.cu`.
//!
//! `grad_norm_sq_partial` is a per-block reduction driving global
//! `Σ gradᵢ²`; the global reduction is finalized with a `sum_block`
//! pass 2 launch in `ops/grad_util.rs`.

#[cutile::module]
pub mod grad_util_kernels {
    use cutile::core::*;

    /// In-place gradient scaling: `grad[i] *= scale`.
    #[cutile::entry()]
    pub fn grad_clip<const S: [i32; 1]>(grad: &mut Tensor<f32, S>, scale: f32) {
        let tg = load_tile_mut(grad);
        let s: Tile<f32, S> = scale.broadcast(grad.shape());
        grad.store(tg * s);
    }

    /// Per-block `Σ gradᵢ²`.  One scalar per block, then finalized by a
    /// `sum_block` pass 2.
    #[cutile::entry()]
    pub fn grad_norm_sq_partial<const BLOCK: i32>(
        partials: &mut Tensor<f32, { [1] }>,
        grad: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<f32, { [BLOCK] }> = grad.partition(const_shape![BLOCK]);
        let tile: Tile<f32, { [BLOCK] }> = part.load([pid.0]);
        let sq: Tile<f32, { [BLOCK] }> = tile * tile;
        let s: Tile<f32, { [] }> = reduce_sum(sq, 0i32);
        let s1: Tile<f32, { [1] }> = s.reshape(const_shape![1]);
        partials.store(s1);
    }
}

pub use grad_util_kernels::{grad_clip, grad_norm_sq_partial};
