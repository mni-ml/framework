//! Gradient utility cuTile kernels — ports of `native/kernels/grad_util.cu`.
//!
//! `grad_norm_sq_atomic` single-passes `Σ gradᵢ²` into a zero-initialised
//! scalar via `atomic_rmw_tko "addf"`, matching the CUDA C++ SIMT
//! `grad_norm_sq` launch.  Tail lanes load `0.0f32` via cuTile's zero-
//! padded `partition()` so the reduction stays correct for arbitrary `n`.

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

    /// Single-pass atomic global `Σ gradᵢ²`.
    ///
    /// Each block reduces its `BLOCK`-sized tile of `gradᵢ²` with
    /// `reduce_sum` and `atomic_rmw_tko "addf"`s the partial into
    /// `out_ptr[0]`.  Grid size = `ceil(n / BLOCK)`, caller zero-inits
    /// `*out_ptr`.
    #[cutile::entry()]
    pub unsafe fn grad_norm_sq_atomic<const BLOCK: i32>(
        out_ptr: *mut f32,
        grad: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<f32, { [BLOCK] }> = grad.partition(const_shape![BLOCK]);
        let tile: Tile<f32, { [BLOCK] }> = part.load([pid.0]);
        let sq: Tile<f32, { [BLOCK] }> = tile * tile;
        let s_scalar: Tile<f32, { [] }> = reduce_sum(sq, 0i32);
        let s_one: Tile<f32, { [1] }> = s_scalar.reshape(const_shape![1]);
        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let base_1: PointerTile<*mut f32, { [1] }> = base.reshape(const_shape![1]);
        let (_old, _tok): (Tile<f32, { [1] }>, Token) = atomic_rmw_tko(
            base_1,
            s_one,
            "addf",
            "relaxed",
            "device",
            None,
            None,
        );
    }
}

pub use grad_util_kernels::{grad_clip, grad_norm_sq_atomic};
