//! Softmax cuTile kernels — port of `native/kernels/softmax.cu`.
//!
//! Written as 2D kernels over `[rows, DIM]` where `rows = outer * inner`; the
//! general 3D `(outer, dim, inner)` case in the CUDA kernel is handled in the
//! ops layer by permuting the reduction dim to be last before launch (inner =
//! 1 post-permute), which matches the fast path for every softmax call in ML
//! (softmax-along-last-axis).  One block per row; `DIM` is a const generic.

#[cutile::module]
pub mod softmax_kernels {
    use cutile::core::*;

    /// `out[r, :] = softmax(x[r, :])` via max-shift → exp → divide-by-sum.
    #[cutile::entry()]
    pub fn softmax_forward<const BM: i32, const DIM: i32>(
        out: &mut Tensor<f32, { [BM, DIM] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tx: Tile<f32, { [BM, DIM] }> = load_tile_like_2d(x, out);
        let row_max: Tile<f32, { [BM] }> = reduce_max(tx, 1i32);
        let row_max: Tile<f32, { [BM, DIM] }> = row_max
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, DIM]);
        let e: Tile<f32, { [BM, DIM] }> = exp(tx - row_max);
        let row_sum: Tile<f32, { [BM] }> = reduce_sum(e, 1i32);
        let row_sum: Tile<f32, { [BM, DIM] }> = row_sum
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, DIM]);
        out.store(e / row_sum);
    }

    /// `dx[r, i] = out[r, i] · (dy[r, i] - Σⱼ dy[r, j]·out[r, j])`.
    #[cutile::entry()]
    pub fn softmax_backward<const BM: i32, const DIM: i32>(
        dx: &mut Tensor<f32, { [BM, DIM] }>,
        dy: &Tensor<f32, { [-1, -1] }>,
        out: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tdy: Tile<f32, { [BM, DIM] }> = load_tile_like_2d(dy, dx);
        let tout: Tile<f32, { [BM, DIM] }> = load_tile_like_2d(out, dx);
        let dot_terms: Tile<f32, { [BM, DIM] }> = tdy * tout;
        let dot: Tile<f32, { [BM] }> = reduce_sum(dot_terms, 1i32);
        let dot: Tile<f32, { [BM, DIM] }> = dot
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, DIM]);
        dx.store(tout * (tdy - dot));
    }
}

pub use softmax_kernels::{softmax_backward, softmax_forward};
