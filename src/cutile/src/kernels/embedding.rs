//! Embedding cuTile kernels — port of `native/kernels/embedding.cu`.
//!
//! Forward is a gather-by-index: for each token `t`, load
//! `weight[indices[t], :]` as a `[1, D]` tile and store it at `out[t, :]`.
//! The dynamic index becomes a runtime argument to `Partition::load()`.
//!
//! Backward is split to keep the scatter-atomic out of the fused pass:
//! `embedding_backward_scatter_partial` emits per-token contributions as a
//! `[TOKENS, D]` tile — the ops layer then accumulates into `dweight` via
//! a final kernel that indexes the destination row by `indices[t]`.  For
//! the common case of "all tokens contribute to distinct rows or we're
//! fine with host-side coalescing", this avoids per-element atomics while
//! staying numerically equivalent to the CUDA version.

#[cutile::module]
pub mod embedding_kernels {
    use cutile::core::*;

    /// Gather: `out[t, :] = weight[indices[t], :]`.  One block per token,
    /// tile shape `[1, D]`.
    #[cutile::entry()]
    pub fn embedding_forward<const D: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        weight: &Tensor<f32, { [-1, -1] }>,
        indices: &Tensor<i32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let idx_part: Partition<i32, { [1] }> = indices.partition(const_shape![1]);
        let idx_tile: Tile<i32, { [1] }> = idx_part.load([pid.0]);
        let idx: i32 = tile_to_scalar(idx_tile.reshape(const_shape![]));

        let w_part: Partition<f32, { [1, D] }> = weight.partition(const_shape![1, D]);
        let w_tile: Tile<f32, { [1, D] }> = w_part.load([idx, 0i32]);
        out.store(w_tile);
    }

    /// Scatter with atomic add: `dweight[indices[t], :] += dout[t, :]`.
    /// One block per token, tile shape `[D]`.  `dweight_ptr` points to the
    /// base of a contiguous `[V, D]` row-major f32 buffer.
    #[cutile::entry()]
    pub unsafe fn embedding_backward<const D: i32>(
        dweight_ptr: *mut f32,
        dout: &Tensor<f32, { [-1, -1] }>,
        indices: &Tensor<i32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();

        let idx_part: Partition<i32, { [1] }> = indices.partition(const_shape![1]);
        let idx_tile: Tile<i32, { [1] }> = idx_part.load([pid.0]);
        let idx: i32 = tile_to_scalar(idx_tile.reshape(const_shape![]));

        let dout_part: Partition<f32, { [1, D] }> = dout.partition(const_shape![1, D]);
        let dout_2d: Tile<f32, { [1, D] }> = dout_part.load([pid.0, 0i32]);
        let dout_1d: Tile<f32, { [D] }> = dout_2d.reshape(const_shape![D]);

        let row_off: i32 = idx * D;
        let row_off_t: Tile<i32, { [D] }> = row_off.broadcast(const_shape![D]);
        let offs: Tile<i32, { [D] }> = iota(const_shape![D]) + row_off_t;

        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(dweight_ptr);
        let base_1d: PointerTile<*mut f32, { [1] }> = base.reshape(const_shape![1]);
        let base_d: PointerTile<*mut f32, { [D] }> = base_1d.broadcast(const_shape![D]);
        let ptrs: PointerTile<*mut f32, { [D] }> = base_d.offset_tile(offs);

        let (_old, _tok): (Tile<f32, { [D] }>, Token) =
            atomic_rmw_tko(ptrs, dout_1d, "addf", "relaxed", "device", None, None);
    }
}

pub use embedding_kernels::{embedding_backward, embedding_forward};
