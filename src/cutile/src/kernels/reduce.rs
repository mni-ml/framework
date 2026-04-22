//! Reduction cuTile kernels.
//!
//! Two flavors of reduction:
//!
//! 1. **Global reductions** (`sum_atomic`) run a single pass: each block
//!    in-tile-reduces its `BLOCK`-sized chunk via `reduce_sum` and then
//!    `atomic_rmw_tko "addf"`s the partial into a single scalar output
//!    location — same shape as the CUDA C++ SIMT backend's `sum_all` kernel.
//!    The last block is the usual "partial tile" case: cuTile's
//!    `Tensor::partition()` returns a zero-padded view so out-of-range
//!    lanes contribute the additive identity `0.0f32`, and the reduction
//!    stays correct for arbitrary `n`.
//!
//! 2. **Along-dim reductions** (`sum_along_last`, `mean_along_last`,
//!    `max_along_last`) are 1-block-per-output-row kernels that collapse
//!    the last axis of a 2D tile `[BM, DIM]` → `[BM]`.  The general 3D
//!    `(outer, dim, inner)` case in the CUDA kernels is handled in the ops
//!    layer by permuting the reduction axis to be last before launch
//!    (matches the `dim == -1` fast path).
//!
//!    `broadcast_last` is the inverse: expands `[BM]` across a new last
//!    dim of size `DIM` — the forward of the sum-backward pattern
//!    (`sum_broadcast_f32` in the CUDA backend).

#[cutile::module]
pub mod reduce_kernels {
    use cutile::core::*;

    /// Single-pass atomic global sum.
    ///
    /// Each block reduces its `BLOCK`-sized tile of `x` with `reduce_sum`
    /// and `atomic_rmw_tko "addf"`s the partial into `out_ptr[0]`.  The
    /// caller zero-initialises `*out_ptr` before launch; the grid size is
    /// `ceil(n / BLOCK)`.  Tail handling is automatic via cuTile's
    /// zero-padded `partition()` — out-of-range lanes load `0.0f32`.
    #[cutile::entry()]
    pub unsafe fn sum_atomic<const BLOCK: i32>(
        out_ptr: *mut f32,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part_x: Partition<f32, { [BLOCK] }> = x.partition(const_shape![BLOCK]);
        let tile_x: Tile<f32, { [BLOCK] }> = part_x.load([pid.0]);
        let s_scalar: Tile<f32, { [] }> = reduce_sum(tile_x, 0i32);
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

    /// `out[r] = Σⱼ x[r, j]`.  One block per row tile.
    #[cutile::entry()]
    pub fn sum_along_last<const BM: i32, const DIM: i32>(
        out: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<f32, { [BM, DIM] }> = x.partition(const_shape![BM, DIM]);
        let tx: Tile<f32, { [BM, DIM] }> = part.load([pid.0, 0i32]);
        let s: Tile<f32, { [BM] }> = reduce_sum(tx, 1i32);
        out.store(s);
    }

    /// `out[r] = Σⱼ x[r, j] / DIM`.
    #[cutile::entry()]
    pub fn mean_along_last<const BM: i32, const DIM: i32>(
        out: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<f32, { [BM, DIM] }> = x.partition(const_shape![BM, DIM]);
        let tx: Tile<f32, { [BM, DIM] }> = part.load([pid.0, 0i32]);
        let s: Tile<f32, { [BM] }> = reduce_sum(tx, 1i32);
        let inv_s: f32 = 1.0f32 / (DIM as f32);
        let inv: Tile<f32, { [BM] }> = inv_s.broadcast(out.shape());
        out.store(s * inv);
    }

    /// `out[r] = maxⱼ x[r, j]`.
    #[cutile::entry()]
    pub fn max_along_last<const BM: i32, const DIM: i32>(
        out: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<f32, { [BM, DIM] }> = x.partition(const_shape![BM, DIM]);
        let tx: Tile<f32, { [BM, DIM] }> = part.load([pid.0, 0i32]);
        let s: Tile<f32, { [BM] }> = reduce_max(tx, 1i32);
        out.store(s);
    }

    /// `out[r, j] = x[r]`.  Broadcast a per-row scalar across a new last
    /// dim of size `DIM` — forward of sum-backward.
    #[cutile::entry()]
    pub fn broadcast_last<const BM: i32, const DIM: i32>(
        out: &mut Tensor<f32, { [BM, DIM] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<f32, { [BM] }> = x.partition(const_shape![BM]);
        let tx: Tile<f32, { [BM] }> = part.load([pid.0]);
        let tx_b: Tile<f32, { [BM, DIM] }> = tx
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, DIM]);
        out.store(tx_b);
    }

    /// `out[i, j] = Σᵢ g[i, j]` — sum along dim 0 of a `[ROWS, BN]` tile.
    /// Used by the ops layer to fold per-row `dgamma`/`dbeta` / `dbias`
    /// partials into their final shape.
    #[cutile::entry()]
    pub fn sum_along_first<const ROWS: i32, const BN: i32>(
        out: &mut Tensor<f32, { [BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<f32, { [ROWS, BN] }> = x.partition(const_shape![ROWS, BN]);
        let tx: Tile<f32, { [ROWS, BN] }> = part.load([0i32, pid.0]);
        let s: Tile<f32, { [BN] }> = reduce_sum(tx, 0i32);
        out.store(s);
    }
}

pub use reduce_kernels::{
    broadcast_last, max_along_last, mean_along_last, sum_along_first, sum_along_last, sum_atomic,
};
