//! Data-loader cuTile kernels — port of `native/kernels/data.cu`.
//!
//! `sample_batch` gathers `batch_size` contiguous slices from a 1D dataset at
//! runtime-specified offsets — one block per batch element, one block_size
//! tile of `i32` per (inputs, targets).  Matches the CUDA
//! `sample_batch_i32` semantics (targets are shifted by +1).
//!
//! ## Why the pointer-tile gather (not view-based loads)
//!
//! TMA-lowerable view loads (`partition(tile).load(idx)`) require `idx` in
//! *tile units* — valid element starts are `0, BLOCK, 2·BLOCK, …`.  GPT-style
//! training sampling picks arbitrary random token positions (e.g. `7, 42, 3`)
//! that are not multiples of `BLOCK`, so there is no way to express
//! "load `BLOCK` contiguous elements starting at runtime offset `s`" as a
//! single tile-aligned `load_from_view`.  TMA descriptors are also fixed at
//! launch, so per-block variable offsets are outside the TMA model entirely.
//! The CUDA C++ SIMT reference has the same constraint and uses a scalar
//! per-thread loop; this cuTile version improves on it by doing a
//! block-cooperative vectorized gather via `load_ptr_tko`.

#[cutile::module]
pub mod data_kernels {
    use cutile::core::*;

    /// Per batch element `b`:
    ///
    /// ```text
    ///   start = offsets[b]
    ///   out_inputs[b, d]  = dataset[start + d]         (d ∈ 0..BLOCK)
    ///   out_targets[b, d] = dataset[start + d + 1]
    /// ```
    #[cutile::entry()]
    pub unsafe fn sample_batch<const BLOCK: i32>(
        out_inputs: &mut Tensor<i32, { [1, BLOCK] }>,
        out_targets: &mut Tensor<i32, { [1, BLOCK] }>,
        dataset_ptr: *mut i32,
        offsets: &Tensor<i32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();

        let off_part: Partition<i32, { [1] }> = offsets.partition(const_shape![1]);
        let off_tile: Tile<i32, { [1] }> = off_part.load([pid.0]);
        let start: i32 = tile_to_scalar(off_tile.reshape(const_shape![]));
        let start_t: Tile<i32, { [BLOCK] }> = start.broadcast(const_shape![BLOCK]);
        let one_t: Tile<i32, { [BLOCK] }> = constant(1i32, const_shape![BLOCK]);

        let offs_d: Tile<i32, { [BLOCK] }> = iota(const_shape![BLOCK]);
        let in_offs: Tile<i32, { [BLOCK] }> = start_t + offs_d;
        let tgt_offs: Tile<i32, { [BLOCK] }> = in_offs + one_t;

        let base: PointerTile<*mut i32, { [] }> = pointer_to_tile(dataset_ptr);
        let base_1d: PointerTile<*mut i32, { [1] }> = base.reshape(const_shape![1]);
        let base_d: PointerTile<*mut i32, { [BLOCK] }> = base_1d.broadcast(const_shape![BLOCK]);

        let in_ptrs: PointerTile<*mut i32, { [BLOCK] }> = base_d.offset_tile(in_offs);
        let tgt_ptrs: PointerTile<*mut i32, { [BLOCK] }> = base_d.offset_tile(tgt_offs);

        let in_result: (Tile<i32, { [BLOCK] }>, Token) =
            load_ptr_tko(in_ptrs, "relaxed", "device", None, None, None, None);
        let in_tile: Tile<i32, { [BLOCK] }> = in_result.0;
        let in_2d: Tile<i32, { [1, BLOCK] }> = in_tile.reshape(const_shape![1, BLOCK]);
        out_inputs.store(in_2d);

        let tgt_result: (Tile<i32, { [BLOCK] }>, Token) =
            load_ptr_tko(tgt_ptrs, "relaxed", "device", None, None, None, None);
        let tgt_tile: Tile<i32, { [BLOCK] }> = tgt_result.0;
        let tgt_2d: Tile<i32, { [1, BLOCK] }> = tgt_tile.reshape(const_shape![1, BLOCK]);
        out_targets.store(tgt_2d);
    }
}

pub use data_kernels::sample_batch;
