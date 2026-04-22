//! Cross-entropy cuTile kernels — port of `native/kernels/cross_entropy.cu`.
//!
//! `cross_entropy_forward` is a fused softmax + negative-log-likelihood per
//! row — one block per row, tile shape `[BM, V]`.  It writes the softmax
//! probabilities (needed by the backward pass) alongside the per-row loss.
//!
//! `cross_entropy_backward` is the `(softmax - onehot(target)) · grad_scale`
//! gradient — also per row, fused in a single tile pass.

#[cutile::module]
pub mod cross_entropy_kernels {
    use cutile::core::*;

    /// Per row:
    ///   `softmax_out[r, :] = softmax(logits[r, :])`,
    ///   `losses[r] = -log(softmax_out[r, targets[r]] + 1e-9)`.
    #[cutile::entry()]
    pub fn cross_entropy_forward<const BM: i32, const V: i32>(
        losses: &mut Tensor<f32, { [BM] }>,
        softmax_out: &mut Tensor<f32, { [BM, V] }>,
        logits: &Tensor<f32, { [-1, -1] }>,
        targets: &Tensor<i32, { [-1] }>,
    ) {
        let tl: Tile<f32, { [BM, V] }> = load_tile_like_2d(logits, softmax_out);
        let tt: Tile<i32, { [BM] }> = load_tile_like_1d(targets, losses);

        let row_max: Tile<f32, { [BM] }> = reduce_max(tl, 1i32);
        let row_max_b: Tile<f32, { [BM, V] }> = row_max
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, V]);
        let e: Tile<f32, { [BM, V] }> = exp(tl - row_max_b);
        let row_sum: Tile<f32, { [BM] }> = reduce_sum(e, 1i32);
        let row_sum_b: Tile<f32, { [BM, V] }> = row_sum
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, V]);
        let sm: Tile<f32, { [BM, V] }> = e / row_sum_b;
        softmax_out.store(sm);

        // Gather softmax[targets[r]] per row via iota-equals-target mask.
        let col_idx: Tile<i32, { [V] }> = iota(const_shape![V]);
        let col_idx_b: Tile<i32, { [BM, V] }> = col_idx
            .reshape(const_shape![1, V])
            .broadcast(const_shape![BM, V]);
        let tgt_b: Tile<i32, { [BM, V] }> = tt
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, V]);
        let hit: Tile<bool, { [BM, V] }> = eq_tile(col_idx_b, tgt_b);
        let zero: Tile<f32, { [BM, V] }> = constant(0.0f32, const_shape![BM, V]);
        let picked: Tile<f32, { [BM, V] }> = select(hit, sm, zero);
        let picked_row: Tile<f32, { [BM] }> = reduce_sum(picked, 1i32);

        let eps: Tile<f32, { [BM] }> = constant(1.0e-9f32, losses.shape());
        let zero1: Tile<f32, { [BM] }> = constant(0.0f32, losses.shape());
        let logp: Tile<f32, { [BM] }> = log(picked_row + eps);
        losses.store(zero1 - logp);
    }

    /// Per row:
    ///   `dlogits[r, j] = (softmax_out[r, j] - 1[j == targets[r]]) * grad_scale`.
    #[cutile::entry()]
    pub fn cross_entropy_backward<const BM: i32, const V: i32>(
        dlogits: &mut Tensor<f32, { [BM, V] }>,
        softmax_out: &Tensor<f32, { [-1, -1] }>,
        targets: &Tensor<i32, { [-1] }>,
        grad_scale: f32,
    ) {
        let tsm: Tile<f32, { [BM, V] }> = load_tile_like_2d(softmax_out, dlogits);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tt_part: Partition<i32, { [BM] }> = targets.partition(const_shape![BM]);
        let tt: Tile<i32, { [BM] }> = tt_part.load([pid.0]);

        let col_idx: Tile<i32, { [V] }> = iota(const_shape![V]);
        let col_idx_b: Tile<i32, { [BM, V] }> = col_idx
            .reshape(const_shape![1, V])
            .broadcast(const_shape![BM, V]);
        let tgt_b: Tile<i32, { [BM, V] }> = tt
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, V]);
        let hit: Tile<bool, { [BM, V] }> = eq_tile(col_idx_b, tgt_b);
        let one: Tile<f32, { [BM, V] }> = constant(1.0f32, const_shape![BM, V]);
        let zero: Tile<f32, { [BM, V] }> = constant(0.0f32, const_shape![BM, V]);
        let indicator: Tile<f32, { [BM, V] }> = select(hit, one, zero);
        let scale: Tile<f32, { [BM, V] }> = grad_scale.broadcast(const_shape![BM, V]);
        dlogits.store((tsm - indicator) * scale);
    }
}

pub use cross_entropy_kernels::{cross_entropy_backward, cross_entropy_forward};
