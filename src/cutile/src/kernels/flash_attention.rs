//! FlashAttention-2 forward + backward — port of
//! `native/kernels/flash_attention.cu`.
//!
//! Tile layout mirrors the cutile-rs `flash_attention.rs` example — one
//! block per query tile `[BM, D]`, iterating over `ceil_div(S, BN)`
//! key/value tiles of shape `[BN, D]` with online softmax.
//!
//! Unlike the example (which uses `exp2` with a rescaled `qk_scale /
//! ln2`), these kernels use `exp` in natural-log space throughout.
//! This costs one op per tile but lets us write `LSE = m_i + log(l_i)`
//! directly in the log units expected by the backward pass.
//!
//! Causal masking is applied as a tile-level mask based on
//! `row_idx < col_idx` rather than early loop termination — this keeps
//! the kernel body branch-free, matching the CUDA semantics
//! `col < row + 1`.
//!
//! Backward: per block = one query tile `[BM, D]`, iterating over K/V
//! tiles.  `dQ` is written directly (one block per row tile), while
//! `dK` / `dV` contributions are scattered with `atomic_rmw_tko "addf"`
//! — every K/V row potentially receives contributions from every Q row
//! tile.

#[cutile::module]
pub mod flash_attention_kernels {
    use cutile::core::*;

    /// Per query tile `[BM, D]` at `(bh, q_m_idx)`:
    ///
    /// ```text
    ///   S[i, j] = scale · Σ_d  Q[bh, row(i), d] · K[bh, col(j), d]
    ///   apply causal mask, then:
    ///   m_i ← max_j S[i, j]              (running max across K tiles)
    ///   l_i ← Σ_j exp(S[i, j] − m_i)     (running sum across K tiles)
    ///   out[i, d] = (Σ_j exp(S[i,j] − m_i) · V[bh, col(j), d]) / l_i
    ///   LSE[i]   = m_i + log(l_i)
    /// ```
    ///
    /// `causal == 1` masks positions where `col > row` to `-inf`
    /// before the softmax, matching the CUDA `col < row + 1` bound.
    #[cutile::entry()]
    pub fn flash_attention_forward<const BM: i32, const BN: i32, const D: i32>(
        out: &mut Tensor<f32, { [1, BM, D] }>,
        lse: &mut Tensor<f32, { [1, BM] }>,
        q: &Tensor<f32, { [-1, -1, -1] }>,
        k: &Tensor<f32, { [-1, -1, -1] }>,
        v: &Tensor<f32, { [-1, -1, -1] }>,
        scale: f32,
        causal: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let bh: i32 = pid.0;
        let q_m_idx: i32 = pid.1;

        let s_len: i32 = get_shape_dim(q.shape(), 1i32);
        let num_tiles: i32 = ceil_div(s_len, BN);

        let scale_t: Tile<f32, { [BM, BN] }> = scale.broadcast(const_shape![BM, BN]);

        // Running softmax statistics.
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(f32::NEG_INFINITY, const_shape![BM, 1]);
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        // Load Q tile.
        let q_part: Partition<f32, { [1, BM, D] }> = q.partition(const_shape![1, BM, D]);
        let tq_3d: Tile<f32, { [1, BM, D] }> = q_part.load([bh, q_m_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = tq_3d.reshape(const_shape![BM, D]);

        let k_part: Partition<f32, { [1, BN, D] }> = k.partition(const_shape![1, BN, D]);
        let v_part: Partition<f32, { [1, BN, D] }> = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        // Row index tile for causal mask: row = q_m_idx*BM + iota.
        let row_base: i32 = q_m_idx * BM;
        let row_base_t: Tile<i32, { [BM] }> = row_base.broadcast(const_shape![BM]);
        let row_iota: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let row_idx_1d: Tile<i32, { [BM] }> = row_base_t + row_iota;
        let row_idx: Tile<i32, { [BM, BN] }> = row_idx_1d
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, BN]);

        let neg_inf_mask: Tile<f32, { [BM, BN] }> =
            constant(f32::NEG_INFINITY, const_shape![BM, BN]);
        let zero_mask: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);

        for j in 0i32..num_tiles {
            let k_tile_3d: Tile<f32, { [1, BN, D] }> = k_part.load([bh, j, 0i32]);
            let k_tile: Tile<f32, { [BN, D] }> = k_tile_3d.reshape(const_shape![BN, D]);
            let k_tile_t: Tile<f32, { [D, BN] }> = permute(k_tile, transpose);

            let qk_init: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let qk_raw: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_t, qk_init);
            let mut qk: Tile<f32, { [BM, BN] }> = qk_raw * scale_t;

            if causal != 0i32 {
                let col_base: i32 = j * BN;
                let col_base_t: Tile<i32, { [BN] }> = col_base.broadcast(const_shape![BN]);
                let col_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
                let col_idx_1d: Tile<i32, { [BN] }> = col_base_t + col_iota;
                let col_idx: Tile<i32, { [BM, BN] }> = col_idx_1d
                    .reshape(const_shape![1, BN])
                    .broadcast(const_shape![BM, BN]);
                let keep: Tile<bool, { [BM, BN] }> = ge_tile(row_idx, col_idx);
                let mask_add: Tile<f32, { [BM, BN] }> = select(keep, zero_mask, neg_inf_mask);
                qk = qk + mask_add;
            }

            // Online softmax update.
            let qk_max_1d: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max_1d.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let m_ij_b: Tile<f32, { [BM, BN] }> = m_ij.broadcast(const_shape![BM, BN]);
            let qk_shift: Tile<f32, { [BM, BN] }> = qk - m_ij_b;
            let p: Tile<f32, { [BM, BN] }> = exp(qk_shift);

            let l_ij_1d: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij_1d.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp(m_i - m_ij);
            l_i = l_i * alpha + l_ij;

            let alpha_d: Tile<f32, { [BM, D] }> = alpha.broadcast(const_shape![BM, D]);
            acc = acc * alpha_d;

            let v_tile_3d: Tile<f32, { [1, BN, D] }> = v_part.load([bh, j, 0i32]);
            let v_tile: Tile<f32, { [BN, D] }> = v_tile_3d.reshape(const_shape![BN, D]);
            acc = mma(p, v_tile, acc);
            m_i = m_ij;
        }

        let l_i_d: Tile<f32, { [BM, D] }> = l_i.broadcast(const_shape![BM, D]);
        acc = true_div(acc, l_i_d);
        let acc_3d: Tile<f32, { [1, BM, D] }> = acc.reshape(const_shape![1, BM, D]);
        out.store(acc_3d);

        let log_li: Tile<f32, { [BM, 1] }> = log(l_i);
        let lse_val: Tile<f32, { [BM, 1] }> = m_i + log_li;
        let lse_2d: Tile<f32, { [1, BM] }> = lse_val.reshape(const_shape![1, BM]);
        lse.store(lse_2d);
    }

    /// Per query tile `[BM, D]` — computes `dQ[rows, :]` directly and
    /// atomically accumulates `dK`/`dV` contributions for every K/V row
    /// touched by this block.
    ///
    /// ```text
    ///   Di[i]     = Σ_d dO[row, d] · Out[row, d]
    ///   S[i, j]   = scale · Σ_d Q[row, d] · K[col, d]      (causal masked)
    ///   P[i, j]   = exp(S[i, j] − LSE[row])
    ///   dV[col, d] += Σ_i P[i, j] · dO[row, d]             (atomic)
    ///   dP[i, j]  = Σ_d dO[row, d] · V[col, d]
    ///   dS[i, j]  = P[i, j] · (dP[i, j] − Di[i]) · scale
    ///   dQ[row,d] = Σ_j dS[i, j] · K[col, d]               (direct store)
    ///   dK[col,d] += Σ_i dS[i, j] · Q[row, d]              (atomic)
    /// ```
    #[cutile::entry()]
    pub unsafe fn flash_attention_backward<const BM: i32, const BN: i32, const D: i32>(
        dq: &mut Tensor<f32, { [1, BM, D] }>,
        dk_ptr: *mut f32,
        dv_ptr: *mut f32,
        d_out: &Tensor<f32, { [-1, -1, -1] }>,
        q: &Tensor<f32, { [-1, -1, -1] }>,
        k: &Tensor<f32, { [-1, -1, -1] }>,
        v: &Tensor<f32, { [-1, -1, -1] }>,
        out: &Tensor<f32, { [-1, -1, -1] }>,
        lse: &Tensor<f32, { [-1, -1] }>,
        scale: f32,
        causal: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let bh: i32 = pid.0;
        let q_m_idx: i32 = pid.1;

        let s_len: i32 = get_shape_dim(q.shape(), 1i32);
        let num_tiles: i32 = ceil_div(s_len, BN);

        let scale_t_bmbn: Tile<f32, { [BM, BN] }> = scale.broadcast(const_shape![BM, BN]);

        // Load Q, dO, Out tiles (all [BM, D] at (bh, q_m_idx)).
        let q_part: Partition<f32, { [1, BM, D] }> = q.partition(const_shape![1, BM, D]);
        let tq: Tile<f32, { [BM, D] }> = q_part
            .load([bh, q_m_idx, 0i32])
            .reshape(const_shape![BM, D]);

        let do_part: Partition<f32, { [1, BM, D] }> = d_out.partition(const_shape![1, BM, D]);
        let tdo: Tile<f32, { [BM, D] }> = do_part
            .load([bh, q_m_idx, 0i32])
            .reshape(const_shape![BM, D]);

        let o_part: Partition<f32, { [1, BM, D] }> = out.partition(const_shape![1, BM, D]);
        let to: Tile<f32, { [BM, D] }> = o_part
            .load([bh, q_m_idx, 0i32])
            .reshape(const_shape![BM, D]);

        // Di[i] = Σ_d dO[i, d] · Out[i, d].
        let do_out: Tile<f32, { [BM, D] }> = tdo * to;
        let di_1d: Tile<f32, { [BM] }> = reduce_sum(do_out, 1i32);
        let di: Tile<f32, { [BM, 1] }> = di_1d.reshape(const_shape![BM, 1]);

        // Load LSE[bh, row_block] as [BM].
        let lse_part: Partition<f32, { [1, BM] }> = lse.partition(const_shape![1, BM]);
        let lse_2d: Tile<f32, { [1, BM] }> = lse_part.load([bh, q_m_idx]);
        let lse_col: Tile<f32, { [BM, 1] }> = lse_2d.reshape(const_shape![BM, 1]);

        let k_part: Partition<f32, { [1, BN, D] }> = k.partition(const_shape![1, BN, D]);
        let v_part: Partition<f32, { [1, BN, D] }> = v.partition(const_shape![1, BN, D]);
        let transpose_mn: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        let row_base: i32 = q_m_idx * BM;
        let row_base_t: Tile<i32, { [BM] }> = row_base.broadcast(const_shape![BM]);
        let row_iota: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let row_idx_1d: Tile<i32, { [BM] }> = row_base_t + row_iota;
        let row_idx: Tile<i32, { [BM, BN] }> = row_idx_1d
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, BN]);

        let neg_inf_mask: Tile<f32, { [BM, BN] }> =
            constant(f32::NEG_INFINITY, const_shape![BM, BN]);
        let zero_mask_bmbn: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);

        let mut dq_acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        // Base linear offsets into dK / dV for this (bh) slice.
        // dK[bh, col, d] is at offset bh*S*D + col*D + d.
        let bh_off: i32 = bh * s_len * D;

        for j in 0i32..num_tiles {
            let k_tile: Tile<f32, { [BN, D] }> = k_part
                .load([bh, j, 0i32])
                .reshape(const_shape![BN, D]);
            let v_tile: Tile<f32, { [BN, D] }> = v_part
                .load([bh, j, 0i32])
                .reshape(const_shape![BN, D]);

            // S = scale · Q @ K^T.
            let k_t_tile: Tile<f32, { [D, BN] }> = permute(k_tile, transpose_mn);
            let s_init: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let s_raw: Tile<f32, { [BM, BN] }> = mma(tq, k_t_tile, s_init);
            let mut s_scaled: Tile<f32, { [BM, BN] }> = s_raw * scale_t_bmbn;

            if causal != 0i32 {
                let col_base: i32 = j * BN;
                let col_base_t: Tile<i32, { [BN] }> = col_base.broadcast(const_shape![BN]);
                let col_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
                let col_idx_1d: Tile<i32, { [BN] }> = col_base_t + col_iota;
                let col_idx: Tile<i32, { [BM, BN] }> = col_idx_1d
                    .reshape(const_shape![1, BN])
                    .broadcast(const_shape![BM, BN]);
                let keep: Tile<bool, { [BM, BN] }> = ge_tile(row_idx, col_idx);
                let mask_add: Tile<f32, { [BM, BN] }> =
                    select(keep, zero_mask_bmbn, neg_inf_mask);
                s_scaled = s_scaled + mask_add;
            }

            // P = exp(S − LSE).
            let lse_b: Tile<f32, { [BM, BN] }> = lse_col.broadcast(const_shape![BM, BN]);
            let p: Tile<f32, { [BM, BN] }> = exp(s_scaled - lse_b);

            // dV_partial = P^T @ dO    [BN, D]   — atomically add into dV[bh, col_block, :].
            let p_t: Tile<f32, { [BN, BM] }> = permute(p, transpose_mn);
            let dv_init: Tile<f32, { [BN, D] }> = constant(0.0f32, const_shape![BN, D]);
            let dv_partial: Tile<f32, { [BN, D] }> = mma(p_t, tdo, dv_init);

            // dP = dO @ V^T   [BM, BN].
            let v_t_tile: Tile<f32, { [D, BN] }> = permute(v_tile, transpose_mn);
            let dp_init: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let dp: Tile<f32, { [BM, BN] }> = mma(tdo, v_t_tile, dp_init);

            // dS = P · (dP − Di) · scale.
            let di_b: Tile<f32, { [BM, BN] }> = di.broadcast(const_shape![BM, BN]);
            let ds: Tile<f32, { [BM, BN] }> = p * (dp - di_b) * scale_t_bmbn;

            // dQ += dS @ K   [BM, D].
            dq_acc = mma(ds, k_tile, dq_acc);

            // dK_partial = dS^T @ Q    [BN, D]    — atomically add into dK[bh, col_block, :].
            let ds_t: Tile<f32, { [BN, BM] }> = permute(ds, transpose_mn);
            let dk_init: Tile<f32, { [BN, D] }> = constant(0.0f32, const_shape![BN, D]);
            let dk_partial: Tile<f32, { [BN, D] }> = mma(ds_t, tq, dk_init);

            // Build pointer tile [BN, D] for dK / dV atomic scatter at (bh, col_block, :).
            let col_block_off: i32 = bh_off + j * BN * D;
            let col_block_t: Tile<i32, { [BN, D] }> =
                col_block_off.broadcast(const_shape![BN, D]);
            let bn_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
            let d_const_bn: i32 = D;
            let d_t_bn: Tile<i32, { [BN] }> = d_const_bn.broadcast(const_shape![BN]);
            let row_stride_1d: Tile<i32, { [BN] }> = bn_iota * d_t_bn;
            let row_stride: Tile<i32, { [BN, D] }> = row_stride_1d
                .reshape(const_shape![BN, 1])
                .broadcast(const_shape![BN, D]);
            let d_iota: Tile<i32, { [D] }> = iota(const_shape![D]);
            let d_off: Tile<i32, { [BN, D] }> = d_iota
                .reshape(const_shape![1, D])
                .broadcast(const_shape![BN, D]);
            let kv_offs: Tile<i32, { [BN, D] }> = col_block_t + row_stride + d_off;

            // dK scatter.  PointerTile reshape goes one rank at a time:
            // [] → [1] → [1, 1] → broadcast to [BN, D].  A direct [] → [1, 1]
            // reshape fails the cuTile verifier ("source/result must have same rank").
            let dk_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(dk_ptr);
            let dk_base_1: PointerTile<*mut f32, { [1] }> = dk_base.reshape(const_shape![1]);
            let dk_base_11: PointerTile<*mut f32, { [1, 1] }> =
                dk_base_1.reshape(const_shape![1, 1]);
            let dk_base_2d: PointerTile<*mut f32, { [BN, D] }> =
                dk_base_11.broadcast(const_shape![BN, D]);
            let dk_ptrs: PointerTile<*mut f32, { [BN, D] }> = dk_base_2d.offset_tile(kv_offs);

            // Tail mask: guard OOB when S is not a multiple of BN.
            let col_base_j: i32 = j * BN;
            let col_base_t_bn: Tile<i32, { [BN] }> = col_base_j.broadcast(const_shape![BN]);
            let col_idx_bn_1d: Tile<i32, { [BN] }> = col_base_t_bn + bn_iota;
            let s_t_bn: Tile<i32, { [BN] }> = s_len.broadcast(const_shape![BN]);
            let col_valid_1d: Tile<bool, { [BN] }> = lt_tile(col_idx_bn_1d, s_t_bn);
            let col_valid: Tile<bool, { [BN, D] }> = col_valid_1d
                .reshape(const_shape![BN, 1])
                .broadcast(const_shape![BN, D]);

            let (_odk, _tdk): (Tile<f32, { [BN, D] }>, Token) = atomic_rmw_tko(
                dk_ptrs,
                dk_partial,
                "addf",
                "relaxed",
                "device",
                Some(col_valid),
                None,
            );

            // dV scatter (same offsets, different base ptr).
            let dv_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(dv_ptr);
            let dv_base_1: PointerTile<*mut f32, { [1] }> = dv_base.reshape(const_shape![1]);
            let dv_base_11: PointerTile<*mut f32, { [1, 1] }> =
                dv_base_1.reshape(const_shape![1, 1]);
            let dv_base_2d: PointerTile<*mut f32, { [BN, D] }> =
                dv_base_11.broadcast(const_shape![BN, D]);
            let dv_ptrs: PointerTile<*mut f32, { [BN, D] }> = dv_base_2d.offset_tile(kv_offs);

            let (_odv, _tdv): (Tile<f32, { [BN, D] }>, Token) = atomic_rmw_tko(
                dv_ptrs,
                dv_partial,
                "addf",
                "relaxed",
                "device",
                Some(col_valid),
                None,
            );
        }

        let dq_3d: Tile<f32, { [1, BM, D] }> = dq_acc.reshape(const_shape![1, BM, D]);
        dq.store(dq_3d);
    }
}

pub use flash_attention_kernels::{flash_attention_backward, flash_attention_forward};
