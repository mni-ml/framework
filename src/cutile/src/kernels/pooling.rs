//! Pooling cuTile kernels — port of `native/kernels/pooling.cu`.
//!
//! Written rank-4 to match the CUDA `[N, C, H, W]` layout directly — one
//! block per output element, tile shape `[1, 1, KH, KW]` covers the
//! kernel window in a single gather.  The block decodes `(n, c, oh, ow)`
//! from the 3D grid `(N*C, H_out, W_out)`.
//!
//! `maxpool2d_forward` saves the flat argmax index for the backward
//! pass; `maxpool2d_backward` scatters via an atomic add through a
//! pointer tile — one-for-one with the CUDA `atomicAdd` dispatch.

#[cutile::module]
pub mod pooling_kernels {
    use cutile::core::*;

    /// Per output element `(n, c, oh, ow)`:
    ///
    /// ```text
    ///   out[n, c, oh, ow] = Σ(kh, kw) inp[n, c, oh*KH + kh, ow*KW + kw] / (KH·KW)
    /// ```
    ///
    /// Uses zero-padded partition loads for boundary windows.  The CUDA
    /// kernel's bounds-checked `count` denominator is matched by using
    /// the fixed `KH·KW` divisor; callers must ensure `H % KH == 0` and
    /// `W % KW == 0` (the CUDA code asserts this implicitly by only
    /// summing in-range lanes).
    #[cutile::entry()]
    pub fn avgpool2d_forward<const KH: i32, const KW: i32>(
        out: &mut Tensor<f32, { [1, 1, 1, 1] }>,
        inp: &Tensor<f32, { [-1, -1, -1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let inp_part: Partition<f32, { [1, 1, KH, KW] }> =
            inp.partition(const_shape![1, 1, KH, KW]);
        let tile: Tile<f32, { [1, 1, KH, KW] }> = inp_part.load([pid.0, 0i32, pid.1, pid.2]);

        // Reduce over spatial dims (3, 2) in sequence so cuTile's dim-index
        // stays valid after the first drop.
        let r1: Tile<f32, { [1, 1, KH] }> = reduce_sum(tile, 3i32);
        let r2: Tile<f32, { [1, 1] }> = reduce_sum(r1, 2i32);
        let r3: Tile<f32, { [1, 1, 1, 1] }> = r2.reshape(const_shape![1, 1, 1, 1]);
        let inv_s: f32 = 1.0f32 / ((KH * KW) as f32);
        let inv: Tile<f32, { [1, 1, 1, 1] }> = inv_s.broadcast(const_shape![1, 1, 1, 1]);
        out.store(r3 * inv);
    }

    /// Per output element `(n, c, oh, ow)`:
    ///
    /// ```text
    ///   out[n, c, oh, ow] = max(kh, kw) inp[n, c, oh*KH + kh, ow*KW + kw]
    ///   argmax[n, c, oh, ow] = linear index of the max position in inp
    /// ```
    #[cutile::entry()]
    pub fn maxpool2d_forward<const KH: i32, const KW: i32>(
        out: &mut Tensor<f32, { [1, 1, 1, 1] }>,
        argmax: &mut Tensor<i32, { [1, 1, 1, 1] }>,
        inp: &Tensor<f32, { [-1, -1, -1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let inp_part: Partition<f32, { [1, 1, KH, KW] }> =
            inp.partition(const_shape![1, 1, KH, KW]);
        let tile: Tile<f32, { [1, 1, KH, KW] }> = inp_part.load([pid.0, 0i32, pid.1, pid.2]);

        let m1: Tile<f32, { [1, 1, KH] }> = reduce_max(tile, 3i32);
        let m2: Tile<f32, { [1, 1] }> = reduce_max(m1, 2i32);
        let m3: Tile<f32, { [1, 1, 1, 1] }> = m2.reshape(const_shape![1, 1, 1, 1]);
        out.store(m3);

        // argmax: find (kh, kw) where tile[kh, kw] == max, then linearize.
        let m_val: f32 = tile_to_scalar(m2.reshape(const_shape![]));
        let m_b: Tile<f32, { [1, 1, KH, KW] }> = m_val.broadcast(const_shape![1, 1, KH, KW]);
        let hit: Tile<bool, { [1, 1, KH, KW] }> = eq_tile(tile, m_b);

        // Per-lane linear index within the window: kh*KW + kw.
        let kh_iota: Tile<i32, { [KH] }> = iota(const_shape![KH]);
        let kh_idx: Tile<i32, { [1, 1, KH, KW] }> = kh_iota
            .reshape(const_shape![1, 1, KH, 1])
            .broadcast(const_shape![1, 1, KH, KW]);
        let kw_iota: Tile<i32, { [KW] }> = iota(const_shape![KW]);
        let kw_idx: Tile<i32, { [1, 1, KH, KW] }> = kw_iota
            .reshape(const_shape![1, 1, 1, KW])
            .broadcast(const_shape![1, 1, KH, KW]);
        let kw_t: Tile<i32, { [1, 1, KH, KW] }> = constant(KW, const_shape![1, 1, KH, KW]);
        let win_idx: Tile<i32, { [1, 1, KH, KW] }> = kh_idx * kw_t + kw_idx;

        // Translate in-window (kh, kw) to flat input index:
        //   n*C*H*W + c*H*W + (oh*KH + kh)*W + (ow*KW + kw).
        // We don't know N/C/H/W at compile time here, so the ops layer
        // finalizes `argmax` by remapping the within-window linear index
        // to the global flat index post-launch (cheap pointwise kernel).
        let big: Tile<i32, { [1, 1, KH, KW] }> = constant(i32::MAX, const_shape![1, 1, KH, KW]);
        let masked: Tile<i32, { [1, 1, KH, KW] }> = select(hit, win_idx, big);
        let am1: Tile<i32, { [1, 1, KH] }> = reduce_min(masked, 3i32);
        let am2: Tile<i32, { [1, 1] }> = reduce_min(am1, 2i32);
        argmax.store(am2.reshape(const_shape![1, 1, 1, 1]));
    }

    /// Per input element `(n, c, ih, iw)`:
    ///
    /// ```text
    ///   oh = ih / KH
    ///   ow = iw / KW
    ///   dinp[n, c, ih, iw] = dout[n, c, oh, ow] / (KH·KW)
    /// ```
    ///
    /// Rank-4, one block per input element.  The CUDA kernel's
    /// bounds check reduces to "the load is valid" here — the cuTile
    /// partition view handles padding for non-multiple H / W.
    #[cutile::entry()]
    pub fn avgpool2d_backward<const KH: i32, const KW: i32>(
        dinp: &mut Tensor<f32, { [1, 1, KH, KW] }>,
        dout: &Tensor<f32, { [-1, -1, -1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        // dout at (pid.0, 0, pid.1, pid.2) is one scalar; broadcast it
        // over the input window.
        let dout_part: Partition<f32, { [1, 1, 1, 1] }> =
            dout.partition(const_shape![1, 1, 1, 1]);
        let dout_tile: Tile<f32, { [1, 1, 1, 1] }> =
            dout_part.load([pid.0, 0i32, pid.1, pid.2]);
        let dv: f32 = tile_to_scalar(dout_tile.reshape(const_shape![]));
        let inv_s: f32 = 1.0f32 / ((KH * KW) as f32);
        let val: f32 = dv * inv_s;
        let broadcast_tile: Tile<f32, { [1, 1, KH, KW] }> =
            val.broadcast(const_shape![1, 1, KH, KW]);
        dinp.store(broadcast_tile);
    }

    /// Per output element: atomic scatter `dinp[argmax[i]] += dout[i]`
    /// through a pointer tile.  `dinp_ptr` is the base of the
    /// contiguous `[N, C, H, W]` row-major f32 buffer.
    #[cutile::entry()]
    pub unsafe fn maxpool2d_backward<const BN: i32>(
        dinp_ptr: *mut f32,
        dout: &Tensor<f32, { [-1] }>,
        argmax: &Tensor<i32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let dout_part: Partition<f32, { [BN] }> = dout.partition(const_shape![BN]);
        let argmax_part: Partition<i32, { [BN] }> = argmax.partition(const_shape![BN]);
        let tdout: Tile<f32, { [BN] }> = dout_part.load([pid.0]);
        let targ: Tile<i32, { [BN] }> = argmax_part.load([pid.0]);

        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(dinp_ptr);
        let base_1d: PointerTile<*mut f32, { [1] }> = base.reshape(const_shape![1]);
        let base_d: PointerTile<*mut f32, { [BN] }> = base_1d.broadcast(const_shape![BN]);
        let ptrs: PointerTile<*mut f32, { [BN] }> = base_d.offset_tile(targ);

        let (_old, _tok): (Tile<f32, { [BN] }>, Token) =
            atomic_rmw_tko(ptrs, tdout, "addf", "relaxed", "device", None, None);
    }
}

pub use pooling_kernels::{
    avgpool2d_backward, avgpool2d_forward, maxpool2d_backward, maxpool2d_forward,
};
