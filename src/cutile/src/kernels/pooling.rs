//! Pooling cuTile kernels — port of `native/kernels/pooling.cu`.
//!
//! The `[N, C, H, W]` layout is flattened to `[N*C, H, W]` on the ops side
//! because cuTile partition loads support rank ≤ 3 today — one block per
//! output element, tile shape `[1, KH, KW]` covers the kernel window in a
//! single gather.  The block decodes `(nc, oh, ow)` directly from the 3D
//! grid `(N*C, H_out, W_out)`; spatial indices stay intact, so conv-style
//! arithmetic still works unchanged.
//!
//! `maxpool2d_forward` saves the within-window argmax index for the
//! backward pass (the ops layer rewrites it to a global flat index before
//! the scatter), and `maxpool2d_backward` scatters via an atomic add
//! through a pointer tile — one-for-one with the CUDA `atomicAdd` dispatch.

#[cutile::module]
pub mod pooling_kernels {
    use cutile::core::*;

    /// Per output element `(nc, oh, ow)` on a flattened `[N*C, H, W]` tensor:
    ///
    /// ```text
    ///   out[nc, oh, ow] = Σ(kh, kw) inp[nc, oh*KH + kh, ow*KW + kw] / (KH·KW)
    /// ```
    ///
    /// Callers must ensure `H % KH == 0` and `W % KW == 0` (the CUDA code
    /// asserts this implicitly by only summing in-range lanes).
    #[cutile::entry()]
    pub fn avgpool2d_forward<const KH: i32, const KW: i32>(
        out: &mut Tensor<f32, { [1, 1, 1] }>,
        inp: &Tensor<f32, { [-1, -1, -1] }>,
        inv_kh_kw: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let inp_part: Partition<f32, { [1, KH, KW] }> =
            inp.partition(const_shape![1, KH, KW]);
        let tile: Tile<f32, { [1, KH, KW] }> = inp_part.load([pid.0, pid.1, pid.2]);

        // Reduce over spatial dims (2, 1) in sequence so cuTile's dim-index
        // stays valid after the first drop.
        let r1: Tile<f32, { [1, KH] }> = reduce_sum(tile, 2i32);
        let r2: Tile<f32, { [1] }> = reduce_sum(r1, 1i32);
        let r3: Tile<f32, { [1, 1, 1] }> = r2.reshape(const_shape![1, 1, 1]);
        let inv: Tile<f32, { [1, 1, 1] }> = inv_kh_kw.broadcast(const_shape![1, 1, 1]);
        out.store(r3 * inv);
    }

    /// Per output element `(nc, oh, ow)`:
    ///
    /// ```text
    ///   out[nc, oh, ow]    = max(kh, kw) inp[nc, oh*KH + kh, ow*KW + kw]
    ///   argmax[nc, oh, ow] = kh*KW + kw  (within-window)
    /// ```
    ///
    /// The ops layer finalizes `argmax` by remapping the within-window linear
    /// index to a global flat index post-launch.
    #[cutile::entry()]
    pub fn maxpool2d_forward<const KH: i32, const KW: i32>(
        out: &mut Tensor<f32, { [1, 1, 1] }>,
        argmax: &mut Tensor<i32, { [1, 1, 1] }>,
        inp: &Tensor<f32, { [-1, -1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let inp_part: Partition<f32, { [1, KH, KW] }> =
            inp.partition(const_shape![1, KH, KW]);
        let tile: Tile<f32, { [1, KH, KW] }> = inp_part.load([pid.0, pid.1, pid.2]);

        let m1: Tile<f32, { [1, KH] }> = reduce_max(tile, 2i32);
        let m2: Tile<f32, { [1] }> = reduce_max(m1, 1i32);
        let m3: Tile<f32, { [1, 1, 1] }> = m2.reshape(const_shape![1, 1, 1]);
        out.store(m3);

        // argmax: find (kh, kw) where tile[kh, kw] == max, then linearize.
        let m_val: f32 = tile_to_scalar(m2.reshape(const_shape![]));
        let m_b: Tile<f32, { [1, KH, KW] }> = m_val.broadcast(const_shape![1, KH, KW]);
        let hit: Tile<bool, { [1, KH, KW] }> = eq_tile(tile, m_b);

        // Per-lane linear index within the window: kh*KW + kw.
        let kh_iota: Tile<i32, { [KH] }> = iota(const_shape![KH]);
        let kh_idx: Tile<i32, { [1, KH, KW] }> = kh_iota
            .reshape(const_shape![1, KH, 1])
            .broadcast(const_shape![1, KH, KW]);
        let kw_iota: Tile<i32, { [KW] }> = iota(const_shape![KW]);
        let kw_idx: Tile<i32, { [1, KH, KW] }> = kw_iota
            .reshape(const_shape![1, 1, KW])
            .broadcast(const_shape![1, KH, KW]);
        let kw_val: i32 = KW;
        let kw_t: Tile<i32, { [1, KH, KW] }> = kw_val.broadcast(const_shape![1, KH, KW]);
        let win_idx: Tile<i32, { [1, KH, KW] }> = kh_idx * kw_t + kw_idx;

        let big: Tile<i32, { [1, KH, KW] }> =
            constant(0x7FFFFFFFi32, const_shape![1, KH, KW]);
        let masked: Tile<i32, { [1, KH, KW] }> = select(hit, win_idx, big);
        let am1: Tile<i32, { [1, KH] }> = reduce_min(masked, 2i32);
        let am2: Tile<i32, { [1] }> = reduce_min(am1, 1i32);
        argmax.store(am2.reshape(const_shape![1, 1, 1]));
    }

    /// Per input element `(nc, ih, iw)`:
    ///
    /// ```text
    ///   oh = ih / KH
    ///   ow = iw / KW
    ///   dinp[nc, ih, iw] = dout[nc, oh, ow] / (KH·KW)
    /// ```
    #[cutile::entry()]
    pub fn avgpool2d_backward<const KH: i32, const KW: i32>(
        dinp: &mut Tensor<f32, { [1, KH, KW] }>,
        dout: &Tensor<f32, { [-1, -1, -1] }>,
        inv_kh_kw: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let dout_part: Partition<f32, { [1, 1, 1] }> =
            dout.partition(const_shape![1, 1, 1]);
        let dout_tile: Tile<f32, { [1, 1, 1] }> =
            dout_part.load([pid.0, pid.1, pid.2]);
        let dv: f32 = tile_to_scalar(dout_tile.reshape(const_shape![]));
        let val: f32 = dv * inv_kh_kw;
        let broadcast_tile: Tile<f32, { [1, KH, KW] }> =
            val.broadcast(const_shape![1, KH, KW]);
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
