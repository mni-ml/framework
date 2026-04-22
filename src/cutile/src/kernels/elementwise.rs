//! Elementwise cuTile kernels — one-for-one port of `native/kernels/elementwise.cu`.
//!
//! All tensor kernels flatten to 1D at launch time (rank 1), so the elementwise
//! kernels compile once for `Tensor<f32, {[-1]}>` regardless of the caller's
//! logical shape.  Kernels that broadcast a 1D bias across the last dimension
//! of a 2D tensor (`add_bias`, `broadcast_add`, `broadcast_mul`) are written
//! rank-2 so the bias loads as a single tile column per block.

#[cutile::module]
pub mod elementwise_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        z.store(tx + ty);
    }

    #[cutile::entry()]
    pub fn sub<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        z.store(tx - ty);
    }

    #[cutile::entry()]
    pub fn mul<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        z.store(tx * ty);
    }

    #[cutile::entry()]
    pub fn div<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        z.store(tx / ty);
    }

    #[cutile::entry()]
    pub fn neg<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tx = load_tile_like_1d(x, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        z.store(zero - tx);
    }

    #[cutile::entry()]
    pub fn mul_scalar<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        s: f32,
    ) {
        let tx = load_tile_like_1d(x, z);
        let s_tile = s.broadcast(z.shape());
        z.store(s_tile * tx);
    }

    /// Fused saxpy: `z = a·x + y`.  Two GMEM reads, one FMA, one GMEM write.
    #[cutile::entry()]
    pub fn saxpy<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        a: f32,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        let a_tile = a.broadcast(z.shape());
        z.store(a_tile * tx + ty);
    }

    #[cutile::entry()]
    pub fn exp_f32<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tx = load_tile_like_1d(x, z);
        let r: Tile<f32, S> = exp(tx);
        z.store(r);
    }

    #[cutile::entry()]
    pub fn log_f32<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tx = load_tile_like_1d(x, z);
        let r: Tile<f32, S> = log(tx);
        z.store(r);
    }

    #[cutile::entry()]
    pub fn fill<const S: [i32; 1]>(z: &mut Tensor<f32, S>, val: f32) {
        z.store(val.broadcast(z.shape()));
    }

    #[cutile::entry()]
    pub fn copy<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tx = load_tile_like_1d(x, z);
        z.store(tx);
    }

    #[cutile::entry()]
    pub fn pow_f32<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        exponent: f32,
    ) {
        let tx = load_tile_like_1d(x, z);
        let e = exponent.broadcast(z.shape());
        let r: Tile<f32, S> = pow(tx, e);
        z.store(r);
    }

    /// `dx = dy · e · x^(e-1)`.
    #[cutile::entry()]
    pub fn pow_backward_f32<const S: [i32; 1]>(
        dx: &mut Tensor<f32, S>,
        dy: &Tensor<f32, { [-1] }>,
        x: &Tensor<f32, { [-1] }>,
        exponent: f32,
    ) {
        let tdy = load_tile_like_1d(dy, dx);
        let tx = load_tile_like_1d(x, dx);
        let em1_scalar: f32 = exponent - 1.0f32;
        let e: Tile<f32, S> = exponent.broadcast(dx.shape());
        let em1: Tile<f32, S> = em1_scalar.broadcast(dx.shape());
        let p: Tile<f32, S> = pow(tx, em1);
        dx.store(tdy * e * p);
    }

    /// `da = dy / b`.
    #[cutile::entry()]
    pub fn div_backward_a<const S: [i32; 1]>(
        da: &mut Tensor<f32, S>,
        dy: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let tdy = load_tile_like_1d(dy, da);
        let tb = load_tile_like_1d(b, da);
        da.store(tdy / tb);
    }

    /// `db = -dy · a / b²`.
    #[cutile::entry()]
    pub fn div_backward_b<const S: [i32; 1]>(
        db: &mut Tensor<f32, S>,
        dy: &Tensor<f32, { [-1] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let tdy = load_tile_like_1d(dy, db);
        let ta = load_tile_like_1d(a, db);
        let tb = load_tile_like_1d(b, db);
        let zero: Tile<f32, S> = constant(0.0f32, db.shape());
        db.store((zero - tdy) * ta / (tb * tb));
    }

    /// Elementwise `a < b`, producing 1.0 / 0.0 tile (matches CUDA's `ternary -> float`).
    #[cutile::entry()]
    pub fn lt<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        let mask: Tile<bool, S> = lt_tile(tx, ty);
        z.store(select(mask, one, zero));
    }

    #[cutile::entry()]
    pub fn gt<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        let mask: Tile<bool, S> = gt_tile(tx, ty);
        z.store(select(mask, one, zero));
    }

    /// `|a - b| < 1e-6 ? 1 : 0`.  Matches `eq_f32` in the CUDA backend, which
    /// tests near-equality rather than bitwise equality.
    #[cutile::entry()]
    pub fn eq<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        let tol: Tile<f32, S> = constant(1e-6f32, z.shape());
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        let diff: Tile<f32, S> = absf(tx - ty);
        let mask: Tile<bool, S> = lt_tile(diff, tol);
        z.store(select(mask, one, zero));
    }

    /// `|a - b| < tol ? 1 : 0`.
    #[cutile::entry()]
    pub fn is_close<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
        tol: f32,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        let t: Tile<f32, S> = tol.broadcast(z.shape());
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        let diff: Tile<f32, S> = absf(tx - ty);
        let mask: Tile<bool, S> = lt_tile(diff, t);
        z.store(select(mask, one, zero));
    }

    /// Fused `z = x + bias[j]` where `x : [N, C]` and `bias : [C]`.
    /// The bias broadcasts along the outer dimension of `x`.
    #[cutile::entry()]
    pub fn add_bias<const BN: i32, const C: i32>(
        z: &mut Tensor<f32, { [BN, C] }>,
        x: &Tensor<f32, { [-1, C] }>,
        bias: &Tensor<f32, { [C] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let x_part = x.partition(const_shape![BN, C]);
        let bias_part = bias.partition(const_shape![C]);
        let tx: Tile<f32, { [BN, C] }> = x_part.load([pid.0, 0i32]);
        let tb: Tile<f32, { [C] }> = bias_part.load([0i32]);
        let tb2: Tile<f32, { [BN, C] }> = tb.reshape(const_shape![1, C]).broadcast(const_shape![BN, C]);
        z.store(tx + tb2);
    }

    /// `z = x + y` where `y` broadcasts by repetition across `x` along the last
    /// dim — `x : [N, C]`, `y : [C]`, `out : [N, C]`.  Mirrors
    /// `broadcast_add_f32` in the CUDA backend.
    #[cutile::entry()]
    pub fn broadcast_add<const BN: i32, const C: i32>(
        z: &mut Tensor<f32, { [BN, C] }>,
        x: &Tensor<f32, { [-1, C] }>,
        y: &Tensor<f32, { [C] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let x_part = x.partition(const_shape![BN, C]);
        let y_part = y.partition(const_shape![C]);
        let tx: Tile<f32, { [BN, C] }> = x_part.load([pid.0, 0i32]);
        let ty: Tile<f32, { [C] }> = y_part.load([0i32]);
        let ty2: Tile<f32, { [BN, C] }> = ty.reshape(const_shape![1, C]).broadcast(const_shape![BN, C]);
        z.store(tx + ty2);
    }

    /// `z = x * y` where `y` broadcasts by repetition.  See `broadcast_add`.
    #[cutile::entry()]
    pub fn broadcast_mul<const BN: i32, const C: i32>(
        z: &mut Tensor<f32, { [BN, C] }>,
        x: &Tensor<f32, { [-1, C] }>,
        y: &Tensor<f32, { [C] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let x_part = x.partition(const_shape![BN, C]);
        let y_part = y.partition(const_shape![C]);
        let tx: Tile<f32, { [BN, C] }> = x_part.load([pid.0, 0i32]);
        let ty: Tile<f32, { [C] }> = y_part.load([0i32]);
        let ty2: Tile<f32, { [BN, C] }> = ty.reshape(const_shape![1, C]).broadcast(const_shape![BN, C]);
        z.store(tx * ty2);
    }

    /// Generic up-to-4D permute via runtime stride arithmetic — port of
    /// `permute_f32` in the CUDA backend.  Per output element:
    ///
    /// ```text
    ///   rem = i
    ///   src_i = 0
    ///   for axis ∈ 0..ndim:
    ///       c = rem / ds[axis]
    ///       rem = rem % ds[axis]
    ///       src_i += c * es[axis]
    ///   out[i] = src[src_i]
    /// ```
    ///
    /// `ds[a]` is the dest-side "block divisor" for axis `a` (so that
    /// splitting `i` by successive divisors recovers the permuted
    /// output coordinates), and `es[a]` is the source-side flat stride
    /// for the source axis that maps to the dest's `a`-th axis.
    ///
    /// The `ndim >= k` branches are on a runtime `i32`, but all
    /// divisors / strides for unused levels are passed as zero so the
    /// unused branches produce no effect — matching the CUDA version's
    /// pattern of unused `ds_i`/`es_i` parameters.  We keep the
    /// branches explicit (rather than folding all four unconditionally)
    /// so unused levels don't generate division-by-zero lanes.
    #[cutile::entry()]
    pub unsafe fn permute_runtime<const BLOCK: i32>(
        out_ptr: *mut f32,
        src_ptr: *mut f32,
        n: i32,
        ds0: i32,
        ds1: i32,
        ds2: i32,
        _ds3: i32,
        es0: i32,
        es1: i32,
        es2: i32,
        es3: i32,
        ndim: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let base: i32 = pid.0 * BLOCK;
        let base_t: Tile<i32, { [BLOCK] }> = base.broadcast(const_shape![BLOCK]);
        let iota_t: Tile<i32, { [BLOCK] }> = iota(const_shape![BLOCK]);
        let idx: Tile<i32, { [BLOCK] }> = base_t + iota_t;

        let n_t: Tile<i32, { [BLOCK] }> = n.broadcast(const_shape![BLOCK]);
        let valid: Tile<bool, { [BLOCK] }> = lt_tile(idx, n_t);

        let mut rem: Tile<i32, { [BLOCK] }> = idx;
        let mut src_idx: Tile<i32, { [BLOCK] }> = constant(0i32, const_shape![BLOCK]);

        if ndim >= 1i32 {
            let ds0_t: Tile<i32, { [BLOCK] }> = ds0.broadcast(const_shape![BLOCK]);
            let es0_t: Tile<i32, { [BLOCK] }> = es0.broadcast(const_shape![BLOCK]);
            let c: Tile<i32, { [BLOCK] }> = rem / ds0_t;
            rem = rem % ds0_t;
            src_idx = src_idx + c * es0_t;
        }
        if ndim >= 2i32 {
            let ds1_t: Tile<i32, { [BLOCK] }> = ds1.broadcast(const_shape![BLOCK]);
            let es1_t: Tile<i32, { [BLOCK] }> = es1.broadcast(const_shape![BLOCK]);
            let c: Tile<i32, { [BLOCK] }> = rem / ds1_t;
            rem = rem % ds1_t;
            src_idx = src_idx + c * es1_t;
        }
        if ndim >= 3i32 {
            let ds2_t: Tile<i32, { [BLOCK] }> = ds2.broadcast(const_shape![BLOCK]);
            let es2_t: Tile<i32, { [BLOCK] }> = es2.broadcast(const_shape![BLOCK]);
            let c: Tile<i32, { [BLOCK] }> = rem / ds2_t;
            rem = rem % ds2_t;
            src_idx = src_idx + c * es2_t;
        }
        if ndim >= 4i32 {
            let es3_t: Tile<i32, { [BLOCK] }> = es3.broadcast(const_shape![BLOCK]);
            src_idx = src_idx + rem * es3_t;
        }

        let src_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(src_ptr);
        let src_base_bl: PointerTile<*mut f32, { [BLOCK] }> = src_base
            .reshape(const_shape![1])
            .broadcast(const_shape![BLOCK]);
        let src_ptrs: PointerTile<*mut f32, { [BLOCK] }> = src_base_bl.offset_tile(src_idx);
        let (src_vals, _ltok): (Tile<f32, { [BLOCK] }>, Token) = load_ptr_tko(
            src_ptrs,
            "relaxed",
            "device",
            Some(valid),
            Some(0.0f32),
            None,
            None,
        );

        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_base_bl: PointerTile<*mut f32, { [BLOCK] }> = out_base
            .reshape(const_shape![1])
            .broadcast(const_shape![BLOCK]);
        let out_ptrs: PointerTile<*mut f32, { [BLOCK] }> = out_base_bl.offset_tile(idx);
        let _stok: Token =
            store_ptr_tko(out_ptrs, src_vals, "relaxed", "device", Some(valid), None, None);
    }
}

pub use elementwise_kernels::{
    add, add_bias, broadcast_add, broadcast_mul, copy, div, div_backward_a, div_backward_b, eq,
    exp_f32, fill, gt, is_close, log_f32, lt, mul, mul_scalar, neg, permute_runtime,
    pow_backward_f32, pow_f32, saxpy, sub,
};
