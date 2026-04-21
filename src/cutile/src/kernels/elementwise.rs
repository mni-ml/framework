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
}

pub use elementwise_kernels::{
    add, add_bias, broadcast_add, broadcast_mul, copy, div, div_backward_a, div_backward_b, eq,
    exp_f32, fill, gt, is_close, log_f32, lt, mul, mul_scalar, neg, pow_backward_f32, pow_f32,
    saxpy, sub,
};
