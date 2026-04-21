//! Activation cuTile kernels — one-for-one port of `native/kernels/activation.cu`.

#[cutile::module]
pub mod activation_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn relu<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tx = load_tile_like_1d(x, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        z.store(max_tile(zero, tx));
    }

    /// `dx = grad where x > 0, else 0`.
    #[cutile::entry()]
    pub fn relu_backward<const S: [i32; 1]>(
        dx: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        grad: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, dx);
        let tg = load_tile_like_1d(grad, dx);
        let zero: Tile<f32, S> = constant(0.0f32, dx.shape());
        let pos: Tile<bool, S> = gt_tile(tx, zero);
        dx.store(select(pos, tg, zero));
    }

    /// GELU (tanh approx): `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
    #[cutile::entry()]
    pub fn gelu_forward<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let half: Tile<f32, S> = constant(0.5f32, z.shape());
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let k0: Tile<f32, S> = constant(0.7978845608f32, z.shape()); // √(2/π)
        let k1: Tile<f32, S> = constant(0.044715f32, z.shape());
        let x2: Tile<f32, S> = tx * tx;
        let x3: Tile<f32, S> = x2 * tx;
        let inner: Tile<f32, S> = k0 * (tx + k1 * x3);
        let th: Tile<f32, S> = tanh(inner);
        z.store(half * tx * (one + th));
    }

    /// GELU backward (derivative of the tanh approximation).
    #[cutile::entry()]
    pub fn gelu_backward<const S: [i32; 1]>(
        dx: &mut Tensor<f32, S>,
        dy: &Tensor<f32, { [-1] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let tdy = load_tile_like_1d(dy, dx);
        let tx = load_tile_like_1d(x, dx);
        let half: Tile<f32, S> = constant(0.5f32, dx.shape());
        let one: Tile<f32, S> = constant(1.0f32, dx.shape());
        let k0: Tile<f32, S> = constant(0.7978845608f32, dx.shape());
        let k1: Tile<f32, S> = constant(0.044715f32, dx.shape());
        let k1_3: Tile<f32, S> = constant(0.134145f32, dx.shape()); // 3 · 0.044715
        let x2: Tile<f32, S> = tx * tx;
        let x3: Tile<f32, S> = x2 * tx;
        let inner: Tile<f32, S> = k0 * (tx + k1 * x3);
        let th: Tile<f32, S> = tanh(inner);
        let sech2: Tile<f32, S> = one - th * th;
        let d_inner: Tile<f32, S> = k0 * (one + k1_3 * x2);
        let grad: Tile<f32, S> = half * (one + th) + half * tx * sech2 * d_inner;
        dx.store(tdy * grad);
    }

    /// `σ(x) = 1 / (1 + e^-x)`.
    #[cutile::entry()]
    pub fn sigmoid_forward<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        let neg_x: Tile<f32, S> = zero - tx;
        let e: Tile<f32, S> = exp(neg_x);
        z.store(one / (one + e));
    }

    /// `dx = dy · σ(x) · (1 - σ(x))`, using the saved forward output.
    #[cutile::entry()]
    pub fn sigmoid_backward<const S: [i32; 1]>(
        dx: &mut Tensor<f32, S>,
        dy: &Tensor<f32, { [-1] }>,
        out: &Tensor<f32, { [-1] }>,
    ) {
        let tdy = load_tile_like_1d(dy, dx);
        let to = load_tile_like_1d(out, dx);
        let one: Tile<f32, S> = constant(1.0f32, dx.shape());
        dx.store(tdy * to * (one - to));
    }
}

pub use activation_kernels::{
    gelu_backward, gelu_forward, relu, relu_backward, sigmoid_backward, sigmoid_forward,
};
