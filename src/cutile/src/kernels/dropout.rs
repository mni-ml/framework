//! Dropout cuTile kernels — one-for-one port of `native/kernels/dropout.cu`.

#[cutile::module]
pub mod dropout_kernels {
    use cutile::core::*;

    /// `out[i] = x[i] · mask[i] · scale` (scale = `1/(1-p)`).
    #[cutile::entry()]
    pub fn dropout_apply<const S: [i32; 1]>(
        out: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        mask: &Tensor<f32, { [-1] }>,
        scale: f32,
    ) {
        let tx = load_tile_like_1d(x, out);
        let tm = load_tile_like_1d(mask, out);
        let s: Tile<f32, S> = scale.broadcast(out.shape());
        out.store(tx * tm * s);
    }

    /// `dx[i] = dy[i] · mask[i] · scale`.
    #[cutile::entry()]
    pub fn dropout_backward<const S: [i32; 1]>(
        dx: &mut Tensor<f32, S>,
        dy: &Tensor<f32, { [-1] }>,
        mask: &Tensor<f32, { [-1] }>,
        scale: f32,
    ) {
        let tdy = load_tile_like_1d(dy, dx);
        let tm = load_tile_like_1d(mask, dx);
        let s: Tile<f32, S> = scale.broadcast(dx.shape());
        dx.store(tdy * tm * s);
    }
}

pub use dropout_kernels::{dropout_apply, dropout_backward};
