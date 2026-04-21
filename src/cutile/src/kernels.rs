//! cuTile Rust kernels used by the backend.
//!
//! Each `#[cutile::entry()]` function is compiled to PTX on first launch and
//! cached for subsequent launches.
//!
//! Elementwise ops always operate on flattened 1D tensors — the launch grid
//! is the partition count.  GEMM uses 2D tiling.

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    // ===== Elementwise (equal-shape, 1D flattened) =====

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

    // ===== Unary =====

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

    // Fused saxpy: z = a*x + y.  Shows fusion (two GMEM reads, one GMEM write,
    // one FMA) vs the two-kernel path (temporary buffer + add).
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

    // ===== Activations =====

    #[cutile::entry()]
    pub fn relu<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tx = load_tile_like_1d(x, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        z.store(max_tile(zero, tx));
    }

    // relu_backward: dx = grad where x > 0, else 0.
    #[cutile::entry()]
    pub fn relu_backward<const S: [i32; 1]>(
        dx: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        grad: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, dx);
        let tg = load_tile_like_1d(grad, dx);
        let zero: Tile<f32, S> = constant(0.0f32, dx.shape());
        let pos = gt_tile(tx, zero);
        dx.store(select(pos, tg, zero));
    }

    // ===== GEMM =====
    //
    // z [M, N] = x [M, K] @ y [K, N], tiled BM×BN with reduction over BK.

    #[cutile::entry()]
    pub fn gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = z.load();
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
            continue;
        }
        z.store(tile_z);
    }
}

pub use kernels::*;
