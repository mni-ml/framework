//! GEMM cuTile kernel.

#[cutile::module]
pub mod matmul_kernels {
    use cutile::core::*;

    /// `z [M, N] = x [M, K] @ y [K, N]`, tiled `BM×BN` with reduction over `BK`.
    /// `K` is a compile-time constant so the inner loop is bounded — every
    /// distinct `(BM, BN, BK, K)` triggers one PTX recompile, cached after.
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

pub use matmul_kernels::gemm;
