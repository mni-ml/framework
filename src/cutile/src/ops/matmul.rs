//! Matmul backed by the cuTile GEMM kernel.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::PartitionMut;
use cutile::tile_kernel::TileKernel;

/// Largest block candidate that divides `n` (>= 1).
fn largest_divisor(n: usize, candidates: &[usize]) -> usize {
    for &c in candidates {
        if n % c == 0 {
            return c;
        }
    }
    1
}

/// 2D matmul: z[M, N] = x[M, K] @ y[K, N].  Returns a new TensorId.
///
/// If either operand has rank > 2, the leading dimensions are collapsed into a
/// single batch M, and the output shape is [M_original_leading..., N].  This
/// matches the batched matmul convention in the mni framework, where a linear
/// layer calls `matmul(x_flat, w.T)` on a reshaped [*, in_features] input.
pub fn matmul(store: &mut TensorStore, a: TensorId, b: TensorId) -> TensorId {
    let a_shape = store.shape(a).to_vec();
    let b_shape = store.shape(b).to_vec();
    assert!(
        a_shape.len() >= 2 && b_shape.len() == 2,
        "matmul: expected a.rank >= 2 and b.rank == 2, got {:?} @ {:?}",
        a_shape,
        b_shape,
    );

    let k_a = *a_shape.last().unwrap();
    let (k_b, n) = (b_shape[0], b_shape[1]);
    assert_eq!(k_a, k_b, "matmul: inner dim mismatch {} vs {}", k_a, k_b);

    let m: usize = a_shape[..a_shape.len() - 1].iter().product();
    let k = k_a;

    // Choose tile dims.  Prefer 32 / 32 / 16 if shapes allow.
    let bm = largest_divisor(m, &[64, 32, 16, 8, 4, 2, 1]);
    let bn = largest_divisor(n, &[64, 32, 16, 8, 4, 2, 1]);
    let bk = largest_divisor(k, &[32, 16, 8, 4, 2, 1]);

    let rt = runtime();

    // Allocate output [M, N].
    let mut z = api::zeros::<f32>(&[m, n]).sync_on(&rt.stream).expect("alloc z");

    {
        let x_tensor = store.tensor(a);
        let y_tensor = store.tensor(b);
        // Flatten x to [M, K] if it was higher rank.
        let x_view = x_tensor.view(&[m, k]).expect("view x");
        let y_view = y_tensor.view(&[k, n]).expect("view y");
        let generics = vec![
            bm.to_string(),
            bn.to_string(),
            bk.to_string(),
            k.to_string(),
        ];
        let _ = kernels::gemm((&mut z).partition([bm, bn]), &x_view, &y_view)
            .generics(generics)
            .sync_on(&rt.stream)
            .expect("gemm kernel");
    }

    let out_shape: Vec<usize> = a_shape[..a_shape.len() - 1]
        .iter()
        .copied()
        .chain(std::iter::once(n))
        .collect();
    let out_tensor = if out_shape.len() == 2 {
        z
    } else {
        use cutile::tensor::Reshape;
        z.reshape(&out_shape).expect("reshape")
    };
    store.insert_tensor(out_tensor, out_shape)
}
