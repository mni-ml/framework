//! Elementwise ops backed by cuTile kernels.
//!
//! All ops flatten the inputs/outputs to 1D and launch a 1D tiled kernel.
//! The block size is chosen to divide the element count so the kernel never
//! writes out of bounds.  Fused 2D shape ops (`add_bias`, `broadcast_*`)
//! launch with `[BN, C]` tiles so the broadcast across the row dim is
//! compiled as a single register move rather than an extra load.

use crate::device::runtime;
use crate::kernels;
use crate::tensor::{shape_size, TensorId, TensorStore};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{PartitionMut, Reshape};
use cutile::tile_kernel::TileKernel;

const CANDIDATE_BLOCKS: [usize; 9] = [256, 128, 64, 32, 16, 8, 4, 2, 1];

/// Largest block size from `CANDIDATE_BLOCKS` that divides `n`.
fn pick_block(n: usize) -> usize {
    for &b in &CANDIDATE_BLOCKS {
        if n % b == 0 {
            return b;
        }
    }
    1
}

// ---------------------------------------------------------------------------
// Binary ops: add / sub / mul / div
// ---------------------------------------------------------------------------

macro_rules! binary_op {
    ($name:ident, $kernel:ident, $label:literal) => {
        pub fn $name(store: &mut TensorStore, a: TensorId, b: TensorId) -> TensorId {
            let shape = store.shape(a).to_vec();
            let size = shape_size(&shape);
            assert_eq!(store.size(b), size, concat!($label, ": shape mismatch"));
            let block = pick_block(size);
            let rt = runtime();
            let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
            {
                let at = store.tensor(a);
                let bt = store.tensor(b);
                let av = at.view(&[size]).expect("view a");
                let bv = bt.view(&[size]).expect("view b");
                let _ = kernels::$kernel((&mut out).partition([block]), &av, &bv)
                    .sync_on(&rt.stream)
                    .expect(concat!($label, " kernel"));
            }
            let logical = out.reshape(&shape).expect("reshape");
            store.insert_tensor(logical, shape)
        }
    };
}

binary_op!(add, add, "add");
binary_op!(sub, sub, "sub");
binary_op!(mul, mul, "mul");
binary_op!(div, div, "div");

pub fn neg(store: &mut TensorStore, a: TensorId) -> TensorId {
    let shape = store.shape(a).to_vec();
    let size = shape_size(&shape);
    let block = pick_block(size);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let at = store.tensor(a);
        let av = at.view(&[size]).expect("view a");
        let _ = kernels::neg((&mut out).partition([block]), &av)
            .sync_on(&rt.stream)
            .expect("neg kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

pub fn mul_scalar(store: &mut TensorStore, a: TensorId, s: f32) -> TensorId {
    let shape = store.shape(a).to_vec();
    let size = shape_size(&shape);
    let block = pick_block(size);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let at = store.tensor(a);
        let av = at.view(&[size]).expect("view a");
        let _ = kernels::mul_scalar((&mut out).partition([block]), &av, s)
            .sync_on(&rt.stream)
            .expect("mul_scalar kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

/// Fused z = a*x + y.
pub fn saxpy(store: &mut TensorStore, a_scalar: f32, x: TensorId, y: TensorId) -> TensorId {
    let shape = store.shape(x).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(y), size, "saxpy: shape mismatch");
    let block = pick_block(size);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let xt = store.tensor(x);
        let yt = store.tensor(y);
        let xv = xt.view(&[size]).expect("view x");
        let yv = yt.view(&[size]).expect("view y");
        let _ = kernels::saxpy((&mut out).partition([block]), a_scalar, &xv, &yv)
            .sync_on(&rt.stream)
            .expect("saxpy kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

// ---------------------------------------------------------------------------
// Unary: exp / log / copy
// ---------------------------------------------------------------------------

macro_rules! unary_op {
    ($name:ident, $kernel:ident, $label:literal) => {
        pub fn $name(store: &mut TensorStore, a: TensorId) -> TensorId {
            let shape = store.shape(a).to_vec();
            let size = shape_size(&shape);
            let block = pick_block(size);
            let rt = runtime();
            let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
            {
                let at = store.tensor(a);
                let av = at.view(&[size]).expect("view a");
                let _ = kernels::$kernel((&mut out).partition([block]), &av)
                    .sync_on(&rt.stream)
                    .expect(concat!($label, " kernel"));
            }
            let logical = out.reshape(&shape).expect("reshape");
            store.insert_tensor(logical, shape)
        }
    };
}

unary_op!(exp, exp_f32, "exp");
unary_op!(log, log_f32, "log");
unary_op!(copy, copy, "copy");

// ---------------------------------------------------------------------------
// Fill — writes a constant into a fresh tensor of a given shape.
// ---------------------------------------------------------------------------

pub fn fill(store: &mut TensorStore, shape: &[usize], val: f32) -> TensorId {
    let size = shape_size(shape);
    let block = pick_block(size);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let _ = kernels::fill((&mut out).partition([block]), val)
            .sync_on(&rt.stream)
            .expect("fill kernel");
    }
    let logical = out.reshape(shape).expect("reshape");
    store.insert_tensor(logical, shape.to_vec())
}

// ---------------------------------------------------------------------------
// Pow + its backward (both take a scalar `exponent`).
// ---------------------------------------------------------------------------

pub fn pow(store: &mut TensorStore, a: TensorId, exponent: f32) -> TensorId {
    let shape = store.shape(a).to_vec();
    let size = shape_size(&shape);
    let block = pick_block(size);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let at = store.tensor(a);
        let av = at.view(&[size]).expect("view a");
        let _ = kernels::pow_f32((&mut out).partition([block]), &av, exponent)
            .sync_on(&rt.stream)
            .expect("pow kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

pub fn pow_backward(
    store: &mut TensorStore,
    dy: TensorId,
    x: TensorId,
    exponent: f32,
) -> TensorId {
    let shape = store.shape(dy).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(x), size, "pow_backward: shape mismatch");
    let block = pick_block(size);
    let rt = runtime();
    let mut dx = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let dyt = store.tensor(dy);
        let xt = store.tensor(x);
        let dyv = dyt.view(&[size]).expect("view dy");
        let xv = xt.view(&[size]).expect("view x");
        let _ = kernels::pow_backward_f32((&mut dx).partition([block]), &dyv, &xv, exponent)
            .sync_on(&rt.stream)
            .expect("pow_backward kernel");
    }
    let logical = dx.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

// ---------------------------------------------------------------------------
// Div backward: da = dy / b, db = -dy * a / b².
// ---------------------------------------------------------------------------

pub fn div_backward_a(store: &mut TensorStore, dy: TensorId, b: TensorId) -> TensorId {
    let shape = store.shape(dy).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(b), size, "div_backward_a: shape mismatch");
    let block = pick_block(size);
    let rt = runtime();
    let mut da = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let dyt = store.tensor(dy);
        let bt = store.tensor(b);
        let dyv = dyt.view(&[size]).expect("view dy");
        let bv = bt.view(&[size]).expect("view b");
        let _ = kernels::div_backward_a((&mut da).partition([block]), &dyv, &bv)
            .sync_on(&rt.stream)
            .expect("div_backward_a kernel");
    }
    let logical = da.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

pub fn div_backward_b(
    store: &mut TensorStore,
    dy: TensorId,
    a: TensorId,
    b: TensorId,
) -> TensorId {
    let shape = store.shape(dy).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(a), size, "div_backward_b: shape mismatch");
    assert_eq!(store.size(b), size, "div_backward_b: shape mismatch");
    let block = pick_block(size);
    let rt = runtime();
    let mut db = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let dyt = store.tensor(dy);
        let at = store.tensor(a);
        let bt = store.tensor(b);
        let dyv = dyt.view(&[size]).expect("view dy");
        let av = at.view(&[size]).expect("view a");
        let bv = bt.view(&[size]).expect("view b");
        let _ = kernels::div_backward_b((&mut db).partition([block]), &dyv, &av, &bv)
            .sync_on(&rt.stream)
            .expect("div_backward_b kernel");
    }
    let logical = db.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

// ---------------------------------------------------------------------------
// Elementwise comparison: lt / gt / eq / is_close — all produce a 0/1 f32 mask.
// ---------------------------------------------------------------------------

binary_op!(lt, lt, "lt");
binary_op!(gt, gt, "gt");
binary_op!(eq, eq, "eq");

pub fn is_close(store: &mut TensorStore, a: TensorId, b: TensorId, tol: f32) -> TensorId {
    let shape = store.shape(a).to_vec();
    let size = shape_size(&shape);
    assert_eq!(store.size(b), size, "is_close: shape mismatch");
    let block = pick_block(size);
    let rt = runtime();
    let mut out = api::zeros::<f32>(&[size]).sync_on(&rt.stream).expect("alloc");
    {
        let at = store.tensor(a);
        let bt = store.tensor(b);
        let av = at.view(&[size]).expect("view a");
        let bv = bt.view(&[size]).expect("view b");
        let _ = kernels::is_close((&mut out).partition([block]), &av, &bv, tol)
            .sync_on(&rt.stream)
            .expect("is_close kernel");
    }
    let logical = out.reshape(&shape).expect("reshape");
    store.insert_tensor(logical, shape)
}

// ---------------------------------------------------------------------------
// 2D broadcast ops: add_bias / broadcast_add / broadcast_mul — x : [N, C]
// with a [C] bias/other vector broadcasting along the N axis.
// ---------------------------------------------------------------------------

const ROW_CANDIDATES: [usize; 6] = [32, 16, 8, 4, 2, 1];

fn pick_row_block(n: usize) -> usize {
    for &b in &ROW_CANDIDATES {
        if n % b == 0 {
            return b;
        }
    }
    1
}

fn flat_2d_shape(shape: &[usize]) -> (usize, usize) {
    assert!(shape.len() >= 2, "broadcast op needs rank >= 2 input");
    let c = *shape.last().unwrap();
    let n: usize = shape[..shape.len() - 1].iter().product();
    (n, c)
}

macro_rules! broadcast_2d_op {
    ($name:ident, $kernel:ident, $label:literal) => {
        pub fn $name(store: &mut TensorStore, x: TensorId, y: TensorId) -> TensorId {
            let shape = store.shape(x).to_vec();
            let (n, c) = flat_2d_shape(&shape);
            assert_eq!(
                store.size(y),
                c,
                concat!($label, ": broadcast arg must match last dim")
            );
            let bn = pick_row_block(n);
            let rt = runtime();
            let mut out = api::zeros::<f32>(&[n, c])
                .sync_on(&rt.stream)
                .expect("alloc");
            {
                let xt = store.tensor(x);
                let yt = store.tensor(y);
                let xv = xt.view(&[n, c]).expect("view x");
                let yv = yt.view(&[c]).expect("view y");
                let _ = kernels::$kernel((&mut out).partition([bn, c]), &xv, &yv)
                    .generics(vec![bn.to_string(), c.to_string()])
                    .sync_on(&rt.stream)
                    .expect(concat!($label, " kernel"));
            }
            let logical = out.reshape(&shape).expect("reshape");
            store.insert_tensor(logical, shape)
        }
    };
}

broadcast_2d_op!(add_bias, add_bias, "add_bias");
broadcast_2d_op!(broadcast_add, broadcast_add, "broadcast_add");
broadcast_2d_op!(broadcast_mul, broadcast_mul, "broadcast_mul");

// ---------------------------------------------------------------------------
// Permute — runtime stride arithmetic, up to 4D.
// ---------------------------------------------------------------------------

const PERMUTE_BLOCK: usize = 256;

/// Permute `a`'s axes by `perm` (e.g. [0, 2, 1]).  Returns a new tensor whose
/// shape is `a.shape` reordered by `perm`.  The kernel is launched with a 1D
/// grid over the flat output indices and reconstructs the source index via
/// divisor/stride arithmetic so the same compiled PTX handles any ndim ≤ 4.
pub fn permute(store: &mut TensorStore, a: TensorId, perm: &[usize]) -> TensorId {
    let src_shape = store.shape(a).to_vec();
    let ndim = src_shape.len();
    assert!(ndim <= 4, "permute: ndim must be ≤ 4 (got {ndim})");
    assert_eq!(perm.len(), ndim, "permute: perm length must equal ndim");
    for &p in perm {
        assert!(p < ndim, "permute: invalid axis {p}");
    }

    // Dest shape: src_shape permuted by `perm`.
    let dst_shape: Vec<usize> = perm.iter().map(|&p| src_shape[p]).collect();
    let n = shape_size(&dst_shape);
    if n == 0 {
        return store.zeros(&dst_shape);
    }

    // Src strides (row-major).
    let mut src_strides = vec![0i32; ndim];
    {
        let mut s = 1usize;
        for i in (0..ndim).rev() {
            src_strides[i] = s as i32;
            s *= src_shape[i];
        }
    }

    // `ds[a]` = product of dst dims after axis a (row-major divisor so
    // `idx / ds[a] mod dst_shape[a]` recovers the a-th dst coord).
    // `es[a]` = src stride for the axis that ends up at dst position a.
    let mut ds = [1i32; 4];
    let mut es = [0i32; 4];
    {
        let mut block = 1usize;
        for i in (0..ndim).rev() {
            ds[i] = block as i32;
            block *= dst_shape[i];
        }
        for i in 0..ndim {
            es[i] = src_strides[perm[i]];
        }
    }

    let grid = n.div_ceil(PERMUTE_BLOCK) as u32;
    let rt = runtime();
    let out = api::zeros::<f32>(&[n]).sync_on(&rt.stream).expect("alloc");
    {
        let src_t = store.tensor(a);
        let src_ptr = src_t.device_pointer();
        let out_ptr = out.device_pointer();
        unsafe {
            let _ = kernels::permute_runtime(
                out_ptr,
                src_ptr,
                n as i32,
                ds[0],
                ds[1],
                ds[2],
                ds[3],
                es[0],
                es[1],
                es[2],
                es[3],
                ndim as i32,
            )
            .grid((grid, 1, 1))
            .generics(vec![PERMUTE_BLOCK.to_string()])
            .sync_on(&rt.stream)
            .expect("permute kernel");
        }
    }
    let logical = out.reshape(&dst_shape).expect("reshape");
    store.insert_tensor(logical, dst_shape)
}
