use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore, shape_size};

#[cfg(feature = "cuda")]
use crate::tensor::compute_strides;
#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use cudarc::cublas::safe::{GemmConfig, StridedBatchedConfig, Gemm};
#[cfg(feature = "cuda")]
use cudarc::cublas::sys::cublasOperation_t;
#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchConfig, PushKernelArg};

/// CPU matmul supporting batched dimensions.
/// A: [..., M, K], B: [..., K, N] → C: [..., M, N]
#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn matmul(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let a_id = store.ensure_contiguous(a);
    let b_id = store.ensure_contiguous(b);
    let a_shape = store.shape(a_id).to_vec();
    let b_shape = store.shape(b_id).to_vec();

    assert!(a_shape.len() >= 2 && b_shape.len() >= 2, "matmul requires at least 2D tensors");
    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let k2 = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];
    assert_eq!(k, k2, "matmul inner dimensions must match: {} vs {}", k, k2);

    let a_batch: Vec<usize> = a_shape[..a_shape.len()-2].to_vec();
    let b_batch: Vec<usize> = b_shape[..b_shape.len()-2].to_vec();

    let out_batch = crate::utils::broadcast_shape(&a_batch, &b_batch);
    let batch_size = shape_size(&out_batch);

    let mut out_shape = out_batch.clone();
    out_shape.push(m);
    out_shape.push(n);

    let a_batch_size = shape_size(&a_batch);
    let b_batch_size = shape_size(&b_batch);

    let a_data = store.to_host(a_id);
    let b_data = store.to_host(b_id);
    let out_size = shape_size(&out_shape);
    let mut out = vec![0.0f32; out_size];

    let a_mat = m * k;
    let b_mat = k * n;
    let c_mat = m * n;

    for batch in 0..batch_size {
        let a_batch_idx = if a_batch_size == 1 { 0 } else { batch % a_batch_size };
        let b_batch_idx = if b_batch_size == 1 { 0 } else { batch % b_batch_size };

        let a_off = a_batch_idx * a_mat;
        let b_off = b_batch_idx * b_mat;
        let c_off = batch * c_mat;

        for i in 0..m {
            for kk in 0..k {
                let a_val = a_data[a_off + i * k + kk];
                for j in 0..n {
                    out[c_off + i * n + j] += a_val * b_data[b_off + kk * n + j];
                }
            }
        }
    }

    let out_id = store.from_vec(out, &out_shape);
    tape.record(TapeEntry {
        op: BackwardOp::MatMul, output_id: out_id, input_ids: smallvec![a, b],
        saved: SavedContext::Tensors(smallvec![a, b]),
    });
    out_id
}

/// CUDA matmul using cuBLAS gemm / gemm_strided_batched.
/// A: [..., M, K], B: [..., K, N] → C: [..., M, N]
#[cfg(feature = "cuda")]
pub fn matmul(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let a_id = store.ensure_contiguous(a);
    let b_id = store.ensure_contiguous(b);
    let a_shape = store.shape(a_id).to_vec();
    let b_shape = store.shape(b_id).to_vec();

    assert!(a_shape.len() >= 2 && b_shape.len() >= 2, "matmul requires at least 2D tensors");
    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let k2 = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];
    assert_eq!(k, k2, "matmul inner dimensions must match: {} vs {}", k, k2);

    let a_batch: Vec<usize> = a_shape[..a_shape.len()-2].to_vec();
    let b_batch: Vec<usize> = b_shape[..b_shape.len()-2].to_vec();
    let out_batch = crate::utils::broadcast_shape(&a_batch, &b_batch);
    let batch_size = shape_size(&out_batch);

    let mut out_shape = out_batch.clone();
    out_shape.push(m);
    out_shape.push(n);

    let a_batch_size = shape_size(&a_batch);
    let b_batch_size = shape_size(&b_batch);

    // cuBLAS strided-batched supports stride=0 for broadcasting when one
    // operand has batch_size=1. For rarer multi-dim broadcast patterns,
    // fall back to host-side computation.
    let simple_broadcast = (a_batch_size == batch_size || a_batch_size == 1)
        && (b_batch_size == batch_size || b_batch_size == 1);

    if !simple_broadcast {
        let a_data = store.to_host(a_id);
        let b_data = store.to_host(b_id);
        let out_size = shape_size(&out_shape);
        let mut out = vec![0.0f32; out_size];
        let a_mat = m * k;
        let b_mat = k * n;
        let c_mat = m * n;
        for batch in 0..batch_size {
            let a_bi = if a_batch_size == 1 { 0 } else { batch % a_batch_size };
            let b_bi = if b_batch_size == 1 { 0 } else { batch % b_batch_size };
            let a_off = a_bi * a_mat;
            let b_off = b_bi * b_mat;
            let c_off = batch * c_mat;
            for i in 0..m {
                for kk in 0..k {
                    let a_val = a_data[a_off + i * k + kk];
                    for j in 0..n {
                        out[c_off + i * n + j] += a_val * b_data[b_off + kk * n + j];
                    }
                }
            }
        }
        let out_id = store.from_vec(out, &out_shape);
        tape.record(TapeEntry {
            op: BackwardOp::MatMul, output_id: out_id, input_ids: smallvec![a, b],
            saved: SavedContext::Tensors(smallvec![a, b]),
        });
        return out_id;
    }

    let out_id = store.zeros(&out_shape);
    let dev = GpuDevice::instance();

    // cuBLAS is column-major. For row-major C = A @ B where A:[M,K], B:[K,N]:
    // pass B as first arg, A as second, with m=N, n=M, k=K.
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0f32,
        lda: n as i32,
        ldb: k as i32,
        beta: 0.0f32,
        ldc: n as i32,
    };

    // Use raw pointer arithmetic to obtain non-overlapping references to
    // distinct tensor slots, satisfying the borrow checker.
    let tensors_ptr = store.tensors.as_mut_ptr();

    if batch_size <= 1 {
        unsafe {
            let b_data = &(*tensors_ptr.add(b_id)).as_ref().unwrap().data;
            let a_data = &(*tensors_ptr.add(a_id)).as_ref().unwrap().data;
            let c_data = &mut (*tensors_ptr.add(out_id)).as_mut().unwrap().data;
            dev.blas.gemm(cfg, b_data, a_data, c_data).unwrap();
        }
    } else {
        let stride_a = if b_batch_size == 1 { 0 } else { (k * n) as i64 };
        let stride_b = if a_batch_size == 1 { 0 } else { (m * k) as i64 };
        let batched_cfg = StridedBatchedConfig {
            gemm: cfg,
            batch_size: batch_size as i32,
            stride_a,
            stride_b,
            stride_c: (m * n) as i64,
        };
        unsafe {
            let b_data = &(*tensors_ptr.add(b_id)).as_ref().unwrap().data;
            let a_data = &(*tensors_ptr.add(a_id)).as_ref().unwrap().data;
            let c_data = &mut (*tensors_ptr.add(out_id)).as_mut().unwrap().data;
            dev.blas.gemm_strided_batched(batched_cfg, b_data, a_data, c_data).unwrap();
        }
    }

    tape.record(TapeEntry {
        op: BackwardOp::MatMul, output_id: out_id, input_ids: smallvec![a, b],
        saved: SavedContext::Tensors(smallvec![a, b]),
    });
    out_id
}

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn matmul_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensors(ids) = saved {
        let a = ids[0]; let b = ids[1];
        let a_shape = store.shape(a).to_vec();
        let b_shape = store.shape(b).to_vec();

        let b_t = transpose_last2(b, store);
        let grad_a = matmul_no_grad(grad, b_t, store);
        let grad_a_data = store.to_host(grad_a);
        let grad_a_shape = store.shape(grad_a).to_vec();
        let ga = crate::utils::unbroadcast(&grad_a_data, &grad_a_shape, &a_shape);
        let ga_id = store.from_vec(ga, &a_shape);

        let a_t = transpose_last2(a, store);
        let grad_b = matmul_no_grad(a_t, grad, store);
        let grad_b_data = store.to_host(grad_b);
        let grad_b_shape = store.shape(grad_b).to_vec();
        let gb = crate::utils::unbroadcast(&grad_b_data, &grad_b_shape, &b_shape);
        let gb_id = store.from_vec(gb, &b_shape);

        vec![Some(ga_id), Some(gb_id)]
    } else { vec![None, None] }
}

#[cfg(feature = "cuda")]
pub fn matmul_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensors(ids) = saved {
        let a = ids[0]; let b = ids[1];
        let a_shape = store.shape(a).to_vec();
        let b_shape = store.shape(b).to_vec();

        let b_t = transpose_last2(b, store);
        let grad_a = matmul_no_grad(grad, b_t, store);
        let grad_a_shape = store.shape(grad_a).to_vec();
        let ga_id = if grad_a_shape == a_shape {
            grad_a
        } else {
            let grad_a_data = store.to_host(grad_a);
            let ga = crate::utils::unbroadcast(&grad_a_data, &grad_a_shape, &a_shape);
            store.from_vec(ga, &a_shape)
        };

        let a_t = transpose_last2(a, store);
        let grad_b = matmul_no_grad(a_t, grad, store);
        let grad_b_shape = store.shape(grad_b).to_vec();
        let gb_id = if grad_b_shape == b_shape {
            grad_b
        } else {
            let grad_b_data = store.to_host(grad_b);
            let gb = crate::utils::unbroadcast(&grad_b_data, &grad_b_shape, &b_shape);
            store.from_vec(gb, &b_shape)
        };

        vec![Some(ga_id), Some(gb_id)]
    } else { vec![None, None] }
}

/// Transpose the last two dimensions on CPU.
#[cfg(any(feature = "cpu", feature = "webgpu"))]
fn transpose_last2(a: TensorId, store: &mut TensorStore) -> TensorId {
    let shape = store.shape(a).to_vec();
    let ndim = shape.len();
    let a_id = store.ensure_contiguous(a);
    let data = store.to_host(a_id);

    let m = shape[ndim - 2];
    let n = shape[ndim - 1];
    let batch_size: usize = shape[..ndim-2].iter().product::<usize>().max(1);

    let mut out = vec![0.0f32; data.len()];
    let mat_size = m * n;
    for b in 0..batch_size {
        let off = b * mat_size;
        for i in 0..m {
            for j in 0..n {
                out[off + j * m + i] = data[off + i * n + j];
            }
        }
    }

    let mut out_shape = shape[..ndim-2].to_vec();
    out_shape.push(n);
    out_shape.push(m);
    store.from_vec(out, &out_shape)
}

/// Transpose the last two dimensions on GPU via the `permute_f32` kernel.
#[cfg(feature = "cuda")]
fn transpose_last2(a: TensorId, store: &mut TensorStore) -> TensorId {
    let shape = store.shape(a).to_vec();
    let ndim = shape.len();
    let a_id = store.ensure_contiguous(a);

    let mut dims: Vec<usize> = (0..ndim).collect();
    dims.swap(ndim - 2, ndim - 1);

    let mut new_shape = shape.clone();
    new_shape.swap(ndim - 2, ndim - 1);
    let src_strides = compute_strides(&shape);
    let dst_strides = compute_strides(&new_shape);
    let size = shape_size(&shape);

    let out = store.zeros(&new_shape);
    let dev = GpuDevice::instance();
    let src_ptr = store.dev_ptr(a_id);
    let out_ptr = store.dev_ptr(out);

    let mut ds = [1i32; 4];
    let mut es = [0i32; 4];
    for d in 0..ndim.min(4) {
        ds[d] = dst_strides[d] as i32;
        es[d] = src_strides[dims[d]] as i32;
    }

    let func = dev.get_func("permute_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr).arg(&src_ptr)
            .arg(&(size as i32))
            .arg(&ds[0]).arg(&ds[1]).arg(&ds[2]).arg(&ds[3])
            .arg(&es[0]).arg(&es[1]).arg(&es[2]).arg(&es[3])
            .arg(&(ndim as i32))
            .launch(LaunchConfig { grid_dim: (((size as u32) + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
            .unwrap();
    }
    out
}

/// Matmul without recording to tape (used in backward).
fn matmul_no_grad(a: TensorId, b: TensorId, store: &mut TensorStore) -> TensorId {
    let mut dummy_tape = crate::autograd::Tape::new();
    dummy_tape.set_enabled(false);
    matmul(a, b, store, &mut dummy_tape)
}
