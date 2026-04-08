use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore, compute_strides};

#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchConfig, PushKernelArg};

#[cfg(feature = "cuda")]
fn launch_cfg(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch the `permute_f32` kernel on the GPU.
/// `dims` maps output dimension d to source dimension dims[d].
/// Both `src_id` and `out_id` must be contiguous; `src_shape` is the shape of the source.
#[cfg(feature = "cuda")]
fn launch_permute(
    src_id: TensorId,
    out_id: TensorId,
    dims: &[usize],
    src_shape: &[usize],
    store: &TensorStore,
) {
    let ndim = dims.len();
    let size = store.size(out_id);
    let src_strides = compute_strides(src_shape);
    let out_shape = store.shape(out_id);
    let dst_strides = compute_strides(out_shape);

    let mut ds = [1i32; 4];
    let mut es = [0i32; 4];
    for d in 0..ndim.min(4) {
        ds[d] = dst_strides[d] as i32;
        es[d] = src_strides[dims[d]] as i32;
    }

    let dev = GpuDevice::instance();
    let func = dev.get_func("permute_f32");
    let src_ptr = store.dev_ptr(src_id);
    let out_ptr = store.dev_ptr(out_id);
    let n = size as i32;
    let nd = ndim as i32;
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr)
            .arg(&src_ptr)
            .arg(&n)
            .arg(&ds[0]).arg(&ds[1]).arg(&ds[2]).arg(&ds[3])
            .arg(&es[0]).arg(&es[1]).arg(&es[2]).arg(&es[3])
            .arg(&nd)
            .launch(launch_cfg(size as u32))
            .unwrap();
    }
}

// =========================================================================
// View
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn view(a: TensorId, new_shape: &[usize], store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let orig_shape = store.shape(a).to_vec();
    let a_id = store.ensure_contiguous(a);
    let data = store.to_host(a_id);
    let out = store.from_vec(data, new_shape);
    tape.record(TapeEntry {
        op: BackwardOp::View, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Shape(orig_shape),
    });
    out
}

#[cfg(feature = "cuda")]
pub fn view(a: TensorId, new_shape: &[usize], store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let orig_shape = store.shape(a).to_vec();
    let a_id = store.ensure_contiguous(a);
    let out = store.clone_device(a_id, new_shape);
    tape.record(TapeEntry {
        op: BackwardOp::View, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Shape(orig_shape),
    });
    out
}

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn view_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Shape(orig_shape) = saved {
        let data = store.to_host(grad);
        vec![Some(store.from_vec(data, orig_shape))]
    } else { vec![None] }
}

#[cfg(feature = "cuda")]
pub fn view_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Shape(orig_shape) = saved {
        let g = store.ensure_contiguous(grad);
        vec![Some(store.clone_device(g, orig_shape))]
    } else { vec![None] }
}

// =========================================================================
// Permute
// =========================================================================

/// CPU permute: share data buffer, just rearrange strides (creates non-contiguous view).
#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn permute(a: TensorId, dims: &[usize], store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let a_shape = store.shape(a).to_vec();
    let a_strides = store.get(a).strides.clone();
    let ndim = a_shape.len();

    let mut new_shape = vec![0usize; ndim];
    let mut new_strides = vec![0usize; ndim];
    for i in 0..ndim {
        new_shape[i] = a_shape[dims[i]];
        new_strides[i] = a_strides[dims[i]];
    }

    let data = store.get(a).data.clone();
    let size = store.size(a);
    let out = store.insert_raw(data, new_shape, new_strides, size);

    tape.record(TapeEntry {
        op: BackwardOp::Permute, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Permutation(dims.to_vec(), a_shape),
    });
    out
}

/// CUDA permute: physically rearrange data via `permute_f32` kernel (result is contiguous).
#[cfg(feature = "cuda")]
pub fn permute(a: TensorId, dims: &[usize], store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let a_shape = store.shape(a).to_vec();
    let ndim = a_shape.len();
    let a_id = store.ensure_contiguous(a);

    let mut new_shape = vec![0usize; ndim];
    for i in 0..ndim {
        new_shape[i] = a_shape[dims[i]];
    }

    let out = store.zeros(&new_shape);
    launch_permute(a_id, out, dims, &a_shape, store);

    tape.record(TapeEntry {
        op: BackwardOp::Permute, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Permutation(dims.to_vec(), a_shape),
    });
    out
}

// =========================================================================
// Permute backward
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn permute_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Permutation(order, orig_shape) = saved {
        let ndim = order.len();
        let mut inv = vec![0usize; ndim];
        for i in 0..ndim {
            inv[order[i]] = i;
        }
        let grad_contig = store.ensure_contiguous(grad);
        let data = store.to_host(grad_contig);
        let grad_shape = store.shape(grad_contig).to_vec();

        let src_strides = compute_strides(&grad_shape);
        let size = crate::tensor::shape_size(&grad_shape);

        let mut out = vec![0.0f32; size];
        let out_strides = compute_strides(orig_shape);

        for i in 0..size {
            let mut coord = vec![0usize; ndim];
            let mut rem = i;
            for d in 0..ndim {
                coord[d] = rem / src_strides[d];
                rem %= src_strides[d];
            }
            let mut out_idx = 0;
            for d in 0..ndim {
                out_idx += coord[inv[d]] * out_strides[d];
            }
            out[out_idx] = data[i];
        }
        vec![Some(store.from_vec(out, orig_shape))]
    } else { vec![None] }
}

/// CUDA backward: permute grad with the inverse mapping via `permute_f32` kernel.
#[cfg(feature = "cuda")]
pub fn permute_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Permutation(order, orig_shape) = saved {
        let ndim = order.len();
        let mut inv = vec![0usize; ndim];
        for i in 0..ndim {
            inv[order[i]] = i;
        }
        let grad_id = store.ensure_contiguous(grad);
        let grad_shape = store.shape(grad_id).to_vec();
        let out = store.zeros(orig_shape);
        launch_permute(grad_id, out, &inv, &grad_shape, store);
        vec![Some(out)]
    } else { vec![None] }
}

// =========================================================================
// Contiguous
// =========================================================================

pub fn contiguous(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let out = store.ensure_contiguous(a);
    if out != a {
        tape.record(TapeEntry {
            op: BackwardOp::Contiguous, output_id: out, input_ids: smallvec![a],
            saved: SavedContext::None,
        });
    }
    out
}

pub fn contiguous_backward(grad: TensorId, _saved: &SavedContext, _store: &mut TensorStore) -> Vec<Option<TensorId>> {
    vec![Some(grad)]
}

// =========================================================================
// insert_raw helper — CPU only (used by CPU permute for non-contiguous views)
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
impl TensorStore {
    pub fn insert_raw(&mut self, data: Vec<f32>, shape: Vec<usize>, strides: Vec<usize>, size: usize) -> TensorId {
        use crate::tensor::GpuTensor;
        let t = GpuTensor {
            data, shape, strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
        };
        if let Some(id) = self.free_ids.pop() {
            self.tensors[id] = Some(t);
            id
        } else {
            let id = self.tensors.len();
            self.tensors.push(Some(t));
            id
        }
    }
}
