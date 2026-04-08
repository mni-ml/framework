use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::ops::data::IntStore;
use crate::tensor::{TensorId, TensorStore};

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

// =========================================================================
// Forward
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn embedding_forward(
    weight: TensorId, indices: &[usize], batch: usize, seq_len: usize,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let w_shape = store.shape(weight).to_vec();
    let embed_dim = w_shape[1];
    let w_data = store.to_host(weight);

    let mut out = vec![0.0f32; batch * seq_len * embed_dim];
    for b in 0..batch {
        for t in 0..seq_len {
            let idx = indices[b * seq_len + t];
            let src_off = idx * embed_dim;
            let dst_off = (b * seq_len + t) * embed_dim;
            out[dst_off..dst_off + embed_dim].copy_from_slice(&w_data[src_off..src_off + embed_dim]);
        }
    }

    let out_shape = vec![batch, seq_len, embed_dim];
    let out_id = store.from_vec(out, &out_shape);

    tape.record(TapeEntry {
        op: BackwardOp::Embedding, output_id: out_id,
        input_ids: smallvec![weight],
        saved: SavedContext::Indices(indices.to_vec(), batch, seq_len, weight),
    });
    out_id
}

#[cfg(feature = "cuda")]
pub fn embedding_forward(
    weight: TensorId, indices: &[usize], batch: usize, seq_len: usize,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let w_shape = store.shape(weight).to_vec();
    let embed_dim = w_shape[1];
    let total_tokens = batch * seq_len;

    let dev = GpuDevice::instance();
    let weight_ptr = store.dev_ptr(weight);

    let indices_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let indices_gpu = dev.stream.memcpy_stod(&indices_i32).unwrap();
    let indices_ptr = dev.ptr(&indices_gpu);

    let out_shape = vec![batch, seq_len, embed_dim];
    let out_id = store.zeros(&out_shape);
    let out_ptr = store.dev_ptr(out_id);

    let total_threads = (total_tokens * embed_dim) as u32;
    let func = dev.get_func("embedding_forward_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr)
            .arg(&weight_ptr)
            .arg(&indices_ptr)
            .arg(&(total_tokens as i32))
            .arg(&(embed_dim as i32))
            .launch(launch_cfg(total_threads))
            .unwrap();
    }

    tape.record(TapeEntry {
        op: BackwardOp::Embedding, output_id: out_id,
        input_ids: smallvec![weight],
        saved: SavedContext::Indices(indices.to_vec(), batch, seq_len, weight),
    });
    out_id
}

// =========================================================================
// Backward
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn embedding_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Indices(indices, batch, seq_len, weight_id) = saved {
        let w_shape = store.shape(*weight_id).to_vec();
        let vocab_size = w_shape[0];
        let embed_dim = w_shape[1];
        let grad_data = store.to_host(grad);

        let mut dw = vec![0.0f32; vocab_size * embed_dim];
        for b in 0..*batch {
            for t in 0..*seq_len {
                let idx = indices[b * seq_len + t];
                let src_off = (b * seq_len + t) * embed_dim;
                let dst_off = idx * embed_dim;
                for j in 0..embed_dim {
                    dw[dst_off + j] += grad_data[src_off + j];
                }
            }
        }

        vec![Some(store.from_vec(dw, &w_shape))]
    } else { vec![None] }
}

#[cfg(feature = "cuda")]
pub fn embedding_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Indices(indices, batch, seq_len, weight_id) = saved {
        let w_shape = store.shape(*weight_id).to_vec();
        let embed_dim = w_shape[1];
        let total_tokens = batch * seq_len;

        let dev = GpuDevice::instance();
        let grad_ptr = store.dev_ptr(grad);

        let indices_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
        let indices_gpu = dev.stream.memcpy_stod(&indices_i32).unwrap();
        let indices_ptr = dev.ptr(&indices_gpu);

        let dw_id = store.zeros(&w_shape);
        let dw_ptr = store.dev_ptr(dw_id);

        let total_threads = (total_tokens * embed_dim) as u32;
        let func = dev.get_func("embedding_backward_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&dw_ptr)
                .arg(&grad_ptr)
                .arg(&indices_ptr)
                .arg(&(total_tokens as i32))
                .arg(&(embed_dim as i32))
                .launch(launch_cfg(total_threads))
                .unwrap();
        }

        vec![Some(dw_id)]
    } else { vec![None] }
}

// =========================================================================
// GPU-index forward (indices already on GPU via IntStore, zero PCIe)
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn embedding_forward_gpu(
    weight: TensorId, int_buf_id: usize, batch: usize, seq_len: usize,
    int_store: &IntStore, store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let w_shape = store.shape(weight).to_vec();
    let embed_dim = w_shape[1];
    let w_data = store.to_host(weight);
    let indices = &int_store.get(int_buf_id).data;

    let mut out = vec![0.0f32; batch * seq_len * embed_dim];
    for i in 0..batch * seq_len {
        let idx = indices[i] as usize;
        let src = idx * embed_dim;
        let dst = i * embed_dim;
        out[dst..dst + embed_dim].copy_from_slice(&w_data[src..src + embed_dim]);
    }

    let out_id = store.from_vec(out, &[batch, seq_len, embed_dim]);
    tape.record(TapeEntry {
        op: BackwardOp::EmbeddingGpu, output_id: out_id,
        input_ids: smallvec![weight],
        saved: SavedContext::GpuIndices(int_buf_id, batch, seq_len, weight),
    });
    out_id
}

#[cfg(feature = "cuda")]
pub fn embedding_forward_gpu(
    weight: TensorId, int_buf_id: usize, batch: usize, seq_len: usize,
    int_store: &IntStore, store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let w_shape = store.shape(weight).to_vec();
    let embed_dim = w_shape[1];
    let total_tokens = batch * seq_len;

    let dev = GpuDevice::instance();
    let weight_ptr = store.dev_ptr(weight);
    let indices_ptr = dev.ptr(&int_store.get(int_buf_id).data);

    let out_id = store.zeros(&[batch, seq_len, embed_dim]);
    let out_ptr = store.dev_ptr(out_id);

    let total_threads = (total_tokens * embed_dim) as u32;
    let func = dev.get_func("embedding_forward_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr)
            .arg(&weight_ptr)
            .arg(&indices_ptr)
            .arg(&(total_tokens as i32))
            .arg(&(embed_dim as i32))
            .launch(launch_cfg(total_threads))
            .unwrap();
    }

    tape.record(TapeEntry {
        op: BackwardOp::EmbeddingGpu, output_id: out_id,
        input_ids: smallvec![weight],
        saved: SavedContext::GpuIndices(int_buf_id, batch, seq_len, weight),
    });
    out_id
}

// =========================================================================
// GPU-index backward
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn embedding_backward_gpu(
    grad: TensorId, saved: &SavedContext,
    int_store: &IntStore, store: &mut TensorStore,
) -> Vec<Option<TensorId>> {
    if let SavedContext::GpuIndices(int_buf_id, batch, seq_len, weight_id) = saved {
        let w_shape = store.shape(*weight_id).to_vec();
        let vocab_size = w_shape[0];
        let embed_dim = w_shape[1];
        let grad_data = store.to_host(grad);
        let indices = &int_store.get(*int_buf_id).data;

        let mut dw = vec![0.0f32; vocab_size * embed_dim];
        for i in 0..batch * seq_len {
            let idx = indices[i] as usize;
            let src_off = i * embed_dim;
            let dst_off = idx * embed_dim;
            for j in 0..embed_dim {
                dw[dst_off + j] += grad_data[src_off + j];
            }
        }

        vec![Some(store.from_vec(dw, &w_shape))]
    } else { vec![None] }
}

#[cfg(feature = "cuda")]
pub fn embedding_backward_gpu(
    grad: TensorId, saved: &SavedContext,
    int_store: &IntStore, store: &mut TensorStore,
) -> Vec<Option<TensorId>> {
    if let SavedContext::GpuIndices(int_buf_id, batch, seq_len, weight_id) = saved {
        let w_shape = store.shape(*weight_id).to_vec();
        let embed_dim = w_shape[1];
        let total_tokens = batch * seq_len;

        let dev = GpuDevice::instance();
        let grad_ptr = store.dev_ptr(grad);
        let indices_ptr = dev.ptr(&int_store.get(*int_buf_id).data);

        let dw_id = store.zeros(&w_shape);
        let dw_ptr = store.dev_ptr(dw_id);

        let total_threads = (total_tokens * embed_dim) as u32;
        let func = dev.get_func("embedding_backward_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&dw_ptr)
                .arg(&grad_ptr)
                .arg(&indices_ptr)
                .arg(&(total_tokens as i32))
                .arg(&(embed_dim as i32))
                .launch(launch_cfg(total_threads))
                .unwrap();
        }

        vec![Some(dw_id)]
    } else { vec![None] }
}
