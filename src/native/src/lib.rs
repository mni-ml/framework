mod allocator;
mod autograd;
mod device;
mod ops;
mod tensor;
mod utils;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use parking_lot::Mutex;
use std::sync::OnceLock;

use crate::autograd::Tape;
use crate::ops::data::IntStore;
use crate::tensor::{TensorId, TensorStore};

struct Engine {
    store: TensorStore,
    tape: Tape,
    int_store: IntStore,
}

static ENGINE: OnceLock<Mutex<Engine>> = OnceLock::new();

fn engine() -> &'static Mutex<Engine> {
    ENGINE.get_or_init(|| {
        Mutex::new(Engine {
            store: TensorStore::new(),
            tape: Tape::new(),
            int_store: IntStore::new(),
        })
    })
}

// ---------------------------------------------------------------------------
// Tensor creation
// ---------------------------------------------------------------------------

#[napi]
pub fn zeros(shape: Vec<i64>) -> u32 {
    let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    let mut eng = engine().lock();
    eng.store.zeros(&shape) as u32
}

#[napi]
pub fn ones(shape: Vec<i64>) -> u32 {
    let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    let mut eng = engine().lock();
    eng.store.ones(&shape) as u32
}

#[napi]
pub fn rand_tensor(shape: Vec<i64>) -> u32 {
    let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    let mut eng = engine().lock();
    eng.store.rand(&shape) as u32
}

#[napi]
pub fn randn_tensor(shape: Vec<i64>) -> u32 {
    let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    let mut eng = engine().lock();
    eng.store.randn(&shape) as u32
}

#[napi]
pub fn from_float32(data: Float32Array, shape: Vec<i64>) -> u32 {
    let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    let mut eng = engine().lock();
    eng.store.from_slice(data.as_ref(), &shape) as u32
}

#[napi]
pub fn tensor_shape(id: u32) -> Vec<i64> {
    let eng = engine().lock();
    eng.store.shape(id as TensorId).iter().map(|&s| s as i64).collect()
}

#[napi]
pub fn tensor_size(id: u32) -> i64 {
    let eng = engine().lock();
    eng.store.size(id as TensorId) as i64
}

#[napi]
pub fn to_float32(id: u32) -> Float32Array {
    let eng = engine().lock();
    let data = eng.store.to_host(id as TensorId);
    Float32Array::new(data)
}

#[napi]
pub fn get_scalar(id: u32) -> f64 {
    let eng = engine().lock();
    eng.store.get_scalar(id as TensorId) as f64
}

#[napi]
pub fn free_tensor(id: u32) {
    let mut eng = engine().lock();
    eng.store.free(id as TensorId);
}

// ---------------------------------------------------------------------------
// Parameter management (marks a tensor as a leaf requiring grad)
// ---------------------------------------------------------------------------

#[napi]
pub fn set_requires_grad(id: u32, requires: bool) {
    let mut eng = engine().lock();
    eng.store.set_requires_grad(id as TensorId, requires);
}

#[napi]
pub fn get_grad(id: u32) -> Option<u32> {
    let eng = engine().lock();
    eng.store.get_grad(id as TensorId).map(|g| g as u32)
}

// ---------------------------------------------------------------------------
// Autograd
// ---------------------------------------------------------------------------

#[napi]
pub fn backward(loss_id: u32) {
    let mut eng = engine().lock();
    let Engine { ref mut store, ref mut tape, ref int_store, .. } = *eng;
    let old_tape = std::mem::replace(tape, Tape::new());
    old_tape.backward(loss_id as TensorId, store, int_store);
}

#[napi]
pub fn zero_grad(param_ids: Vec<u32>) {
    let mut eng = engine().lock();
    let Engine { ref mut store, .. } = *eng;
    for &id in &param_ids {
        store.zero_grad(id as TensorId);
    }
}

#[napi]
pub fn no_grad_start() {
    let mut eng = engine().lock();
    eng.tape.set_enabled(false);
}

#[napi]
pub fn no_grad_end() {
    let mut eng = engine().lock();
    eng.tape.set_enabled(true);
}

// ---------------------------------------------------------------------------
// Ops (forward — each records to tape if enabled)
// ---------------------------------------------------------------------------

#[napi]
pub fn add(a: u32, b: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::add(a as TensorId, b as TensorId, store, tape) as u32
}

#[napi]
pub fn mul(a: u32, b: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::mul(a as TensorId, b as TensorId, store, tape) as u32
}

#[napi]
pub fn sub(a: u32, b: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::sub(a as TensorId, b as TensorId, store, tape) as u32
}

#[napi]
pub fn neg(a: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::neg(a as TensorId, store, tape) as u32
}

#[napi]
pub fn mul_scalar(a: u32, s: f64) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::mul_scalar(a as TensorId, s as f32, store, tape) as u32
}

#[napi]
pub fn matmul(a: u32, b: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::matmul::matmul(a as TensorId, b as TensorId, store, tape) as u32
}

#[napi]
pub fn gelu(a: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::activation::gelu_forward(a as TensorId, store, tape) as u32
}

#[napi]
pub fn relu(a: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::activation::relu_forward(a as TensorId, store, tape) as u32
}

#[napi]
pub fn exp_op(a: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::exp(a as TensorId, store, tape) as u32
}

#[napi]
pub fn log_op(a: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::log(a as TensorId, store, tape) as u32
}

#[napi]
pub fn sin_op(a: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::sin(a as TensorId, store, tape) as u32
}

#[napi]
pub fn cos_op(a: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::cos(a as TensorId, store, tape) as u32
}

#[napi]
pub fn sqrt_op(a: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::sqrt(a as TensorId, store, tape) as u32
}

#[napi]
pub fn sum_op(a: u32, dim: i64) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::reduce::sum(a as TensorId, dim as i32, store, tape) as u32
}

#[napi]
pub fn mean_op(a: u32, dim: i64) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::reduce::mean(a as TensorId, dim as i32, store, tape) as u32
}

#[napi]
pub fn max_op(a: u32, dim: i64) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::reduce::max(a as TensorId, dim as i32, store, tape) as u32
}

#[napi]
pub fn sum_all(a: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    let shape = store.shape(a as TensorId).to_vec();
    let mut current = a as TensorId;
    for d in (0..shape.len()).rev() {
        current = ops::reduce::sum(current, d as i32, store, tape);
    }
    current as u32
}

#[napi]
pub fn mean_all(a: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    let shape = store.shape(a as TensorId).to_vec();
    let mut current = a as TensorId;
    for d in (0..shape.len()).rev() {
        current = ops::reduce::mean(current, d as i32, store, tape);
    }
    current as u32
}

#[napi]
pub fn view(a: u32, shape: Vec<i64>) -> u32 {
    let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::layout::view(a as TensorId, &shape, store, tape) as u32
}

#[napi]
pub fn permute(a: u32, dims: Vec<i64>) -> u32 {
    let dims: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::layout::permute(a as TensorId, &dims, store, tape) as u32
}

#[napi]
pub fn contiguous(a: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::layout::contiguous(a as TensorId, store, tape) as u32
}

#[napi]
pub fn softmax_op(a: u32, dim: i64) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::norm::softmax(a as TensorId, dim as i32, store, tape) as u32
}

#[napi]
pub fn layernorm_op(x: u32, gamma: u32, beta: u32, eps: f64) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::norm::layernorm(
        x as TensorId, gamma as TensorId, beta as TensorId, eps as f32, store, tape,
    ) as u32
}

#[napi]
pub fn embedding_forward(weight: u32, indices: Vec<i64>, batch: i64, seq_len: i64) -> u32 {
    let indices: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::embedding::embedding_forward(
        weight as TensorId, &indices, batch as usize, seq_len as usize, store, tape,
    ) as u32
}

#[napi]
pub fn cross_entropy_loss(logits: u32, targets: Vec<i64>) -> u32 {
    let targets: Vec<usize> = targets.iter().map(|&t| t as usize).collect();
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::loss::cross_entropy(
        logits as TensorId, &targets, store, tape,
    ) as u32
}

#[napi]
pub fn div(a: u32, b: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::div(a as TensorId, b as TensorId, store, tape) as u32
}

#[napi]
pub fn lt(a: u32, b: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::lt(a as TensorId, b as TensorId, store, tape) as u32
}

#[napi]
pub fn eq_op(a: u32, b: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::eq_op(a as TensorId, b as TensorId, store, tape) as u32
}

#[napi]
pub fn gt(a: u32, b: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::gt(a as TensorId, b as TensorId, store, tape) as u32
}

#[napi]
pub fn is_close(a: u32, b: u32, tol: f64) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::is_close(a as TensorId, b as TensorId, tol as f32, store, tape) as u32
}

#[napi]
pub fn sigmoid(a: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::activation::sigmoid_forward(a as TensorId, store, tape) as u32
}

#[napi]
pub fn pow_op(a: u32, exponent: f64) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::elementwise::pow(a as TensorId, exponent as f32, store, tape) as u32
}

#[napi]
pub fn dropout_op(x: u32, rate: f64, training: bool) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::dropout::dropout_forward(
        x as TensorId, rate as f32, training, store, tape,
    ) as u32
}

// ---------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------

#[napi]
pub fn adamw_step(
    param_ids: Vec<u32>,
    lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64, step: i64,
) {
    let mut eng = engine().lock();
    let ids: Vec<TensorId> = param_ids.iter().map(|&id| id as TensorId).collect();
    ops::optimizer::adamw_step(
        &ids,
        lr as f32, beta1 as f32, beta2 as f32, eps as f32, weight_decay as f32,
        step as u32, &mut eng.store,
    );
}

#[napi]
pub fn grad_norm(param_ids: Vec<u32>) -> f64 {
    let eng = engine().lock();
    let ids: Vec<TensorId> = param_ids.iter().map(|&id| id as TensorId).collect();
    ops::optimizer::grad_norm(&ids, &eng.store) as f64
}

#[napi]
pub fn clip_grad_norm(param_ids: Vec<u32>, max_norm: f64) {
    let mut eng = engine().lock();
    let ids: Vec<TensorId> = param_ids.iter().map(|&id| id as TensorId).collect();
    ops::optimizer::clip_grad_norm(&ids, max_norm as f32, &mut eng.store);
}

#[napi]
pub fn clip_and_step(
    param_ids: Vec<u32>,
    lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64,
    step: i64, max_grad_norm: f64,
) -> f64 {
    let mut eng = engine().lock();
    let ids: Vec<TensorId> = param_ids.iter().map(|&id| id as TensorId).collect();
    ops::optimizer::clip_and_step(
        &ids,
        lr as f32, beta1 as f32, beta2 as f32, eps as f32, weight_decay as f32,
        step as u32, max_grad_norm as f32,
        &mut eng.store,
    ) as f64
}

// ---------------------------------------------------------------------------
// Mixed precision (GradScaler support)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[napi]
pub fn scale_grads(param_ids: Vec<u32>, inv_scale: f64) -> bool {
    let mut eng = engine().lock();
    let ids: Vec<TensorId> = param_ids.iter().map(|&id| id as TensorId).collect();
    ops::mixed_precision::scale_grads(&ids, inv_scale as f32, &mut eng.store)
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#[napi]
pub fn reset_engine() {
    let mut eng = engine().lock();
    eng.store = TensorStore::new();
    eng.tape = Tape::new();
    eng.int_store = IntStore::new();
}

// ---------------------------------------------------------------------------
// GPU data pipeline (i32 tensors for indices/targets) — CUDA only
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[napi]
pub fn create_dataset(data: Int32Array) -> u32 {
    let mut eng = engine().lock();
    ops::data::create_dataset(data.as_ref(), &mut eng.int_store) as u32
}

#[cfg(feature = "cuda")]
#[napi]
pub fn sample_batch(dataset_id: u32, block_size: i64, batch_size: i64) -> Vec<u32> {
    let mut eng = engine().lock();
    let (inp, tgt) = ops::data::sample_batch(
        dataset_id as usize, block_size as usize, batch_size as usize,
        &mut eng.int_store,
    );
    vec![inp as u32, tgt as u32]
}

#[cfg(feature = "cuda")]
#[napi]
pub fn free_int_buffer(id: u32) {
    let mut eng = engine().lock();
    eng.int_store.free(id as usize);
}

#[cfg(feature = "cuda")]
#[napi]
pub fn embedding_forward_gpu(weight: u32, int_buf_id: u32, batch: i64, seq_len: i64) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, int_store, .. } = &mut *e;
    ops::embedding::embedding_forward_gpu(
        weight as TensorId,
        int_buf_id as usize,
        batch as usize,
        seq_len as usize,
        int_store, store, tape,
    ) as u32
}

#[cfg(feature = "cuda")]
#[napi]
pub fn residual_layernorm(x: u32, residual: u32, gamma: u32, beta: u32, eps: f64) -> u32 {
    let mut eng = engine().lock();
    let Engine { store, tape, .. } = &mut *eng;
    ops::fused::residual_layernorm(
        x as TensorId, residual as TensorId,
        gamma as TensorId, beta as TensorId, eps as f32,
        store, tape,
    ) as u32
}

#[cfg(feature = "cuda")]
#[napi]
pub fn bias_gelu(x: u32, bias: u32) -> u32 {
    let mut eng = engine().lock();
    let Engine { store, tape, .. } = &mut *eng;
    ops::fused::bias_gelu(x as TensorId, bias as TensorId, store, tape) as u32
}

#[napi]
pub fn conv1d_forward(input: u32, weight: u32, stride: i64, padding: i64) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::conv::conv1d_forward(
        input as TensorId, weight as TensorId,
        stride as usize, padding as usize, store, tape,
    ) as u32
}

#[napi]
pub fn conv2d_forward(input: u32, weight: u32, stride: i64, padding: i64) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::conv::conv2d_forward(
        input as TensorId, weight as TensorId,
        stride as usize, padding as usize, store, tape,
    ) as u32
}

#[napi]
pub fn avgpool2d(input: u32, kh: i64, kw: i64) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::pooling::avgpool2d_forward(input as TensorId, kh as usize, kw as usize, store, tape) as u32
}

#[napi]
pub fn maxpool2d(input: u32, kh: i64, kw: i64) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::pooling::maxpool2d_forward(input as TensorId, kh as usize, kw as usize, store, tape) as u32
}

#[napi]
pub fn tile(input: u32, reps: Vec<i64>) -> u32 {
    let reps: Vec<usize> = reps.iter().map(|&r| r as usize).collect();
    let mut e = engine().lock();
    let Engine { store, tape, .. } = &mut *e;
    ops::pooling::tile(input as TensorId, &reps, store, tape) as u32
}

#[cfg(feature = "cuda")]
#[napi]
pub fn flash_attention(q: u32, k: u32, v: u32, scale: f64, causal: bool) -> u32 {
    let mut eng = engine().lock();
    let Engine { store, tape, .. } = &mut *eng;
    ops::attention::flash_attention(
        q as TensorId, k as TensorId, v as TensorId,
        scale as f32, causal, store, tape,
    ) as u32
}

#[cfg(feature = "cuda")]
#[napi]
pub fn cross_entropy_loss_gpu(logits: u32, int_buf_id: u32) -> u32 {
    let mut e = engine().lock();
    let Engine { store, tape, int_store, .. } = &mut *e;
    let shape = store.shape(logits as TensorId).to_vec();
    let v = shape[shape.len() - 1];
    let bt = store.size(logits as TensorId) / v;
    let flat_logits_id = ops::layout::view(logits as TensorId, &[bt, v], store, tape);
    ops::loss::cross_entropy_gpu(
        flat_logits_id,
        int_buf_id as usize,
        int_store, store, tape,
    ) as u32
}

#[napi]
pub fn gc_tensors(keep_ids: Vec<u32>) {
    let mut eng = engine().lock();
    let keep: std::collections::HashSet<usize> =
        keep_ids.iter().map(|&id| id as usize).collect();
    let len = eng.store.tensors.len();
    for id in 0..len {
        if !keep.contains(&id) && eng.store.tensors[id].is_some() {
            eng.store.free(id);
        }
    }
    eng.store.clear_alloc_cache();
    eng.tape = Tape::new();
}
