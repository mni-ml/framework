//! cuTile Rust backend for mni-ml/framework.
//!
//! This crate is a sibling of `mni-framework-native` that replaces the cudarc
//! + NVRTC path with NVIDIA's cuTile Rust DSL.  The public surface is an
//! ID-keyed `TensorStore` and a small family of op functions; the optional
//! `napi` feature additionally exposes them as N-API entry points so that
//! the framework's TypeScript loader can consume this backend.
//!
//! Kernels live in `src/kernels.rs` (compiled to PTX on first launch by the
//! `#[cutile::module]` macro) and are wrapped in `src/ops/`.
//!
//! The shared CUDA runtime (context + stream) lives in `src/device.rs`.

pub mod device;
pub mod kernels;
pub mod ops;
pub mod tensor;

pub use crate::tensor::{shape_size, CuTileTensor, TensorId, TensorStore};

#[cfg(feature = "napi")]
mod napi_bindings {
    //! Napi glue: a global engine wrapping a `TensorStore`, plus the subset
    //! of `native`'s exports that correspond to ops we've implemented.

    use crate::tensor::{TensorId, TensorStore};
    use napi::bindgen_prelude::*;
    use napi_derive::napi;
    use parking_lot::Mutex;
    use std::sync::OnceLock;

    struct Engine {
        store: TensorStore,
    }

    static ENGINE: OnceLock<Mutex<Engine>> = OnceLock::new();

    fn engine() -> &'static Mutex<Engine> {
        ENGINE.get_or_init(|| {
            Mutex::new(Engine {
                store: TensorStore::new(),
            })
        })
    }

    // ---- Tensor creation ----

    #[napi]
    pub fn zeros(shape: Vec<i64>) -> u32 {
        let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        engine().lock().store.zeros(&shape) as u32
    }

    #[napi]
    pub fn ones(shape: Vec<i64>) -> u32 {
        let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        engine().lock().store.ones(&shape) as u32
    }

    #[napi]
    pub fn rand_tensor(shape: Vec<i64>) -> u32 {
        let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        engine().lock().store.rand(&shape) as u32
    }

    #[napi]
    pub fn randn_tensor(shape: Vec<i64>) -> u32 {
        let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        engine().lock().store.randn(&shape) as u32
    }

    #[napi]
    pub fn from_float32(data: Float32Array, shape: Vec<i64>) -> u32 {
        let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        engine().lock().store.from_slice(data.as_ref(), &shape) as u32
    }

    #[napi]
    pub fn tensor_shape(id: u32) -> Vec<i64> {
        let eng = engine().lock();
        eng.store
            .shape(id as TensorId)
            .iter()
            .map(|&s| s as i64)
            .collect()
    }

    #[napi]
    pub fn tensor_size(id: u32) -> i64 {
        engine().lock().store.size(id as TensorId) as i64
    }

    #[napi]
    pub fn to_float32(id: u32) -> Float32Array {
        let data = engine().lock().store.to_host(id as TensorId);
        Float32Array::new(data)
    }

    #[napi]
    pub fn get_scalar(id: u32) -> f64 {
        engine().lock().store.get_scalar(id as TensorId) as f64
    }

    #[napi]
    pub fn free_tensor(id: u32) {
        engine().lock().store.free(id as TensorId);
    }

    // ---- Ops ----

    #[napi]
    pub fn add(a: u32, b: u32) -> u32 {
        let mut eng = engine().lock();
        crate::ops::elementwise::add(&mut eng.store, a as TensorId, b as TensorId) as u32
    }

    #[napi]
    pub fn sub(a: u32, b: u32) -> u32 {
        let mut eng = engine().lock();
        crate::ops::elementwise::sub(&mut eng.store, a as TensorId, b as TensorId) as u32
    }

    #[napi]
    pub fn mul(a: u32, b: u32) -> u32 {
        let mut eng = engine().lock();
        crate::ops::elementwise::mul(&mut eng.store, a as TensorId, b as TensorId) as u32
    }

    #[napi]
    pub fn div(a: u32, b: u32) -> u32 {
        let mut eng = engine().lock();
        crate::ops::elementwise::div(&mut eng.store, a as TensorId, b as TensorId) as u32
    }

    #[napi]
    pub fn neg(a: u32) -> u32 {
        let mut eng = engine().lock();
        crate::ops::elementwise::neg(&mut eng.store, a as TensorId) as u32
    }

    #[napi]
    pub fn mul_scalar(a: u32, s: f64) -> u32 {
        let mut eng = engine().lock();
        crate::ops::elementwise::mul_scalar(&mut eng.store, a as TensorId, s as f32) as u32
    }

    #[napi]
    pub fn saxpy(a: f64, x: u32, y: u32) -> u32 {
        let mut eng = engine().lock();
        crate::ops::elementwise::saxpy(&mut eng.store, a as f32, x as TensorId, y as TensorId)
            as u32
    }

    #[napi]
    pub fn relu(a: u32) -> u32 {
        let mut eng = engine().lock();
        crate::ops::activation::relu(&mut eng.store, a as TensorId) as u32
    }

    #[napi]
    pub fn relu_backward(x: u32, grad: u32) -> u32 {
        let mut eng = engine().lock();
        crate::ops::activation::relu_backward(&mut eng.store, x as TensorId, grad as TensorId)
            as u32
    }

    #[napi]
    pub fn matmul(a: u32, b: u32) -> u32 {
        let mut eng = engine().lock();
        crate::ops::matmul::matmul(&mut eng.store, a as TensorId, b as TensorId) as u32
    }

    #[napi]
    pub fn sum_all(a: u32) -> u32 {
        let mut eng = engine().lock();
        crate::ops::reduce::sum_all(&mut eng.store, a as TensorId) as u32
    }

    #[napi]
    pub fn mean_all(a: u32) -> u32 {
        let mut eng = engine().lock();
        crate::ops::reduce::mean_all(&mut eng.store, a as TensorId) as u32
    }
}
