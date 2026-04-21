//! Tensor storage for the cuTile backend.
//!
//! Each tensor is an owned `cutile::Tensor<f32>` keyed by a small integer ID.
//! The design mirrors the mni framework's CUDA store so that the TypeScript
//! side (or any FFI consumer) only sees IDs.

use crate::device::runtime;
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::{Reshape, Tensor};
use cutile::tile_kernel::ToHostVecOp;
use rand::Rng;
use rand_distr::StandardNormal;

pub type TensorId = usize;

pub struct CuTileTensor {
    pub tensor: Tensor<f32>,
    pub shape: Vec<usize>,
    pub size: usize,
    pub requires_grad: bool,
    pub grad: Option<TensorId>,
}

pub struct TensorStore {
    pub(crate) tensors: Vec<Option<CuTileTensor>>,
    free_ids: Vec<TensorId>,
}

pub fn shape_size(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

impl TensorStore {
    pub fn new() -> Self {
        // Touch the runtime early so the first operation doesn't pay the
        // init cost inside a benchmark.
        let _ = runtime();
        Self {
            tensors: Vec::new(),
            free_ids: Vec::new(),
        }
    }

    fn insert(&mut self, t: CuTileTensor) -> TensorId {
        if let Some(id) = self.free_ids.pop() {
            self.tensors[id] = Some(t);
            id
        } else {
            let id = self.tensors.len();
            self.tensors.push(Some(t));
            id
        }
    }

    pub fn get(&self, id: TensorId) -> &CuTileTensor {
        self.tensors[id].as_ref().expect("tensor already freed")
    }

    pub fn get_mut(&mut self, id: TensorId) -> &mut CuTileTensor {
        self.tensors[id].as_mut().expect("tensor already freed")
    }

    pub fn tensor(&self, id: TensorId) -> &Tensor<f32> {
        &self.get(id).tensor
    }

    pub fn tensor_mut(&mut self, id: TensorId) -> &mut Tensor<f32> {
        &mut self.get_mut(id).tensor
    }

    pub fn shape(&self, id: TensorId) -> &[usize] {
        &self.get(id).shape
    }

    pub fn size(&self, id: TensorId) -> usize {
        self.get(id).size
    }

    pub fn free(&mut self, id: TensorId) {
        if self.tensors[id].take().is_some() {
            self.free_ids.push(id);
        }
    }

    // ---- Creation ops (synchronous convenience wrappers) ----

    pub fn zeros(&mut self, shape: &[usize]) -> TensorId {
        let rt = runtime();
        let tensor = api::zeros::<f32>(shape).sync_on(&rt.stream).expect("zeros");
        let size = tensor.size();
        self.insert(CuTileTensor {
            tensor,
            shape: shape.to_vec(),
            size,
            requires_grad: false,
            grad: None,
        })
    }

    pub fn ones(&mut self, shape: &[usize]) -> TensorId {
        let rt = runtime();
        let tensor = api::ones::<f32>(shape).sync_on(&rt.stream).expect("ones");
        let size = tensor.size();
        self.insert(CuTileTensor {
            tensor,
            shape: shape.to_vec(),
            size,
            requires_grad: false,
            grad: None,
        })
    }

    pub fn from_slice(&mut self, src: &[f32], shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        assert!(src.len() >= size, "source slice smaller than shape");
        let rt = runtime();
        // `copy_host_vec_to_device` takes an Arc<Vec<T>> and produces a flat 1D
        // tensor; then we reshape.  For simplicity we construct the tensor via
        // `zeros + memcpy` since reshape on the resulting tensor handles the
        // shape metadata.
        let vec = std::sync::Arc::new(src[..size].to_vec());
        let flat = api::copy_host_vec_to_device(&vec)
            .sync_on(&rt.stream)
            .expect("h2d copy");
        let reshaped = flat.reshape(shape).expect("reshape");
        self.insert(CuTileTensor {
            tensor: reshaped,
            shape: shape.to_vec(),
            size,
            requires_grad: false,
            grad: None,
        })
    }

    pub fn rand(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let mut host = vec![0.0f32; size];
        let mut rng = rand::thread_rng();
        for v in host.iter_mut() {
            *v = rng.gen::<f32>();
        }
        self.from_slice(&host, shape)
    }

    pub fn randn(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let mut host = vec![0.0f32; size];
        let mut rng = rand::thread_rng();
        for v in host.iter_mut() {
            *v = rng.sample::<f32, _>(StandardNormal);
        }
        self.from_slice(&host, shape)
    }

    pub fn to_host(&self, id: TensorId) -> Vec<f32> {
        let rt = runtime();
        // `to_host_vec` on a `Tensor<T>` consumes it, so we `dup` first to
        // get an owned copy via an async device op, then chain the host
        // copy.  `ToHostVecOp` extends `DeviceOp<Output = Tensor<T>>` with
        // a `to_host_vec` that returns a `DeviceOp<Output = Vec<T>>`.
        self.tensor(id)
            .dup()
            .to_host_vec()
            .sync_on(&rt.stream)
            .expect("d2h copy")
    }

    pub fn get_scalar(&self, id: TensorId) -> f32 {
        self.to_host(id)[0]
    }

    pub fn set_requires_grad(&mut self, id: TensorId, requires: bool) {
        self.get_mut(id).requires_grad = requires;
    }

    pub fn get_grad(&self, id: TensorId) -> Option<TensorId> {
        self.get(id).grad
    }

    pub fn ensure_grad(&mut self, id: TensorId) -> TensorId {
        if let Some(g) = self.get(id).grad {
            return g;
        }
        let shape = self.get(id).shape.clone();
        let grad_id = self.zeros(&shape);
        self.get_mut(id).grad = Some(grad_id);
        grad_id
    }

    /// Insert a freshly-created cuTile tensor (takes ownership).
    pub fn insert_tensor(&mut self, tensor: Tensor<f32>, shape: Vec<usize>) -> TensorId {
        let size = tensor.size();
        self.insert(CuTileTensor {
            tensor,
            shape,
            size,
            requires_grad: false,
            grad: None,
        })
    }
}

impl Default for TensorStore {
    fn default() -> Self {
        Self::new()
    }
}
