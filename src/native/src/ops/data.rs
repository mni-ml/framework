use crate::tensor::TensorId;
use rand::Rng;

#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

// GPU-resident dataset: tokenized data stored as i32 on device.
// Avoids per-step PCIe transfers for training data.

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub struct IntBuffer {
    pub data: Vec<i32>,
    pub len: usize,
}

#[cfg(feature = "cuda")]
pub struct IntBuffer {
    pub data: CudaSlice<i32>,
    pub len: usize,
}

pub struct IntStore {
    pub buffers: Vec<Option<IntBuffer>>,
    free_ids: Vec<usize>,
}

impl IntStore {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            free_ids: Vec::new(),
        }
    }

    fn insert(&mut self, buf: IntBuffer) -> usize {
        if let Some(id) = self.free_ids.pop() {
            self.buffers[id] = Some(buf);
            id
        } else {
            let id = self.buffers.len();
            self.buffers.push(Some(buf));
            id
        }
    }

    pub fn get(&self, id: usize) -> &IntBuffer {
        self.buffers[id].as_ref().expect("int buffer already freed")
    }

    pub fn free(&mut self, id: usize) {
        self.buffers[id] = None;
        self.free_ids.push(id);
    }
}

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn create_dataset(data: &[i32], int_store: &mut IntStore) -> usize {
    let buf = IntBuffer {
        len: data.len(),
        data: data.to_vec(),
    };
    int_store.insert(buf)
}

#[cfg(feature = "cuda")]
pub fn create_dataset(data: &[i32], int_store: &mut IntStore) -> usize {
    let dev = GpuDevice::instance();
    let gpu_data = dev.stream.memcpy_stod(data).unwrap();
    let buf = IntBuffer {
        len: data.len(),
        data: gpu_data,
    };
    int_store.insert(buf)
}

/// Sample a batch of (inputs, targets) windows from the dataset.
/// Returns (inputs_int_id, targets_int_id) referencing IntStore buffers.
#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub fn sample_batch(
    dataset_id: usize,
    block_size: usize,
    batch_size: usize,
    int_store: &mut IntStore,
) -> (usize, usize) {
    let ds = int_store.get(dataset_id);
    let ds_len = ds.len;
    let ds_data = &ds.data;
    let mut rng = rand::thread_rng();

    let mut inputs = vec![0i32; batch_size * block_size];
    let mut targets = vec![0i32; batch_size * block_size];

    for b in 0..batch_size {
        let start = rng.gen_range(0..ds_len - block_size - 1);
        for t in 0..block_size {
            inputs[b * block_size + t] = ds_data[start + t];
            targets[b * block_size + t] = ds_data[start + t + 1];
        }
    }

    let inp_id = int_store.insert(IntBuffer { data: inputs, len: batch_size * block_size });
    let tgt_id = int_store.insert(IntBuffer { data: targets, len: batch_size * block_size });
    (inp_id, tgt_id)
}

#[cfg(feature = "cuda")]
pub fn sample_batch(
    dataset_id: usize,
    block_size: usize,
    batch_size: usize,
    int_store: &mut IntStore,
) -> (usize, usize) {
    let dev = GpuDevice::instance();
    let ds = int_store.get(dataset_id);
    let ds_len = ds.len;
    let mut rng = rand::thread_rng();

    let offsets: Vec<i32> = (0..batch_size)
        .map(|_| rng.gen_range(0..(ds_len - block_size - 1) as i32))
        .collect();
    let offsets_gpu: CudaSlice<i32> = dev.stream.memcpy_stod(&offsets).unwrap();

    let total = batch_size * block_size;
    let inp_gpu: CudaSlice<i32> = dev.stream.alloc_zeros(total).unwrap();
    let tgt_gpu: CudaSlice<i32> = dev.stream.alloc_zeros(total).unwrap();

    let ds_ptr = dev.ptr(&int_store.get(dataset_id).data);
    let inp_ptr = dev.ptr(&inp_gpu);
    let tgt_ptr = dev.ptr(&tgt_gpu);
    let off_ptr = dev.ptr(&offsets_gpu);

    let func = dev.get_func("sample_batch_i32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&inp_ptr)
            .arg(&tgt_ptr)
            .arg(&ds_ptr)
            .arg(&(ds_len as i32))
            .arg(&(block_size as i32))
            .arg(&(batch_size as i32))
            .arg(&off_ptr)
            .launch(LaunchConfig {
                grid_dim: (((batch_size as u32) + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }

    let inp_id = int_store.insert(IntBuffer { data: inp_gpu, len: total });
    let tgt_id = int_store.insert(IntBuffer { data: tgt_gpu, len: total });
    (inp_id, tgt_id)
}
