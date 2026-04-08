use std::collections::HashMap;

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub struct CachingAllocator {
    free_lists: HashMap<usize, Vec<Vec<f32>>>,
    total_allocated: usize,
}

#[cfg(any(feature = "cpu", feature = "webgpu"))]
impl CachingAllocator {
    pub fn new() -> Self {
        Self {
            free_lists: HashMap::new(),
            total_allocated: 0,
        }
    }

    pub fn alloc(&mut self, size: usize) -> Vec<f32> {
        if let Some(list) = self.free_lists.get_mut(&size) {
            if let Some(mut buf) = list.pop() {
                buf.iter_mut().for_each(|x| *x = 0.0);
                return buf;
            }
        }
        self.total_allocated += size * 4;
        vec![0.0f32; size]
    }

    pub fn dealloc(&mut self, buf: Vec<f32>) {
        let size = buf.len();
        self.free_lists.entry(size).or_default().push(buf);
    }

    pub fn clear_cache(&mut self) {
        self.free_lists.clear();
    }

    pub fn allocated_bytes(&self) -> usize {
        self.total_allocated
    }
}

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

#[cfg(feature = "cuda")]
pub struct CachingAllocator {
    free_lists: HashMap<usize, Vec<CudaSlice<f32>>>,
    total_allocated: usize,
}

#[cfg(feature = "cuda")]
impl CachingAllocator {
    pub fn new() -> Self {
        Self {
            free_lists: HashMap::new(),
            total_allocated: 0,
        }
    }

    pub fn alloc(&mut self, size: usize) -> CudaSlice<f32> {
        if let Some(list) = self.free_lists.get_mut(&size) {
            if let Some(buf) = list.pop() {
                return buf;
            }
        }
        let dev = crate::device::GpuDevice::instance();
        self.total_allocated += size * 4;
        dev.stream.alloc_zeros(size).unwrap()
    }

    pub fn dealloc(&mut self, buf: CudaSlice<f32>) {
        let size = buf.len();
        self.free_lists.entry(size).or_default().push(buf);
    }

    pub fn clear_cache(&mut self) {
        self.free_lists.clear();
    }

    pub fn allocated_bytes(&self) -> usize {
        self.total_allocated
    }
}
