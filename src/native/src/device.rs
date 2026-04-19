#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr};
#[cfg(feature = "cuda")]
use cudarc::cublas::safe::CudaBlas;
#[cfg(feature = "cuda")]
use cudarc::nvrtc;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "cuda")]
pub struct GpuDevice {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
    functions: HashMap<String, CudaFunction>,
}

#[cfg(feature = "cuda")]
unsafe impl Send for GpuDevice {}
#[cfg(feature = "cuda")]
unsafe impl Sync for GpuDevice {}

#[cfg(feature = "cuda")]
static GPU: OnceLock<GpuDevice> = OnceLock::new();

#[cfg(feature = "cuda")]
impl GpuDevice {
    pub fn instance() -> &'static Self {
        GPU.get_or_init(Self::init)
    }

    fn init() -> Self {
        let ctx = CudaContext::new(0).expect("CUDA context creation failed");
        unsafe { ctx.disable_event_tracking(); }
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).expect("cuBLAS handle creation failed");

        let mut dev = Self {
            ctx,
            stream,
            blas,
            functions: HashMap::new(),
        };
        dev.load_all_kernels();
        dev
    }

    fn compile_and_load(&mut self, name: &str, source: &str, kernel_names: &[&str]) {
        let ptx = nvrtc::compile_ptx(source)
            .unwrap_or_else(|e| panic!("Failed to compile {name}: {e}"));
        let module = self
            .ctx
            .load_module(ptx)
            .unwrap_or_else(|e| panic!("Failed to load module {name}: {e}"));
        for &kname in kernel_names {
            let func = module
                .load_function(kname)
                .unwrap_or_else(|e| panic!("Failed to get function {kname}: {e}"));
            self.functions.insert(kname.to_string(), func);
        }
    }

    fn load_all_kernels(&mut self) {
        self.compile_and_load(
            "elementwise",
            include_str!("../kernels/elementwise.cu"),
            &[
                "add_f32",
                "sub_f32",
                "mul_f32",
                "neg_f32",
                "mul_scalar_f32",
                "exp_f32",
                "log_f32",
                "add_bias_f32",
                "fill_f32",
                "div_f32",
                "copy_f32",
                "permute_f32",
                "broadcast_add_f32",
                "broadcast_mul_f32",
                "sum_reduce_all_f32",
                "lt_f32",
                "eq_f32",
                "gt_f32",
                "is_close_f32",
                "pow_f32",
                "pow_backward_f32",
                "div_backward_a_f32",
                "div_backward_b_f32",
            ],
        );
        self.compile_and_load(
            "activation",
            include_str!("../kernels/activation.cu"),
            &[
                "gelu_forward_f32",
                "gelu_backward_f32",
                "sigmoid_forward_f32",
                "sigmoid_backward_f32",
                "relu_forward_f32",
                "relu_backward_f32",
            ],
        );
        self.compile_and_load(
            "reduce",
            include_str!("../kernels/reduce.cu"),
            &[
                "sum_along_dim_f32",
                "mean_along_dim_f32",
                "max_along_dim_f32",
                "sum_broadcast_f32",
            ],
        );
        self.compile_and_load(
            "layernorm",
            include_str!("../kernels/layernorm.cu"),
            &["layernorm_forward_f32", "layernorm_backward_f32"],
        );
        self.compile_and_load(
            "softmax",
            include_str!("../kernels/softmax.cu"),
            &["softmax_forward_f32", "softmax_backward_f32"],
        );
        self.compile_and_load(
            "cross_entropy",
            include_str!("../kernels/cross_entropy.cu"),
            &["cross_entropy_forward_f32", "cross_entropy_backward_f32"],
        );
        self.compile_and_load(
            "embedding",
            include_str!("../kernels/embedding.cu"),
            &["embedding_forward_f32", "embedding_backward_f32"],
        );
        self.compile_and_load(
            "dropout",
            include_str!("../kernels/dropout.cu"),
            &["dropout_apply_f32", "dropout_backward_f32"],
        );
        self.compile_and_load(
            "adamw",
            include_str!("../kernels/adamw.cu"),
            &["adamw_step_f32"],
        );
        self.compile_and_load(
            "grad_util",
            include_str!("../kernels/grad_util.cu"),
            &["grad_norm_sq_partial_f32", "grad_clip_f32"],
        );
        self.compile_and_load(
            "data",
            include_str!("../kernels/data.cu"),
            &["sample_batch_i32"],
        );
        self.compile_and_load(
            "flash_attention",
            include_str!("../kernels/flash_attention.cu"),
            &["flash_attention_forward_f32", "flash_attention_backward_f32"],
        );
        self.compile_and_load(
            "fused_ops",
            include_str!("../kernels/fused_ops.cu"),
            &["residual_layernorm_forward_f32", "bias_gelu_forward_f32", "bias_gelu_backward_f32"],
        );
        self.compile_and_load(
            "conv",
            include_str!("../kernels/conv.cu"),
            &[
                "conv1d_forward_f32", "conv1d_backward_input_f32", "conv1d_backward_weight_f32",
                "conv2d_forward_f32", "conv2d_backward_input_f32", "conv2d_backward_weight_f32",
            ],
        );
        self.compile_and_load(
            "pooling",
            include_str!("../kernels/pooling.cu"),
            &[
                "avgpool2d_forward_f32", "avgpool2d_backward_f32",
                "maxpool2d_forward_f32", "maxpool2d_backward_f32",
            ],
        );
        self.compile_and_load(
            "mixed_precision",
            include_str!("../kernels/mixed_precision.cu"),
            &["f32_to_bf16", "bf16_to_f32", "scale_f32", "check_inf_nan_f32"],
        );
        self.compile_and_load(
            "kv_quant",
            include_str!("../kernels/kv_quant.cu"),
            &[
                "compute_rowwise_scale_f32",
                "quantize_rowwise_i8_f32",
                "dequantize_rowwise_i8_f32",
            ],
        );
    }

    pub fn get_func(&self, name: &str) -> &CudaFunction {
        self.functions
            .get(name)
            .unwrap_or_else(|| panic!("Kernel function '{name}' not found"))
    }

    pub fn ptr<T>(&self, slice: &CudaSlice<T>) -> u64
    where
        CudaSlice<T>: DevicePtr<T>,
    {
        let (ptr, _guard) = slice.device_ptr(&self.stream);
        ptr
    }
}

#[cfg(feature = "cpu")]
pub struct GpuDevice;

#[cfg(feature = "cpu")]
impl GpuDevice {
    pub fn instance() -> &'static Self {
        static STUB: GpuDevice = GpuDevice;
        &STUB
    }
}

// ===========================================================================
// WebGPU backend via wgpu
// ===========================================================================

#[cfg(feature = "webgpu")]
use std::collections::HashMap;
#[cfg(feature = "webgpu")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "webgpu")]
pub struct GpuDevice {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    shader_modules: HashMap<String, wgpu::ShaderModule>,
}

#[cfg(feature = "webgpu")]
unsafe impl Send for GpuDevice {}
#[cfg(feature = "webgpu")]
unsafe impl Sync for GpuDevice {}

#[cfg(feature = "webgpu")]
static GPU: OnceLock<GpuDevice> = OnceLock::new();

#[cfg(feature = "webgpu")]
impl GpuDevice {
    pub fn instance() -> &'static Self {
        GPU.get_or_init(Self::init)
    }

    fn init() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("No suitable GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("mni-framework"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("Failed to create GPU device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let mut dev = Self {
            device,
            queue,
            pipelines: HashMap::new(),
            shader_modules: HashMap::new(),
        };
        dev.load_all_shaders();
        dev
    }

    fn compile_shader(&mut self, name: &str, source: &str, entry_points: &[&str]) {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        for &ep in entry_points {
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: None,
                module: &module,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            });
            self.pipelines.insert(ep.to_string(), pipeline);
        }
        self.shader_modules.insert(name.to_string(), module);
    }

    fn load_all_shaders(&mut self) {
        self.compile_shader(
            "elementwise",
            include_str!("../shaders/elementwise.wgsl"),
            &[
                "add_f32", "sub_f32", "mul_f32", "neg_f32", "mul_scalar_f32",
                "exp_f32", "log_f32", "div_f32", "fill_f32", "copy_f32",
                "lt_f32", "eq_f32", "gt_f32", "pow_f32",
                "sigmoid_forward_f32", "sigmoid_backward_f32",
            ],
        );
        self.compile_shader(
            "activation",
            include_str!("../shaders/activation.wgsl"),
            &["gelu_forward_f32", "gelu_backward_f32", "relu_forward_f32", "relu_backward_f32"],
        );
        self.compile_shader(
            "reduce",
            include_str!("../shaders/reduce.wgsl"),
            &["sum_along_dim_f32", "mean_along_dim_f32", "max_along_dim_f32"],
        );
        self.compile_shader(
            "layernorm",
            include_str!("../shaders/layernorm.wgsl"),
            &["layernorm_forward_f32", "layernorm_backward_f32"],
        );
        self.compile_shader(
            "softmax",
            include_str!("../shaders/softmax.wgsl"),
            &["softmax_forward_f32", "softmax_backward_f32"],
        );
        self.compile_shader(
            "cross_entropy",
            include_str!("../shaders/cross_entropy.wgsl"),
            &["cross_entropy_forward_f32", "cross_entropy_backward_f32"],
        );
        self.compile_shader(
            "embedding",
            include_str!("../shaders/embedding.wgsl"),
            &["embedding_forward_f32", "embedding_backward_f32"],
        );
        self.compile_shader(
            "dropout",
            include_str!("../shaders/dropout.wgsl"),
            &["dropout_apply_f32"],
        );
        self.compile_shader(
            "adamw",
            include_str!("../shaders/adamw.wgsl"),
            &["adamw_step_f32"],
        );
        self.compile_shader(
            "grad_util",
            include_str!("../shaders/grad_util.wgsl"),
            &["grad_norm_sq_partial_f32", "grad_clip_f32"],
        );
        self.compile_shader(
            "conv",
            include_str!("../shaders/conv.wgsl"),
            &[
                "conv1d_forward_f32", "conv1d_backward_input_f32", "conv1d_backward_weight_f32",
                "conv2d_forward_f32", "conv2d_backward_input_f32", "conv2d_backward_weight_f32",
            ],
        );
        self.compile_shader(
            "pooling",
            include_str!("../shaders/pooling.wgsl"),
            &["avgpool2d_forward_f32", "avgpool2d_backward_f32", "maxpool2d_forward_f32"],
        );
    }

    pub fn get_pipeline(&self, name: &str) -> &wgpu::ComputePipeline {
        self.pipelines
            .get(name)
            .unwrap_or_else(|| panic!("Shader pipeline '{name}' not found"))
    }

    pub fn create_buffer_init(&self, data: &[f32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn create_buffer_zeros(&self, size: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn read_buffer(&self, buffer: &wgpu::Buffer, size: usize) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (size * 4) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result: Result<(), wgpu::BufferAsyncError>| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let view = slice.get_mapped_range();
        let data: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();
        data
    }
}
