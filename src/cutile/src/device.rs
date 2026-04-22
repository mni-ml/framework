//! Shared cuTile runtime: a single CUDA context and stream that all tensors
//! and kernel launches share.

use cuda_core::{CudaContext, CudaStream};
use std::sync::{Arc, OnceLock};

/// Process-global cuTile runtime state.
pub struct Runtime {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
}

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

pub fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        let ctx = CudaContext::new(0).expect("CUDA context creation failed (device 0)");
        let stream = ctx.new_stream().expect("CUDA stream creation failed");
        Runtime { ctx, stream }
    })
}
