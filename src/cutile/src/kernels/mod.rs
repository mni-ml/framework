//! cuTile kernels, organized one file per op family to mirror
//! `src/native/kernels/*.cu` in the sibling CUDA backend.
//!
//! Each submodule declares its own `#[cutile::module]` block (multiple
//! cuTile modules coexist — see NVlabs/cutile-rs `inter_module.rs`
//! example), and its contents are re-exported at this level so callers
//! continue to use `crate::kernels::add`, `crate::kernels::gemm`, etc. —
//! unchanged from the pre-split monolithic `kernels.rs`.
//!
//! Every `#[cutile::entry()]` function is compiled to PTX on first launch
//! and cached thereafter.  Elementwise kernels are compiled for a fixed
//! rank of 1 (the ops layer flattens multi-dim tensors with `TensorView`);
//! reductions and GEMM use rank-specific const generics.

pub mod activation;
pub mod adamw;
pub mod conv;
pub mod cross_entropy;
pub mod data;
pub mod dropout;
pub mod elementwise;
pub mod embedding;
pub mod flash_attention;
pub mod fused_ops;
pub mod grad_util;
pub mod kv_quant;
pub mod layernorm;
pub mod matmul;
pub mod mixed_precision;
pub mod pooling;
pub mod reduce;
pub mod softmax;

pub use activation::*;
pub use adamw::*;
pub use conv::*;
pub use cross_entropy::*;
pub use data::*;
pub use dropout::*;
pub use elementwise::*;
pub use embedding::*;
pub use flash_attention::*;
pub use fused_ops::*;
pub use grad_util::*;
pub use kv_quant::*;
pub use layernorm::*;
pub use matmul::*;
pub use mixed_precision::*;
pub use pooling::*;
pub use reduce::*;
pub use softmax::*;
