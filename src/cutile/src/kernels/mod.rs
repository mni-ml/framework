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
pub mod dropout;
pub mod elementwise;
pub mod grad_util;
pub mod matmul;
pub mod mixed_precision;
pub mod reduce;

pub use activation::*;
pub use adamw::*;
pub use dropout::*;
pub use elementwise::*;
pub use grad_util::*;
pub use matmul::*;
pub use mixed_precision::*;
pub use reduce::*;
