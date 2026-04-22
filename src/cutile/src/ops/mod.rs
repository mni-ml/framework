//! cuTile-backed tensor operations.
//!
//! Each op takes tensor IDs, launches a cuTile kernel (or a composition of
//! kernels) on the shared runtime stream, and returns the ID of a freshly
//! allocated result tensor.

pub mod activation;
pub mod attention;
pub mod conv;
pub mod data;
pub mod dropout;
pub mod elementwise;
pub mod embedding;
pub mod fused;
pub mod grad_util;
pub mod kv_quant;
pub mod loss;
pub mod matmul;
pub mod mixed_precision;
pub mod norm;
pub mod optimizer;
pub mod pooling;
pub mod reduce;
pub mod softmax;
