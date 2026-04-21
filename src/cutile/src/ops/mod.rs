//! cuTile-backed tensor operations.
//!
//! Each op takes tensor IDs, launches a cuTile kernel (or a composition of
//! kernels) on the shared runtime stream, and returns the ID of a freshly
//! allocated result tensor.

pub mod activation;
pub mod elementwise;
pub mod matmul;
pub mod reduce;
