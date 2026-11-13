//! The OpenCL kernel catalogue (spec `opencl-backend`, requirement *OpenCL
//! kernels*).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every kernel lives in its own file and shares the launch plumbing
//! in `common`.

mod common;
mod elementwise;
mod gemm;
mod pairwise;
mod source;
mod tree;

pub use pairwise::SKOpenClDistanceMetric;
pub use tree::SK_OPENCL_TREE_NODE_COLUMNS;

pub(crate) use elementwise::{run_combine, run_transform};
pub(crate) use gemm::run as run_gemm;
pub(crate) use pairwise::run as run_pairwise_distance;
pub(crate) use tree::run as run_tree_predict;

#[cfg(test)]
mod kernels_tests;
