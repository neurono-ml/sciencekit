//! The OpenCL backend operations (spec `opencl-backend`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares the operation
//! submodules; the contract path, the typed kernel path, the elementwise path,
//! the tree path and their shared checks live in their own files.

mod contract;
mod elementwise_operations;
mod kernel_operations;
mod support;
mod tree_operations;
