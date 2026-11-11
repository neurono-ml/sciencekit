//! The closed kernel catalogue (spec `gpu-compute-contracts`).
//!
//! The catalogue is defined in `sciencekit_common` so execution planning can
//! filter devices by the kernel an operation requires without depending on this
//! crate; this module re-exports it as the public dispatch vocabulary.

pub use sciencekit_common::device::SKGpuKernel;
