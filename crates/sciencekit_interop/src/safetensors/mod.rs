//! Extended Safetensors internal model format over [`sciencekit_common::SKModelState`]
//! (PRD §8.2, spec `safetensors-format`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod header;

pub use header::{SK_SAFETENSORS_FORMAT_VERSION, SK_SAFETENSORS_PADDING_TENSOR_NAME};

#[cfg(test)]
mod safetensors_wiring_tests;
