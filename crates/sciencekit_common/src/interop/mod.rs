//! Format-agnostic model export/import contracts (spec `model-export-contracts`).
//!
//! A fitted model describes its learned state as estimator metadata plus named
//! tensors. [`SKExportable`] produces that state and [`SKImportable`]
//! reconstructs a model from it. The module stays free of any file format and
//! of `serde`: format crates own their serialization.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod tensor;

pub use tensor::{SKTensor, SKTensorDType, SKTensorValues};

#[cfg(test)]
mod tensor_tests;
