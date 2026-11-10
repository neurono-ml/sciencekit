//! The named tensor and its dtype (spec `model-export-contracts`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod definition;
mod dtype;
mod values;

pub use definition::SKTensor;
pub use dtype::SKTensorDType;
pub use values::SKTensorValues;

#[cfg(test)]
pub(crate) use dtype::tensor_dtype_for_scalar;
