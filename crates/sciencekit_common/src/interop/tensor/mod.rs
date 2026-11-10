//! The named tensor and its data_type (spec `model-export-contracts`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod data_type;
mod definition;
mod values;

pub use data_type::SKTensorDataType;
pub use definition::SKTensor;
pub use values::SKTensorValues;

#[cfg(test)]
pub(crate) use data_type::tensor_data_type_for_scalar;
