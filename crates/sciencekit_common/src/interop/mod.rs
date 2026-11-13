//! Format-agnostic model export/import contracts (spec `model-export-contracts`).
//!
//! A fitted model describes its learned state as estimator metadata plus named
//! tensors. Export produces that state and import reconstructs a model from it.
//! The module stays free of any file format and of `serde`: format crates own
//! their serialization.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod metadata;
mod nested;
mod parameter;
mod shared;
mod state;
mod tensor;
mod testing;
mod traits;

pub use metadata::SKModelMetadata;
pub use nested::{
    sk_embed_state, sk_extract_state, sk_nested_parameter_prefix, sk_nested_tensor_prefix,
};
pub use parameter::SKParameterValue;
pub use shared::sk_model_tensor_data_type as sk_nested_tensor_data_type;
pub use shared::{
    SK_MODEL_FORMAT_VERSION, sk_backend_parameter, sk_boolean_parameter, sk_float_parameter,
    sk_integer_parameter, sk_model_tensor_data_type, sk_parse_backend_name,
    sk_parse_backend_parameter, sk_read_boolean_tensor, sk_read_floating_tensor,
    sk_read_integer_tensor, sk_string_parameter,
};
pub use state::SKModelState;
pub use tensor::{SKTensor, SKTensorDataType, SKTensorValues};
pub use testing::{sk_assert_export_round_trip, sk_assert_state_names, sk_states_equal};
pub use traits::{SKExportable, SKImportable, sk_validate_import_metadata};

#[cfg(test)]
mod metadata_tests;
#[cfg(test)]
mod parameter_tests;
#[cfg(test)]
mod shared_tests;
#[cfg(test)]
mod state_tests;
#[cfg(test)]
mod tensor_tests;
#[cfg(test)]
mod testing_tests;
#[cfg(test)]
mod traits_tests;
#[cfg(test)]
mod validation_tests;

#[cfg(test)]
mod concurrency_tests;

#[cfg(test)]
pub(crate) mod mock_model;
