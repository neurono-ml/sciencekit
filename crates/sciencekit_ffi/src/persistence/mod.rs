//! Pure dispatcher: the on-disk model-state format. Carries no implementation
//! logic.

mod decoding;
mod encoding;
mod file;
mod reader;

pub use file::{
    sk_ffi_import_linear_regression, sk_ffi_read_model_state, sk_ffi_write_model_state,
};

#[cfg(test)]
mod persistence_tests;
