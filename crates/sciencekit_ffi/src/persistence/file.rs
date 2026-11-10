//! Reading and writing model-state files, and importing a linear-regression
//! state back into a fitted model.

use std::path::Path;

use sciencekit_common::{SKImportable, SKModelState};
use sciencekit_linear_model::linear_regression::model::SKLinearRegressionModel;

use crate::error_kind::SKFfiError;
use crate::model::SKFfiFitted;
use crate::persistence::decoding::sk_ffi_decode_model_state;
use crate::persistence::encoding::sk_ffi_encode_model_state;

/// Write a double-precision model state to `path`.
pub fn sk_ffi_write_model_state(path: &Path, state: &SKModelState<f64>) -> Result<(), SKFfiError> {
    let bytes = sk_ffi_encode_model_state(state);
    std::fs::write(path, bytes)
        .map_err(|error| SKFfiError::Io(format!("cannot write `{}`: {error}", path.display())))
}

/// Read a double-precision model state from `path`.
pub fn sk_ffi_read_model_state(path: &Path) -> Result<SKModelState<f64>, SKFfiError> {
    let bytes = std::fs::read(path)
        .map_err(|error| SKFfiError::Io(format!("cannot read `{}`: {error}", path.display())))?;
    sk_ffi_decode_model_state(&bytes)
}

/// Reconstruct the fitted linear-regression model carried by `state`.
pub fn sk_ffi_import_linear_regression(
    state: &SKModelState<f64>,
) -> Result<SKFfiFitted<f64>, SKFfiError> {
    Ok(SKFfiFitted::LinearRegression(
        SKLinearRegressionModel::<f64>::import_state(state)?,
    ))
}
