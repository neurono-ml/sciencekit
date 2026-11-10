//! File persistence for fitted models (spec `ffi-abi`).
//!
//! Persistence round-trips an [`SKModelState`](sciencekit_common::SKModelState)
//! through the interop contract: the model exports its state, the ABI writes a
//! documented little-endian file, and loading imports the state back. The
//! concrete binary format lives in [`crate::persistence`].

use std::path::Path;

use crate::error_kind::SKFfiError;
use crate::model::definition::{SKFfiFitted, SKFfiModel};

/// Save a fitted model to `path`, through its interop state.
pub fn sk_ffi_save_model_file(model: &SKFfiModel, path: &Path) -> Result<(), SKFfiError> {
    match model.kind() {
        crate::model::definition::SKFfiModelKind::F64(fitted) => {
            sk_ffi_save_fitted_f64(fitted, path)
        }
        crate::model::definition::SKFfiModelKind::F32(_) => Err(SKFfiError::Unsupported {
            operation: "model_save",
            reason: "persistence currently covers double-precision models only".to_string(),
        }),
    }
}

/// Load a fitted model from `path`, through its interop state.
pub fn sk_ffi_load_model_file(path: &Path) -> Result<SKFfiModel, SKFfiError> {
    let state = crate::persistence::sk_ffi_read_model_state(path)?;
    let fitted = crate::persistence::sk_ffi_import_linear_regression(&state)?;
    Ok(SKFfiModel::from_f64(fitted))
}

/// Export and write the double-precision state of one fitted model.
fn sk_ffi_save_fitted_f64(fitted: &SKFfiFitted<f64>, path: &Path) -> Result<(), SKFfiError> {
    use sciencekit_common::SKExportable;

    match fitted {
        SKFfiFitted::LinearRegression(model) => {
            crate::persistence::sk_ffi_write_model_state(path, &model.export_state()?)
        }
        SKFfiFitted::StandardScaler(_) => Err(SKFfiError::Unsupported {
            operation: "model_save",
            reason: "the standard scaler does not implement the export contract yet".to_string(),
        }),
    }
}
