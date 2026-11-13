//! Fitting and model lifecycle entry points.

use std::sync::Arc;

use ndarray::ArrayView2;

use crate::array::{SKFfiArray, SKFfiDataType};
use crate::boundary::builder_entry_points::sk_ffi_resolve_builder;
use crate::boundary::guard::{sk_ffi_guard_pointer, sk_ffi_guard_status};
use crate::boundary::pointers::{sk_ffi_handle_identifier, sk_ffi_validate_descriptor};
use crate::error_kind::SKFfiError;
use crate::handle::{
    SKFfiHandle, sk_ffi_handle_insert, sk_ffi_handle_lookup, sk_ffi_handle_remove,
};
use crate::model::{SKFfiModel, sk_ffi_fit_dense_f32, sk_ffi_fit_dense_f64};
use crate::status::SKFfiStatus;

/// Fit the configured estimator on dense features and optional targets.
///
/// Returns null on failure and records the reason in `sk_ffi_last_error`.
///
/// # Safety
///
/// The builder must be a live handle; every non-null descriptor must be valid
/// for the duration of the call.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_fit(
    builder: *mut crate::builder::SKFfiBuilder,
    features: *const SKFfiArray,
    targets: *const SKFfiArray,
) -> *mut SKFfiModel {
    sk_ffi_guard_pointer(|| {
        let builder = sk_ffi_resolve_builder(builder)?;
        let configuration = builder
            .lock()
            .map_err(|_| {
                SKFfiError::Internal(
                    "the builder lock was poisoned by an earlier panic".to_string(),
                )
            })?
            .clone();
        // SAFETY: the caller guarantees live, readable descriptors.
        unsafe {
            let features = sk_ffi_validate_descriptor(features)?;
            let targets = if targets.is_null() {
                None
            } else {
                Some(sk_ffi_validate_descriptor(targets)?)
            };
            let model = match configuration.dtype() {
                SKFfiDataType::F64 => {
                    let feature_view: ArrayView2<f64> = features.matrix_f64()?;
                    let target_view = match &targets {
                        Some(targets) => Some(targets.vector_f64()?),
                        None => None,
                    };
                    sk_ffi_fit_dense_f64(&configuration, feature_view, target_view)?
                }
                SKFfiDataType::F32 => {
                    let feature_view: ArrayView2<f32> = features.matrix_f32()?;
                    let target_view = match &targets {
                        Some(targets) => Some(targets.vector_f32()?),
                        None => None,
                    };
                    sk_ffi_fit_dense_f32(&configuration, feature_view, target_view)?
                }
                other => {
                    return Err(SKFfiError::Unsupported {
                        operation: "fit",
                        reason: format!("dtype {other:?} is not modeled"),
                    });
                }
            };
            let identifier = sk_ffi_handle_insert(SKFfiHandle::Model(Arc::new(model)));
            Ok(identifier as usize as *mut SKFfiModel)
        }
    })
}

/// Free a model. Null, unknown and already-freed handles are rejected.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_model_free(model: *mut SKFfiModel) -> SKFfiStatus {
    sk_ffi_guard_status(|| {
        let identifier = sk_ffi_handle_identifier(model);
        match sk_ffi_handle_lookup(identifier) {
            Ok(crate::handle::SKFfiResolvedHandle::Model(_)) => {}
            Ok(crate::handle::SKFfiResolvedHandle::Builder(_)) => {
                return Err(SKFfiError::InvalidHandle);
            }
            Err(status) => return Err(SKFfiError::from_status(status)),
        }
        if sk_ffi_handle_remove(identifier) {
            Ok(())
        } else {
            Err(SKFfiError::InvalidHandle)
        }
    })
}
