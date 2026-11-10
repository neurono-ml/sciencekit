//! Predict, transform and score entry points with owned outputs.

use crate::array::{SKFfiArray, SKFfiOwnedArray};
use crate::boundary::guard::sk_ffi_guard_status;
use crate::boundary::model_entry_points::handles::sk_ffi_resolve_model;
use crate::boundary::pointers::{sk_ffi_publish_output, sk_ffi_validate_descriptor};
use crate::error_kind::SKFfiError;
use crate::model::{
    SKFfiModel, SKFfiModelKind, sk_ffi_predict_fitted, sk_ffi_score_fitted, sk_ffi_transform_fitted,
};
use crate::status::SKFfiStatus;

/// Predict responses into a library-owned output descriptor.
///
/// # Safety
///
/// The model must be a live handle, `features` a valid descriptor and
/// `output` a writable `SKFfiArray`.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_predict(
    model: *mut SKFfiModel,
    features: *const SKFfiArray,
    output: *mut SKFfiArray,
) -> SKFfiStatus {
    sk_ffi_guard_status(|| {
        let model = sk_ffi_resolve_model(model)?;
        sk_ffi_require_output(output)?;
        // SAFETY: the caller guarantees a live descriptor.
        unsafe {
            let features = sk_ffi_validate_descriptor(features)?;
            let owned = match model.kind() {
                SKFfiModelKind::F64(fitted) => {
                    let predictions = sk_ffi_predict_fitted(fitted, features.matrix_f64()?)?;
                    SKFfiOwnedArray::from_f64_vector(predictions)
                }
                SKFfiModelKind::F32(fitted) => {
                    let predictions = sk_ffi_predict_fitted(fitted, features.matrix_f32()?)?;
                    SKFfiOwnedArray::from_f32_vector(predictions)
                }
            };
            sk_ffi_store_output(output, owned)
        }
    })
}

/// Transform features into a library-owned output descriptor.
///
/// # Safety
///
/// The model must be a live handle, `features` a valid descriptor and
/// `output` a writable `SKFfiArray`.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_transform(
    model: *mut SKFfiModel,
    features: *const SKFfiArray,
    output: *mut SKFfiArray,
) -> SKFfiStatus {
    sk_ffi_guard_status(|| {
        let model = sk_ffi_resolve_model(model)?;
        sk_ffi_require_output(output)?;
        // SAFETY: the caller guarantees a live descriptor.
        unsafe {
            let features = sk_ffi_validate_descriptor(features)?;
            let owned = match model.kind() {
                SKFfiModelKind::F64(fitted) => {
                    let transformed = sk_ffi_transform_fitted(fitted, features.matrix_f64()?)?;
                    SKFfiOwnedArray::from_f64_matrix(transformed)
                }
                SKFfiModelKind::F32(fitted) => {
                    let transformed = sk_ffi_transform_fitted(fitted, features.matrix_f32()?)?;
                    SKFfiOwnedArray::from_f32_matrix(transformed)
                }
            };
            sk_ffi_store_output(output, owned)
        }
    })
}

/// Score features against targets; regression models report R².
///
/// # Safety
///
/// The model must be a live handle and every pointer valid for the call.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_score(
    model: *mut SKFfiModel,
    features: *const SKFfiArray,
    targets: *const SKFfiArray,
    score: *mut f64,
) -> SKFfiStatus {
    sk_ffi_guard_status(|| {
        let model = sk_ffi_resolve_model(model)?;
        if score.is_null() {
            return Err(SKFfiError::invalid_argument("score out-pointer is null"));
        }
        // SAFETY: the caller guarantees live descriptors and a writable out-pointer.
        unsafe {
            let features = sk_ffi_validate_descriptor(features)?;
            let targets = sk_ffi_validate_descriptor(targets)?;
            let value = match model.kind() {
                SKFfiModelKind::F64(fitted) => {
                    sk_ffi_score_fitted(fitted, features.matrix_f64()?, targets.vector_f64()?)?
                }
                SKFfiModelKind::F32(fitted) => {
                    sk_ffi_score_fitted(fitted, features.matrix_f32()?, targets.vector_f32()?)?
                        as f64
                }
            };
            *score = value;
            Ok(())
        }
    })
}

/// Reject a null output descriptor before doing any work.
fn sk_ffi_require_output(output: *mut SKFfiArray) -> Result<(), SKFfiError> {
    if output.is_null() {
        return Err(SKFfiError::invalid_argument(
            "output descriptor pointer is null",
        ));
    }
    Ok(())
}

/// Publish an owned array into the caller's output slot.
///
/// # Safety
///
/// `output` must be non-null and writable; the previous contents must be
/// released or zero-initialized.
unsafe fn sk_ffi_store_output(
    output: *mut SKFfiArray,
    owned: SKFfiOwnedArray,
) -> Result<(), SKFfiError> {
    // SAFETY: the caller checked and guaranteed a writable out-pointer.
    let published = unsafe { sk_ffi_publish_output(owned)? };
    // SAFETY: same contract as above.
    unsafe { *output = published };
    Ok(())
}
