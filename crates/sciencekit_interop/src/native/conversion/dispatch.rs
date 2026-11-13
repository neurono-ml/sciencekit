//! Dispatch from a neutral state to a native fitted model
//! (spec `native-model-import`).
//!
//! Validation always precedes construction: the estimator name selects the
//! family, the exact tensor-name set is checked, and only then does the existing
//! `SKImportable` implementation rebuild the model. A rejected conversion never
//! yields a partially initialized value.

use sciencekit_common::{SKFloat, SKImportable, SKModelState};

use super::classifiers::convert_classifier;
use super::preprocessing::{convert_clusterer, convert_transformer};
use super::regressors::convert_regressor;
use crate::native::descriptor::{SKNativeModelDescriptor, SKRetrainingSupport};
use crate::native::error::SKNativeImportError;

pub use super::result::SKNativeImport;

/// Convert a validated neutral state into a native fitted model.
///
/// The estimator name selects the family. A name outside the documented matrix
/// returns [`SKNativeImportError::UnsupportedFamily`]; a covered family with the
/// wrong tensor set returns [`SKNativeImportError::MissingTensor`] or
/// [`SKNativeImportError::UnexpectedTensor`].
pub fn sk_import_native_state<F: SKFloat>(
    state: &SKModelState<F>,
) -> Result<SKNativeImport<F>, SKNativeImportError> {
    // A zero-width model cannot answer any prediction; validating the declared
    // feature count here keeps every family from rebuilding one.
    if state.metadata().feature_count() == 0 {
        return Err(SKNativeImportError::ShapeMismatch {
            tensor: "feature_count".to_string(),
            expected: vec![1],
            found: vec![0],
        });
    }
    let estimator_name = state.metadata().estimator_name();
    if let Some(converted) = convert_regressor(state, estimator_name)? {
        return Ok(converted);
    }
    if let Some(converted) = convert_classifier(state, estimator_name)? {
        return Ok(converted);
    }
    if let Some(converted) = convert_transformer(state, estimator_name)? {
        return Ok(converted);
    }
    if let Some(converted) = convert_clusterer(state, estimator_name)? {
        return Ok(converted);
    }
    Err(SKNativeImportError::UnsupportedFamily {
        estimator_name: estimator_name.to_string(),
    })
}

/// Rebuild one model type through its existing `SKImportable` implementation.
pub(crate) fn import_model<F: SKFloat, M: SKImportable<F>>(
    state: &SKModelState<F>,
) -> Result<M, SKNativeImportError> {
    Ok(M::import_state(state)?)
}

/// Describe a conversion that is about to succeed.
pub(crate) fn descriptor<F: SKFloat>(
    state: &SKModelState<F>,
    retraining: SKRetrainingSupport,
) -> SKNativeModelDescriptor {
    SKNativeModelDescriptor::from_state(state, retraining)
}

/// The retraining rule for a family that is inference-only after conversion.
pub(crate) fn not_retrainable(reason: &'static str) -> SKRetrainingSupport {
    SKRetrainingSupport::Unsupported { reason }
}
