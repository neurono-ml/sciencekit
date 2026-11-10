//! Tests for import-time state validation (spec `model-export-contracts`).

use crate::{SKError, SKModelMetadata, SKModelState, SKTensorDataType};

use super::mock_model::{
    MOCK_ESTIMATOR_NAME, MOCK_FORMAT_VERSION, SKMockLinearModel, SKUnsupportedMockModel,
    mock_linear_model,
};
use super::{SKExportable, SKImportable};

/// Rebuild the exported state with different metadata, keeping its tensors.
fn state_with_metadata(
    estimator_name: &str,
    format_version: u32,
    feature_count: usize,
) -> SKModelState<f64> {
    let exported = mock_linear_model().export_state().unwrap();
    let tensors: Vec<_> = exported.tensors().values().cloned().collect();
    let metadata = SKModelMetadata::new(
        estimator_name,
        format_version,
        SKTensorDataType::F64,
        feature_count,
    )
    .unwrap();
    SKModelState::new(metadata, tensors).unwrap()
}

/// A state naming another estimator is refused as an unsupported representation.
#[test]
fn interop_validation_rejects_wrong_estimator() {
    let forged = state_with_metadata("SKOtherEstimator", MOCK_FORMAT_VERSION, 3);

    let outcome = SKMockLinearModel::<f64>::import_state(&forged);

    assert!(matches!(
        outcome,
        Err(SKError::UnsupportedRepresentation { .. })
    ));
}

/// A state written by another format version is refused.
#[test]
fn interop_validation_rejects_wrong_format_version() {
    let forged = state_with_metadata(MOCK_ESTIMATOR_NAME, MOCK_FORMAT_VERSION + 1, 3);

    let outcome = SKMockLinearModel::<f64>::import_state(&forged);

    assert!(matches!(
        outcome,
        Err(SKError::UnsupportedRepresentation { .. })
    ));
}

/// A state whose feature count disagrees with its tensors is a shape mismatch.
#[test]
fn interop_validation_rejects_wrong_feature_count() {
    let forged = state_with_metadata(MOCK_ESTIMATOR_NAME, MOCK_FORMAT_VERSION, 2);

    let outcome = SKMockLinearModel::<f64>::import_state(&forged);

    assert!(matches!(outcome, Err(SKError::ShapeMismatch { .. })));
}

/// A model that cannot be reconstructed reports an unsupported operation.
#[test]
fn interop_validation_rejects_unsupported_model() {
    let state = mock_linear_model().export_state().unwrap();

    let outcome = SKUnsupportedMockModel::import_state(&state);

    assert!(matches!(outcome, Err(SKError::UnsupportedOperation { .. })));
}

/// A state missing a required tensor is refused before any partial model exists.
#[test]
fn interop_validation_rejects_missing_tensor() {
    let metadata = SKModelMetadata::new(
        MOCK_ESTIMATOR_NAME,
        MOCK_FORMAT_VERSION,
        SKTensorDataType::F64,
        3,
    )
    .unwrap();
    let state = SKModelState::<f64>::new(metadata, Vec::new()).unwrap();

    let outcome = SKMockLinearModel::<f64>::import_state(&state);

    assert!(matches!(
        outcome,
        Err(SKError::InvalidHyperparameter { .. })
    ));
}
