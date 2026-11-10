//! Tests for model metadata (spec `model-export-contracts`).

use crate::{SKError, SKTensorDataType};

use super::{SKModelMetadata, SKParameterValue};

/// Metadata exposes the estimator identity and the data contract.
#[test]
fn interop_state_metadata_exposes_descriptive_fields() {
    let metadata = SKModelMetadata::new("SKMockEstimator", 1, SKTensorDataType::F64, 4).unwrap();

    assert_eq!(metadata.estimator_name(), "SKMockEstimator");
    assert_eq!(metadata.format_version(), 1);
    assert_eq!(metadata.scalar_data_type(), SKTensorDataType::F64);
    assert_eq!(metadata.feature_count(), 4);
    assert_eq!(metadata.class_labels(), None);
    assert!(metadata.parameters().is_empty());
}

/// Class labels are preserved in order.
#[test]
fn interop_state_metadata_preserves_class_labels() {
    let metadata = SKModelMetadata::new("SKClassifier", 1, SKTensorDataType::F32, 2)
        .unwrap()
        .with_class_labels(vec!["cat".to_string(), "dog".to_string()])
        .unwrap();

    assert_eq!(
        metadata.class_labels(),
        Some(["cat".to_string(), "dog".to_string()].as_slice())
    );
}

/// Duplicate class labels are rejected, keeping the canonical table reversible.
#[test]
fn interop_state_metadata_rejects_duplicate_class_labels() {
    let outcome = SKModelMetadata::new("SKClassifier", 1, SKTensorDataType::F32, 2)
        .unwrap()
        .with_class_labels(vec!["cat".to_string(), "cat".to_string()]);

    assert!(matches!(
        outcome,
        Err(SKError::InvalidHyperparameter { .. })
    ));
}

/// An empty estimator name cannot identify a state.
#[test]
fn interop_state_metadata_rejects_empty_estimator_name() {
    let outcome = SKModelMetadata::new("", 1, SKTensorDataType::F64, 2);

    assert!(matches!(
        outcome,
        Err(SKError::InvalidHyperparameter { .. })
    ));
}

/// Hyperparameters keep their typed values under stable names.
#[test]
fn interop_state_metadata_preserves_typed_hyperparameters() {
    let metadata = SKModelMetadata::new("SKMockEstimator", 1, SKTensorDataType::F64, 2)
        .unwrap()
        .with_parameter("number_of_iterations", SKParameterValue::from(10_i64))
        .unwrap()
        .with_parameter("tolerance", SKParameterValue::from(1.0e-6_f64))
        .unwrap()
        .with_parameter("intercept", SKParameterValue::from(true))
        .unwrap()
        .with_parameter("solver", SKParameterValue::from("mock"))
        .unwrap();

    assert_eq!(
        metadata
            .parameter("number_of_iterations")
            .and_then(SKParameterValue::as_integer),
        Some(10)
    );
    assert_eq!(
        metadata
            .parameter("tolerance")
            .and_then(SKParameterValue::as_float),
        Some(1.0e-6)
    );
    assert_eq!(
        metadata
            .parameter("intercept")
            .and_then(SKParameterValue::as_bool),
        Some(true)
    );
    assert_eq!(
        metadata
            .parameter("solver")
            .and_then(SKParameterValue::as_string),
        Some("mock")
    );
    assert_eq!(metadata.parameters().len(), 4);
}

/// A parameter name must be present.
#[test]
fn interop_state_metadata_rejects_empty_parameter_name() {
    let outcome = SKModelMetadata::new("SKMockEstimator", 1, SKTensorDataType::F64, 2)
        .unwrap()
        .with_parameter("", SKParameterValue::from(1_i64));

    assert!(matches!(
        outcome,
        Err(SKError::InvalidHyperparameter { .. })
    ));
}
