//! Tests for the model state container (spec `model-export-contracts`).

use crate::{SKError, SKTensor, SKTensorDataType};

use super::{SKModelMetadata, SKModelState, SKParameterValue};

fn metadata() -> SKModelMetadata {
    SKModelMetadata::new("SKMockEstimator", 1, SKTensorDataType::F64, 2).unwrap()
}

/// A state collects its tensors under their stable names.
#[test]
fn interop_state_collects_named_tensors() {
    let coefficients =
        SKTensor::floating("linear.coefficients", vec![2], vec![1.0_f64, 2.0]).unwrap();
    let intercept = SKTensor::floating("linear.intercept", vec![], vec![0.5_f64]).unwrap();

    let state = SKModelState::new(metadata(), vec![coefficients, intercept]).unwrap();

    assert_eq!(state.number_of_tensors(), 2);
    assert_eq!(
        state.tensor("linear.coefficients").map(SKTensor::data_type),
        Some(SKTensorDataType::F64)
    );
    assert!(state.tensors().contains_key("linear.intercept"));
    assert_eq!(state.metadata().feature_count(), 2);
}

/// Two tensors cannot share a name.
#[test]
fn interop_state_rejects_duplicate_tensor_names() {
    let first = SKTensor::floating("linear.coefficients", vec![1], vec![1.0_f64]).unwrap();
    let second = SKTensor::floating("linear.coefficients", vec![1], vec![2.0_f64]).unwrap();

    let outcome = SKModelState::new(metadata(), vec![first, second]);

    assert!(matches!(
        outcome,
        Err(SKError::InvalidHyperparameter { .. })
    ));
}

/// A floating tensor must use the metadata's scalar data_type.
#[test]
fn interop_state_rejects_tensor_data_type_disagreeing_with_metadata_scalar() {
    let tensor = SKTensor::floating("linear.coefficients", vec![1], vec![1.0_f64]).unwrap();
    let single_precision =
        SKModelMetadata::new("SKMockEstimator", 1, SKTensorDataType::F32, 1).unwrap();

    let outcome = SKModelState::new(single_precision, vec![tensor]);

    assert!(matches!(
        outcome,
        Err(SKError::InvalidHyperparameter { .. })
    ));
}

/// The state exposes the metadata's typed hyperparameters.
#[test]
fn interop_state_exposes_metadata_parameters() {
    let metadata = metadata()
        .with_parameter("solver", SKParameterValue::from("mock"))
        .unwrap();
    let state = SKModelState::<f64>::new(metadata, Vec::new()).unwrap();

    assert_eq!(
        state
            .parameters()
            .get("solver")
            .and_then(SKParameterValue::as_string),
        Some("mock")
    );
}

/// An unknown tensor name resolves to nothing instead of panicking.
#[test]
fn interop_state_unknown_tensor_is_absent() {
    let state = SKModelState::<f64>::new(metadata(), Vec::new()).unwrap();

    assert!(state.tensor("linear.coefficients").is_none());
    assert_eq!(state.number_of_tensors(), 0);
}

/// Tensor order depends on the stable name, not on insertion order.
#[test]
fn interop_state_tensor_order_is_deterministic() {
    let metadata =
        crate::interop::SKModelMetadata::new("mock", 1, crate::interop::SKTensorDataType::F64, 2)
            .unwrap();
    let first = crate::interop::SKTensor::floating("zeta", vec![2], vec![1.0, 2.0]).unwrap();
    let second = crate::interop::SKTensor::floating("alpha", vec![2], vec![3.0, 4.0]).unwrap();
    let forward =
        crate::interop::SKModelState::new(metadata.clone(), vec![first.clone(), second.clone()])
            .unwrap();
    let reverse = crate::interop::SKModelState::new(metadata, vec![second, first]).unwrap();
    assert_eq!(
        forward.tensors().keys().cloned().collect::<Vec<_>>(),
        reverse.tensors().keys().cloned().collect::<Vec<_>>()
    );
    assert_eq!(
        forward.tensors().keys().cloned().collect::<Vec<_>>(),
        vec!["alpha".to_string(), "zeta".to_string()]
    );
}
