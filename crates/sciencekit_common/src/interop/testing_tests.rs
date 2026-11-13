//! Companion tests for the shared round-trip helper.

use crate::SKError;
use crate::interop::{
    SKExportable, SKImportable, SKModelMetadata, SKModelState, SKTensor, SKTensorDataType,
    sk_assert_export_round_trip, sk_assert_state_names,
};

/// A three-coefficient mock the helper can round-trip without a real estimator.
#[derive(Debug, Clone, PartialEq)]
struct SKTestingMockModel {
    coefficients: Vec<f64>,
}

const MOCK_NAME: &str = "SKTestingMockModel";
const MOCK_VERSION: u32 = 1;

impl SKExportable<f64> for SKTestingMockModel {
    fn export_state(&self) -> Result<SKModelState<f64>, SKError> {
        let metadata = SKModelMetadata::new(
            MOCK_NAME,
            MOCK_VERSION,
            SKTensorDataType::F64,
            self.coefficients.len(),
        )?;
        let coefficients = SKTensor::floating(
            "mock.coefficients",
            vec![self.coefficients.len()],
            self.coefficients.clone(),
        )?;
        SKModelState::new(metadata, vec![coefficients])
    }
}

impl SKImportable<f64> for SKTestingMockModel {
    fn import_state(state: &SKModelState<f64>) -> Result<Self, SKError> {
        crate::interop::sk_validate_import_metadata(
            state,
            MOCK_NAME,
            MOCK_VERSION,
            state.metadata().feature_count(),
        )?;
        let tensor = state
            .tensor("mock.coefficients")
            .ok_or(SKError::InvalidHyperparameter {
                name: "mock.coefficients",
                reason: "missing".to_string(),
            })?;
        match tensor.values() {
            crate::interop::SKTensorValues::Floating(values) => Ok(SKTestingMockModel {
                coefficients: values.clone(),
            }),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "non-floating tensor",
                suggestion: "export floating values",
            }),
        }
    }
}

#[test]
fn export_round_trip_is_deterministic_and_preserves_metadata() {
    let model = SKTestingMockModel {
        coefficients: vec![0.5, -1.5, 2.0],
    };
    let imported = sk_assert_export_round_trip::<f64, _>(&model)
        .expect("the mock model round-trips through the shared helper");
    assert_eq!(imported, model);
}

#[test]
fn empty_state_is_rejected() {
    let state = SKModelState::<f64>::new(
        SKModelMetadata::new(MOCK_NAME, MOCK_VERSION, SKTensorDataType::F64, 0).unwrap(),
        Vec::new(),
    )
    .unwrap();
    assert!(sk_assert_state_names(&state).is_err());
}

#[test]
fn underscore_only_name_is_rejected() {
    let metadata = SKModelMetadata::new(MOCK_NAME, MOCK_VERSION, SKTensorDataType::F64, 1).unwrap();
    let tensor = SKTensor::floating("coefficients", vec![1], vec![1.0]).unwrap();
    let state = SKModelState::new(metadata, vec![tensor]).unwrap();
    assert!(sk_assert_state_names(&state).is_err());
}
