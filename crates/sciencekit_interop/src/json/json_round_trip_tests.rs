//! Companion round-trip tests for the JSON debug reader (spec `json-debug-format`).

use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};

use sciencekit_common::{
    SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKExportable, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, SKTensorDataType, SKTensorValues,
    sk_validate_import_metadata,
};

use super::{
    SKJsonDebugReader, SKJsonFormatMode, sk_export_model_json, sk_import_model_json,
    sk_read_model_state_json, sk_read_model_state_json_file, sk_write_model_state_json_file,
    sk_write_model_state_json_pretty,
};

static NEXT_ROUND_TRIP_FILE_INDEX: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, PartialEq)]
struct MockFittedScaler {
    with_mean: bool,
    means: Vec<f64>,
}

impl SKExportable<f64> for MockFittedScaler {
    fn export_state(&self) -> Result<SKModelState<f64>, SKError> {
        let metadata = SKModelMetadata::new(
            "SKStandardScaler",
            1,
            SKTensorDataType::F64,
            self.means.len(),
        )?
        .with_parameter("with_mean", SKParameterValue::from(self.with_mean))?;
        let mean_tensor = SKTensor::floating(
            "standard_scaler.mean",
            vec![self.means.len()],
            self.means.clone(),
        )?;
        SKModelState::new(metadata, vec![mean_tensor])
    }
}

impl SKImportable<f64> for MockFittedScaler {
    fn import_state(state: &SKModelState<f64>) -> Result<Self, SKError> {
        sk_validate_import_metadata(
            state,
            "SKStandardScaler",
            1,
            state.metadata().feature_count(),
        )?;
        let with_mean = state
            .metadata()
            .parameter("with_mean")
            .and_then(SKParameterValue::as_bool)
            .unwrap_or(true);
        let tensor =
            state
                .tensor("standard_scaler.mean")
                .ok_or(SKError::InvalidHyperparameter {
                    name: "standard_scaler.mean",
                    reason: "missing standard_scaler.mean tensor".to_string(),
                })?;
        let means = match tensor.values() {
            SKTensorValues::Floating(values) => values.clone(),
            _ => {
                return Err(SKError::InvalidHyperparameter {
                    name: "standard_scaler.mean",
                    reason: "expected floating values".to_string(),
                });
            }
        };
        Ok(MockFittedScaler { with_mean, means })
    }
}

/// Writing and reading back an `SKModelState` preserves every metadata field and tensor exactly.
#[test]
fn json_round_trip_reconstructs_identical_model_state() {
    let metadata = SKModelMetadata::new("SKLogisticRegression", 1, SKTensorDataType::F64, 3)
        .unwrap()
        .with_class_labels(vec!["negative".to_string(), "positive".to_string()])
        .unwrap()
        .with_parameter("fit_intercept", SKParameterValue::from(true))
        .unwrap()
        .with_parameter(
            "maximum_number_of_iterations",
            SKParameterValue::from(100_i64),
        )
        .unwrap()
        .with_parameter("regularization_strength", SKParameterValue::from(1.0_f64))
        .unwrap()
        .with_parameter("solver", SKParameterValue::from("lbfgs"))
        .unwrap();
    let weights =
        SKTensor::floating("logistic.weights", vec![1, 3], vec![0.25_f64, -1.5, 3.125]).unwrap();
    let intercept = SKTensor::floating("logistic.intercept", vec![], vec![-0.75_f64]).unwrap();
    let class_codes = SKTensor::<f64>::integer("logistic.codes", vec![2], vec![0_i64, 1]).unwrap();
    let converged = SKTensor::<f64>::boolean("logistic.converged", vec![1], vec![true]).unwrap();
    let state =
        SKModelState::new(metadata, vec![weights, intercept, class_codes, converged]).unwrap();

    let json = sk_write_model_state_json_pretty(&state).unwrap();
    let restored = sk_read_model_state_json::<f64>(&json).unwrap();

    assert_eq!(restored, state);
}

/// Single-precision states and nested hyperparameter lists round-trip bit-for-bit.
#[test]
fn json_round_trip_preserves_single_precision_state_and_nested_lists() {
    let nested = SKParameterValue::List(vec![
        SKParameterValue::from(false),
        SKParameterValue::from(-7_i64),
        SKParameterValue::from(2.0_f64),
        SKParameterValue::List(vec![SKParameterValue::from("leaf")]),
    ]);
    let metadata = SKModelMetadata::new("SKStandardScaler", 2, SKTensorDataType::F32, 3)
        .unwrap()
        .with_parameter("nested_configuration", nested)
        .unwrap();
    let means =
        SKTensor::floating("standard_scaler.mean", vec![3], vec![0.1_f32, -0.3, 1.0e-5]).unwrap();
    let state = SKModelState::new(metadata, vec![means]).unwrap();

    let json = sk_write_model_state_json_pretty(&state).unwrap();
    let restored = sk_read_model_state_json::<f32>(&json).unwrap();

    assert_eq!(restored, state);
}

/// End-to-end export and import through `SKExportable` and `SKImportable` rebuilds the model.
#[test]
fn json_round_trip_reconstructs_model_through_export_and_import_traits() {
    let original = MockFittedScaler {
        with_mean: true,
        means: vec![10.5, -3.25, 0.0, 42.125],
    };

    let json = sk_export_model_json(&original, SKJsonFormatMode::Pretty).unwrap();
    let imported: MockFittedScaler = sk_import_model_json::<f64, _>(&json).unwrap();

    assert_eq!(imported, original);
}

/// Reader streams, files and unknown forward-compatible fields within major version 1 work.
#[test]
fn json_round_trip_supports_files_streams_and_ignores_unknown_fields() {
    let reader = SKJsonDebugReader::builder()
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    assert_eq!(reader.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        reader.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let forward_compatible_json = r#"{
        "format_version": 1,
        "future_top_level_hint": { "streaming": true },
        "estimator": "SKStandardScaler",
        "dtype": "f64",
        "feature_count": 2,
        "class_labels": null,
        "hyperparameters": { "with_mean": true },
        "tensors": {
            "standard_scaler.mean": {
                "dtype": "F64",
                "shape": [2],
                "values": [1.5, 2.5],
                "future_tensor_note": "ignored"
            }
        }
    }"#;
    let from_stream = reader
        .read_state_from_reader::<f64, _>(Cursor::new(forward_compatible_json.as_bytes()))
        .unwrap();
    assert_eq!(from_stream.metadata().estimator_name(), "SKStandardScaler");

    let index = NEXT_ROUND_TRIP_FILE_INDEX.fetch_add(1, Ordering::Relaxed);
    let temp_path = std::env::temp_dir().join(format!(
        "sciencekit_json_round_trip_test_{}_{index}.json",
        std::process::id()
    ));
    sk_write_model_state_json_file(&temp_path, &from_stream, SKJsonFormatMode::Pretty).unwrap();
    let from_file = sk_read_model_state_json_file::<f64>(&temp_path).unwrap();
    let imported_model: MockFittedScaler = reader.import_from_path(&temp_path).unwrap();
    let _ = std::fs::remove_file(&temp_path);

    assert_eq!(from_file, from_stream);
    assert_eq!(imported_model.means, vec![1.5, 2.5]);
}
