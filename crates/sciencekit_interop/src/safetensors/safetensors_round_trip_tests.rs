//! Single-file round-trip and metadata header tests (spec `safetensors-format`).

use std::collections::HashMap;
use std::io::Cursor;

use sciencekit_common::{
    SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKExportable, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, SKTensorDataType, SKTensorValues,
    sk_validate_import_metadata,
};

use super::{
    SK_SAFETENSORS_FORMAT_VERSION, SKSafetensorsReader, SKSafetensorsWriter,
    sk_export_model_safetensors, sk_import_model_safetensors, sk_read_model_state_safetensors,
    sk_read_model_state_safetensors_file, sk_write_model_state_safetensors,
    sk_write_model_state_safetensors_file,
};

#[derive(Debug, PartialEq)]
struct MockExportableScaler {
    means: Vec<f64>,
    scales: Vec<f64>,
}

impl SKExportable<f64> for MockExportableScaler {
    fn export_state(&self) -> Result<SKModelState<f64>, SKError> {
        let metadata = SKModelMetadata::new(
            "SKStandardScaler",
            1,
            SKTensorDataType::F64,
            self.means.len(),
        )?
        .with_parameter("with_mean", SKParameterValue::from(true))?
        .with_parameter("with_std", SKParameterValue::from(true))?;
        let mean_tensor =
            SKTensor::floating("scaler.mean", vec![self.means.len()], self.means.clone())?;
        let scale_tensor =
            SKTensor::floating("scaler.scale", vec![self.scales.len()], self.scales.clone())?;
        SKModelState::new(metadata, vec![mean_tensor, scale_tensor])
    }
}

impl SKImportable<f64> for MockExportableScaler {
    fn import_state(state: &SKModelState<f64>) -> Result<Self, SKError> {
        sk_validate_import_metadata(
            state,
            "SKStandardScaler",
            1,
            state.metadata().feature_count(),
        )?;
        let means = match state.tensor("scaler.mean").map(SKTensor::values) {
            Some(SKTensorValues::Floating(values)) => values.clone(),
            _ => {
                return Err(SKError::InvalidHyperparameter {
                    name: "scaler.mean",
                    reason: "missing floating mean tensor".to_string(),
                });
            }
        };
        let scales = match state.tensor("scaler.scale").map(SKTensor::values) {
            Some(SKTensorValues::Floating(values)) => values.clone(),
            _ => {
                return Err(SKError::InvalidHyperparameter {
                    name: "scaler.scale",
                    reason: "missing floating scale tensor".to_string(),
                });
            }
        };
        Ok(Self { means, scales })
    }
}

/// Classifier metadata, typed hyperparameters and all tensor dtypes survive a standard Safetensors round-trip.
#[test]
fn safetensors_round_trip_preserves_classifier_metadata_and_all_tensor_dtypes() {
    let metadata = SKModelMetadata::new("SKLogisticRegression", 2, SKTensorDataType::F64, 3)
        .unwrap()
        .with_class_labels(vec![
            "setosa".to_string(),
            "versicolor".to_string(),
            "virginica".to_string(),
        ])
        .unwrap()
        .with_parameter("fit_intercept", SKParameterValue::from(true))
        .unwrap()
        .with_parameter(
            "maximum_number_of_iterations",
            SKParameterValue::from(250_i64),
        )
        .unwrap()
        .with_parameter("inverse_regularization", SKParameterValue::from(0.5_f64))
        .unwrap()
        .with_parameter("solver", SKParameterValue::from("lbfgs"))
        .unwrap()
        .with_parameter(
            "penalties",
            SKParameterValue::List(vec![
                SKParameterValue::from("l2"),
                SKParameterValue::from(1_i64),
            ]),
        )
        .unwrap();

    let weights = SKTensor::floating(
        "linear.weights",
        vec![3, 3],
        vec![0.1, -0.2, 0.3, 1.4, -1.5, 1.6, -2.7, 2.8, -2.9],
    )
    .unwrap();
    let bias_scalar = SKTensor::floating("linear.bias_scale", vec![], vec![0.75_f64]).unwrap();
    let class_indices =
        SKTensor::<f64>::integer("classes.indices", vec![3], vec![0_i64, 1, 2]).unwrap();
    let active_mask =
        SKTensor::<f64>::boolean("features.active", vec![3], vec![true, false, true]).unwrap();

    let original = SKModelState::new(
        metadata,
        vec![weights, bias_scalar, class_indices, active_mask],
    )
    .unwrap();

    let bytes = sk_write_model_state_safetensors(&original).unwrap();
    assert_eq!(&bytes[8..9], b"{");

    let standard = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(standard.len(), 4);

    let decoded = sk_read_model_state_safetensors::<f64>(&bytes).unwrap();
    assert_eq!(decoded, original);
    assert_eq!(decoded.metadata().format_version(), 2);
    assert_eq!(
        decoded.metadata().class_labels(),
        Some(
            &[
                "setosa".to_string(),
                "versicolor".to_string(),
                "virginica".to_string()
            ][..]
        )
    );
}

/// Single-precision `F32` states preserve their `F32` dtype and builder configuration.
#[test]
fn safetensors_round_trip_preserves_single_precision_f32_dtype() {
    let metadata = SKModelMetadata::new("SKRidge", 1, SKTensorDataType::F32, 4)
        .unwrap()
        .with_parameter("regularization_strength", SKParameterValue::from(1.25_f64))
        .unwrap();
    let coefficients = SKTensor::floating(
        "ridge.coefficients",
        vec![4],
        vec![0.1_f32, -0.25_f32, 3.5_f32, -4.75_f32],
    )
    .unwrap();
    let intercept = SKTensor::floating("ridge.intercept", vec![], vec![1.5_f32]).unwrap();
    let state = SKModelState::new(metadata, vec![coefficients, intercept]).unwrap();

    let writer = SKSafetensorsWriter::builder()
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    assert_eq!(writer.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        writer.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let reader = SKSafetensorsReader::builder()
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    assert_eq!(reader.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        reader.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let bytes = writer.write_state_to_bytes(&state).unwrap();
    let restored = reader.read_state_from_bytes::<f32>(&bytes).unwrap();

    assert_eq!(restored, state);
    for tensor in restored.tensors().values() {
        assert_eq!(tensor.data_type(), SKTensorDataType::F32);
    }
}

/// Unknown metadata fields in `__metadata__` are ignored while known fields survive.
#[test]
fn safetensors_round_trip_ignores_unknown_metadata_fields() {
    let raw_values: Vec<u8> = [1.0_f64, -2.0_f64]
        .into_iter()
        .flat_map(f64::to_le_bytes)
        .collect();
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F64, vec![2], &raw_values)
        .unwrap();

    let mut metadata_map = HashMap::new();
    metadata_map.insert(
        "sciencekit.format_version".to_string(),
        SK_SAFETENSORS_FORMAT_VERSION.to_string(),
    );
    metadata_map.insert(
        "sciencekit.estimator_name".to_string(),
        "SKLinearRegression".to_string(),
    );
    metadata_map.insert("sciencekit.scalar_data_type".to_string(), "F64".to_string());
    metadata_map.insert("sciencekit.feature_count".to_string(), "2".to_string());
    metadata_map.insert(
        "sciencekit.parameters".to_string(),
        r#"{"include_intercept":false}"#.to_string(),
    );
    metadata_map.insert("format".to_string(), "pt".to_string());
    metadata_map.insert(
        "custom.exported_by".to_string(),
        "external_tool".to_string(),
    );

    let encoded =
        safetensors::tensor::serialize([("linear.coefficients", view)], Some(metadata_map))
            .unwrap();

    let restored = sk_read_model_state_safetensors::<f64>(&encoded).unwrap();
    assert_eq!(restored.metadata().estimator_name(), "SKLinearRegression");
    assert_eq!(restored.metadata().feature_count(), 2);
    assert_eq!(
        restored
            .metadata()
            .parameter("include_intercept")
            .and_then(SKParameterValue::as_bool),
        Some(false)
    );
    assert_eq!(
        restored.tensor("linear.coefficients").unwrap().values(),
        &SKTensorValues::Floating(vec![1.0_f64, -2.0_f64])
    );
}

/// File paths, `io::Read`/`io::Write` streams and `SKExportable`/`SKImportable` round-trip cleanly.
#[test]
fn safetensors_round_trip_supports_files_streams_and_exportable_models() {
    let fitted = MockExportableScaler {
        means: vec![10.0, 20.0, 30.0],
        scales: vec![2.0, 4.0, 8.0],
    };
    let exported_bytes = sk_export_model_safetensors(&fitted).unwrap();
    let imported: MockExportableScaler = sk_import_model_safetensors(&exported_bytes).unwrap();
    assert_eq!(imported, fitted);

    let state = fitted.export_state().unwrap();
    let mut stream_buffer = Vec::new();
    let writer = SKSafetensorsWriter::builder().build().unwrap();
    writer
        .write_state_to_writer(&state, &mut stream_buffer)
        .unwrap();

    let reader = SKSafetensorsReader::builder().build().unwrap();
    let from_stream = reader
        .read_state_from_reader::<f64, _>(Cursor::new(&stream_buffer))
        .unwrap();
    assert_eq!(from_stream, state);

    let temporary_directory = tempfile::tempdir().unwrap();
    let file_path = temporary_directory.path().join("scaler.safetensors");
    sk_write_model_state_safetensors_file(&file_path, &state).unwrap();
    let from_file = sk_read_model_state_safetensors_file::<f64>(&file_path).unwrap();
    assert_eq!(from_file, state);
}
