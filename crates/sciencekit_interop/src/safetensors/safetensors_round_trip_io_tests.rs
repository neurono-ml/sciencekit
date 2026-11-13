//! File, stream and `SKExportable` / `SKImportable` round-trip tests
//! (spec `safetensors-format`).

use std::io::Cursor;

use sciencekit_common::{
    SKError, SKExportable, SKImportable, SKModelMetadata, SKModelState, SKParameterValue, SKTensor,
    SKTensorDataType, SKTensorValues, sk_validate_import_metadata,
};

use super::{
    SKSafetensorsReader, SKSafetensorsWriter, sk_export_model_safetensors,
    sk_export_model_safetensors_file, sk_import_model_safetensors,
    sk_import_model_safetensors_file, sk_read_model_state_safetensors_file,
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

    let model_path = temporary_directory.path().join("exported.safetensors");
    sk_export_model_safetensors_file(&model_path, &fitted).unwrap();
    let imported_from_file: MockExportableScaler =
        sk_import_model_safetensors_file(&model_path).unwrap();
    assert_eq!(imported_from_file, fitted);
}
