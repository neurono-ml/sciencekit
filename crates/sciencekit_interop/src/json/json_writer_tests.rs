//! Companion tests for the JSON debug writer (spec `json-debug-format`).

use std::sync::atomic::{AtomicUsize, Ordering};

use sciencekit_common::{
    SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKExportable, SKModelMetadata,
    SKModelState, SKParameterValue, SKTensor, SKTensorDataType,
};

use super::{
    SKJsonDebugWriter, SKJsonFormatMode, sk_export_model_json, sk_write_model_state_json_compact,
    sk_write_model_state_json_file, sk_write_model_state_json_pretty,
};

static NEXT_TEMP_FILE_INDEX: AtomicUsize = AtomicUsize::new(0);

fn sample_scaler_state_forward_order() -> SKModelState<f64> {
    let metadata = SKModelMetadata::new("SKStandardScaler", 1, SKTensorDataType::F64, 2)
        .unwrap()
        .with_parameter("alpha_flag", SKParameterValue::from(true))
        .unwrap()
        .with_parameter("zeta_flag", SKParameterValue::from(false))
        .unwrap();
    let mean = SKTensor::floating("standard_scaler.mean", vec![2], vec![1.0_f64, 3.0]).unwrap();
    let scale = SKTensor::floating("standard_scaler.scale", vec![2], vec![2.0_f64, 4.0]).unwrap();
    SKModelState::new(metadata, vec![mean, scale]).unwrap()
}

fn sample_scaler_state_reverse_order() -> SKModelState<f64> {
    let metadata = SKModelMetadata::new("SKStandardScaler", 1, SKTensorDataType::F64, 2)
        .unwrap()
        .with_parameter("zeta_flag", SKParameterValue::from(false))
        .unwrap()
        .with_parameter("alpha_flag", SKParameterValue::from(true))
        .unwrap();
    let scale = SKTensor::floating("standard_scaler.scale", vec![2], vec![2.0_f64, 4.0]).unwrap();
    let mean = SKTensor::floating("standard_scaler.mean", vec![2], vec![1.0_f64, 3.0]).unwrap();
    SKModelState::new(metadata, vec![scale, mean]).unwrap()
}

struct MockExportableModel;

impl SKExportable<f64> for MockExportableModel {
    fn export_state(&self) -> Result<SKModelState<f64>, SKError> {
        Ok(sample_scaler_state_forward_order())
    }
}

/// Default writer builds in pretty mode and emits keys in deterministic order.
#[test]
fn json_writer_defaults_to_pretty_format_and_deterministic_key_ordering() {
    let writer = SKJsonDebugWriter::builder().build().unwrap();
    assert_eq!(writer.format_mode(), SKJsonFormatMode::Pretty);
    assert_eq!(writer.backend(), SKBackendKind::Faer);
    assert_eq!(writer.execution_mode(), SKExecutionMode::Automatic);

    let state = sample_scaler_state_reverse_order();
    let pretty = writer.write_state_to_string(&state).unwrap();

    assert!(pretty.contains('\n'));
    let alpha_position = pretty.find("\"alpha_flag\"").unwrap();
    let zeta_position = pretty.find("\"zeta_flag\"").unwrap();
    assert!(alpha_position < zeta_position);

    let mean_position = pretty.find("\"standard_scaler.mean\"").unwrap();
    let scale_position = pretty.find("\"standard_scaler.scale\"").unwrap();
    assert!(mean_position < scale_position);
}

/// Compact mode omits newlines and indentation while keeping identical JSON structure.
#[test]
fn json_writer_compact_mode_emits_single_line_json() {
    let state = sample_scaler_state_forward_order();
    let pretty = sk_write_model_state_json_pretty(&state).unwrap();
    let compact = sk_write_model_state_json_compact(&state).unwrap();

    assert!(!compact.contains('\n'));
    assert!(compact.len() < pretty.len());

    let pretty_value: serde_json::Value = serde_json::from_str(&pretty).unwrap();
    let compact_value: serde_json::Value = serde_json::from_str(&compact).unwrap();
    assert_eq!(pretty_value, compact_value);
}

/// Insertion order of hyperparameters and tensors never changes serialized bytes.
#[test]
fn json_writer_produces_byte_identical_output_across_insertion_orders() {
    let forward = sample_scaler_state_forward_order();
    let reverse = sample_scaler_state_reverse_order();

    assert_eq!(
        sk_write_model_state_json_pretty(&forward).unwrap(),
        sk_write_model_state_json_pretty(&reverse).unwrap()
    );
    assert_eq!(
        sk_write_model_state_json_compact(&forward).unwrap(),
        sk_write_model_state_json_compact(&reverse).unwrap()
    );
}

/// Stream, file and `SKExportable` entry points produce identical JSON bytes.
#[test]
fn json_writer_writes_to_io_sink_file_and_exportable_model() {
    let state = sample_scaler_state_forward_order();
    let writer = SKJsonDebugWriter::builder()
        .format_mode(SKJsonFormatMode::Compact)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    assert_eq!(writer.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        writer.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let mut buffer = Vec::new();
    writer.write_state_to_writer(&state, &mut buffer).unwrap();
    let from_buffer = String::from_utf8(buffer).unwrap();

    let exported = sk_export_model_json(&MockExportableModel, SKJsonFormatMode::Compact).unwrap();
    assert_eq!(from_buffer, exported);

    let index = NEXT_TEMP_FILE_INDEX.fetch_add(1, Ordering::Relaxed);
    let temp_path = std::env::temp_dir().join(format!(
        "sciencekit_json_writer_test_{}_{index}.json",
        std::process::id()
    ));
    sk_write_model_state_json_file(&temp_path, &state, SKJsonFormatMode::Compact).unwrap();
    let from_file = std::fs::read_to_string(&temp_path).unwrap();
    let _ = std::fs::remove_file(&temp_path);

    assert_eq!(from_file, exported);
}
