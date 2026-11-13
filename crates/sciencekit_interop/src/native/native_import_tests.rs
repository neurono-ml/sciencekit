//! Companion tests: container detection, typed rejections and the coverage table
//! (spec `native-model-import`, requirement "Validation and concurrency").

use sciencekit_common::{SKExportable, SKModelMetadata, SKModelState, SKTensor, SKTensorDataType};

use super::conversion::sk_import_native_state;
use super::error::SKNativeImportError;
use super::format_detection::{SKNativeModelFormat, sk_detect_native_model_format};
use super::{sk_import_native_model_from_bytes, sk_native_import_coverage};
use crate::safetensors::sk_export_model_safetensors;

/// A Safetensors model and its JSON twin are both detected structurally.
#[test]
fn native_import_detects_both_containers() {
    let model = super::test_models::linear_regression();
    let safetensors = sk_export_model_safetensors(&model).unwrap();
    assert_eq!(
        sk_detect_native_model_format(&safetensors).unwrap(),
        SKNativeModelFormat::Safetensors
    );
    let state = model.export_state().unwrap();
    let json = crate::json::sk_write_model_state_json_pretty(&state).unwrap();
    assert_eq!(
        sk_detect_native_model_format(json.as_bytes()).unwrap(),
        SKNativeModelFormat::Json
    );
}

/// Foreign bytes are rejected as a wrong magic, never as a panic.
#[test]
fn native_import_rejects_wrong_magic() {
    let error = sk_detect_native_model_format(b"not a model at all").unwrap_err();
    assert!(matches!(
        error,
        SKNativeImportError::TruncatedFile { .. } | SKNativeImportError::WrongMagic { .. }
    ));
    let mut bytes = 5_u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"hello");
    let error = sk_detect_native_model_format(&bytes).unwrap_err();
    assert!(matches!(error, SKNativeImportError::WrongMagic { .. }));
    let error = sk_detect_native_model_format(&[0_u8; 4]).unwrap_err();
    assert!(matches!(error, SKNativeImportError::WrongMagic { .. }));
}

/// A file that declares more bytes than it carries is a typed truncation.
#[test]
fn native_import_rejects_a_truncated_container() {
    let mut bytes = 4096_u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{\"__metadata__\":{}}");
    let error = sk_detect_native_model_format(&bytes).unwrap_err();
    assert!(matches!(error, SKNativeImportError::TruncatedFile { .. }));
}

/// A header length near `u64::MAX` is a typed truncation, never an overflow.
#[test]
fn native_import_rejects_an_oversized_header_length() {
    for declared in [u64::MAX, u64::MAX - 4, u64::MAX / 2] {
        let mut bytes = declared.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        let error = sk_detect_native_model_format(&bytes).unwrap_err();
        assert!(matches!(
            error,
            SKNativeImportError::TruncatedFile {
                format: "safetensors",
                ..
            }
        ));
    }
}

/// A Safetensors header naming another major version is rejected up front.
#[test]
fn native_import_rejects_an_unknown_safetensors_version() {
    let header = br#"{"__metadata__":{"sciencekit.format_version":"2"}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    let error = sk_detect_native_model_format(&bytes).unwrap_err();
    assert!(matches!(
        error,
        SKNativeImportError::UnknownFormatVersion { found: 2, .. }
    ));
}

/// A JSON document naming another container version is rejected up front.
#[test]
fn native_import_rejects_an_unknown_json_version() {
    let error = sk_detect_native_model_format(br#"{"format_version":99}"#).unwrap_err();
    assert!(matches!(
        error,
        SKNativeImportError::UnknownFormatVersion { found: 99, .. }
    ));
}

/// A covered container naming an uncovered estimator is an unsupported family.
#[test]
fn native_import_rejects_an_uncovered_family() {
    let metadata = SKModelMetadata::new("SKNotCovered", 1, SKTensorDataType::F64, 2).unwrap();
    let state = SKModelState::<f64>::new(metadata, Vec::new()).unwrap();
    let error = sk_import_native_state(&state).unwrap_err();
    assert!(matches!(
        error,
        SKNativeImportError::UnsupportedFamily { estimator_name } if estimator_name == "SKNotCovered"
    ));
}

/// The documented matrix and the dispatch table agree on every covered name.
#[test]
fn native_import_coverage_matches_dispatch() {
    for row in sk_native_import_coverage() {
        assert!(
            super::sk_native_import_supports(row.estimator_name),
            "{} must be covered",
            row.estimator_name
        );
    }
    let metadata = SKModelMetadata::new("SKNotCovered", 1, SKTensorDataType::F64, 2).unwrap();
    let state = SKModelState::<f64>::new(metadata, Vec::new()).unwrap();
    assert!(sk_import_native_state(&state).is_err());
}

/// A model file imported through the loader yields the same predictions.
#[test]
fn native_import_loader_round_trips_a_linear_model() {
    let model = super::test_models::linear_regression();
    let bytes = sk_export_model_safetensors(&model).unwrap();
    let imported = sk_import_native_model_from_bytes::<f64>(&bytes).unwrap();
    let probe = super::test_models::features();
    let original = sciencekit_common::SKRegressorPredictor::predict(&model, probe.view()).unwrap();
    let restored = imported.predict_regression(probe.view()).unwrap();
    assert_eq!(original, restored);
    assert_eq!(imported.descriptor().estimator_name(), "SKLinearRegression");
    assert!(!imported.supports_retraining());
}

/// An export that lost a tensor is a typed missing-tensor rejection.
#[test]
fn native_import_rejects_a_missing_tensor() {
    let model = super::test_models::linear_regression();
    let state = model.export_state().unwrap();
    let metadata = SKModelMetadata::new(
        state.metadata().estimator_name(),
        state.metadata().format_version(),
        SKTensorDataType::F64,
        state.metadata().feature_count(),
    )
    .unwrap();
    let partial = SKModelState::<f64>::new(metadata, Vec::new()).unwrap();
    let error = sk_import_native_state(&partial).unwrap_err();
    assert!(matches!(
        error,
        SKNativeImportError::MissingTensor { ref name } if name == "linear.coefficients"
    ));
}

/// An extra tensor means a different producer and is rejected, not ignored.
#[test]
fn native_import_rejects_an_unexpected_tensor() {
    let model = super::test_models::linear_regression();
    let state = model.export_state().unwrap();
    let mut tensors: Vec<SKTensor<f64>> = state.tensors().values().cloned().collect();
    tensors.push(SKTensor::floating("linear.mystery", vec![1], vec![0.0]).unwrap());
    let extra = SKModelState::new(state.metadata().clone(), tensors).unwrap();
    let error = sk_import_native_state(&extra).unwrap_err();
    assert!(matches!(
        error,
        SKNativeImportError::UnexpectedTensor { ref name } if name == "linear.mystery"
    ));
}

/// A coefficients vector whose length disagrees with the metadata is a shape mismatch.
#[test]
fn native_import_rejects_a_shape_mismatch() {
    let model = super::test_models::linear_regression();
    let state = model.export_state().unwrap();
    let mut tensors: Vec<SKTensor<f64>> = state
        .tensors()
        .values()
        .filter(|tensor| tensor.name() != "linear.coefficients")
        .cloned()
        .collect();
    tensors.push(SKTensor::floating("linear.coefficients", vec![1], vec![1.0]).unwrap());
    let broken = SKModelState::new(state.metadata().clone(), tensors).unwrap();
    assert!(sk_import_native_state(&broken).is_err());
}
