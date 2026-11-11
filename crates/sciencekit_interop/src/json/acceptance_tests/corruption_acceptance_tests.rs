//! Acceptance scenarios for corrupt, truncated and adversarial JSON inputs
//! (PRD §8.7, spec `json-debug-format`).

use std::path::Path;

use sciencekit_common::{
    SKError, SKModelMetadata, SKModelState, SKParameterValue, SKTensor, SKTensorDataType,
};
use tracing_subscriber::layer::SubscriberExt;

use super::fixtures::SpanRecorder;
use crate::json::{
    sk_read_model_state_json, sk_read_model_state_json_file, sk_write_model_state_json_compact,
};

/// Truncating a valid JSON document at every UTF-8 boundary never panics and returns `Conversion`.
#[test]
fn json_acceptance_truncated_prefixes_return_conversion_without_panicking() {
    let metadata = SKModelMetadata::new("SKStandardScaler", 1, SKTensorDataType::F64, 3)
        .unwrap()
        .with_parameter("with_mean", SKParameterValue::from(true))
        .unwrap();
    let mean =
        SKTensor::floating("standard_scaler.mean", vec![3], vec![1.0_f64, 2.5, -3.75]).unwrap();
    let state = SKModelState::new(metadata, vec![mean]).unwrap();
    let valid_json = sk_write_model_state_json_compact(&state).unwrap();

    for (byte_offset, _) in valid_json.char_indices() {
        let prefix = &valid_json[..byte_offset];
        let error = sk_read_model_state_json::<f64>(prefix).unwrap_err();
        assert!(
            matches!(error, SKError::Conversion(_)),
            "expected Conversion at offset {byte_offset}, got {error:?}"
        );
    }
}

/// Corrupt metadata, invalid class tables, unsupported hyperparameters and missing files fail cleanly.
#[test]
fn json_acceptance_corrupt_metadata_and_missing_files_return_structured_errors() {
    let recorder = SpanRecorder::default();
    let subscriber = tracing_subscriber::registry().with(recorder.clone());
    let missing_file_error = tracing::subscriber::with_default(subscriber, || {
        sk_read_model_state_json_file::<f64>(Path::new("/nonexistent/sciencekit_missing.json"))
    })
    .unwrap_err();
    assert!(matches!(missing_file_error, SKError::Io(_)));
    assert!(
        recorder
            .names
            .lock()
            .unwrap()
            .iter()
            .any(|name| name == "sciencekit.interop.json.read")
    );

    let invalid_metadata_cases: &[(&str, &str)] = &[
        (
            "estimator_name",
            r#"{"format_version":1,"estimator":"","dtype":"f64","feature_count":1,"class_labels":null,"hyperparameters":{},"tensors":{}}"#,
        ),
        (
            "class_labels",
            r#"{"format_version":1,"estimator":"SKClassifier","dtype":"f64","feature_count":1,"class_labels":[],"hyperparameters":{},"tensors":{}}"#,
        ),
        (
            "class_labels",
            r#"{"format_version":1,"estimator":"SKClassifier","dtype":"f64","feature_count":1,"class_labels":["a","a"],"hyperparameters":{},"tensors":{}}"#,
        ),
        (
            "parameter_name",
            r#"{"format_version":1,"estimator":"SKClassifier","dtype":"f64","feature_count":1,"class_labels":null,"hyperparameters":{"":true},"tensors":{}}"#,
        ),
        (
            "hyperparameters",
            r#"{"format_version":1,"estimator":"SKClassifier","dtype":"f64","feature_count":1,"class_labels":null,"hyperparameters":{"bad":null},"tensors":{}}"#,
        ),
        (
            "hyperparameters",
            r#"{"format_version":1,"estimator":"SKClassifier","dtype":"f64","feature_count":1,"class_labels":null,"hyperparameters":{"bad":{"nested":1}},"tensors":{}}"#,
        ),
        (
            "hyperparameters",
            r#"{"format_version":1,"estimator":"SKClassifier","dtype":"f64","feature_count":1,"class_labels":null,"hyperparameters":{"bad":18446744073709551615},"tensors":{}}"#,
        ),
        (
            "tensor_name",
            r#"{"format_version":1,"estimator":"SKClassifier","dtype":"f64","feature_count":1,"class_labels":null,"hyperparameters":{},"tensors":{"":{"dtype":"F64","shape":[1],"values":[1.0]}}}"#,
        ),
        (
            "tensor_data_type",
            r#"{"format_version":1,"estimator":"SKClassifier","dtype":"f64","feature_count":1,"class_labels":null,"hyperparameters":{},"tensors":{"idx":{"dtype":"I64","shape":[1],"values":[18446744073709551615]}}}"#,
        ),
        (
            "tensor_data_type",
            r#"{"format_version":1,"estimator":"SKClassifier","dtype":"f64","feature_count":1,"class_labels":null,"hyperparameters":{},"tensors":{"nested":{"dtype":"F64","shape":[2],"values":[[1,2],{"a":3}]}}}"#,
        ),
    ];

    for &(expected_field, document) in invalid_metadata_cases {
        let error = sk_read_model_state_json::<f64>(document).unwrap_err();
        match error {
            SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, expected_field),
            other => panic!("expected InvalidHyperparameter({expected_field}), got {other:?}"),
        }
    }
}
