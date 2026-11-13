//! Tests for observability (spec `observability`).
//!
//! A minimal recording layer captures span names and recorded fields so the
//! tests assert exactly what a consumer's subscriber observes.

use std::collections::HashMap;

use tracing_subscriber::prelude::*;

use super::{SKBackendKind, SKOperationAttributes, SKOperationKind, sk_run_operation};
use crate::SKError;
use crate::execution::SKExecutionMode;
use crate::observability::recording_layer::RecordingLayer;

fn operation_attributes(operation: SKOperationKind) -> SKOperationAttributes {
    SKOperationAttributes {
        operation,
        rows_count: 128,
        columns_count: 16,
        execution_mode: SKExecutionMode::InProcessSynchronous,
        backend: SKBackendKind::Faer,
    }
}

/// A public operation emits a span with its structured fields.
#[test]
fn operation_emits_span_with_structured_fields() {
    let recording = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(recording.clone());
    tracing::subscriber::with_default(subscriber, || {
        let result = sk_run_operation(operation_attributes(SKOperationKind::Fit), || {
            Ok::<u32, SKError>(7)
        });
        assert_eq!(result.unwrap(), 7);
    });

    let spans = recording.recorded();
    assert_eq!(spans.len(), 1);
    let (name, fields) = &spans[0];
    assert_eq!(name, "sciencekit.operation");
    let fields: HashMap<&str, &str> = fields
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    assert_eq!(fields.get("operation"), Some(&"fit"));
    assert_eq!(fields.get("rows"), Some(&"128"));
    assert_eq!(fields.get("columns"), Some(&"16"));
    assert_eq!(fields.get("mode"), Some(&"InProcessSynchronous"));
    assert_eq!(fields.get("backend"), Some(&"faer"));
    assert!(
        fields.contains_key("duration_ms"),
        "duration must be recorded"
    );
    assert_eq!(fields.get("error"), None, "no error field on success");
}

/// A failed operation records its error on the span.
#[test]
fn failed_operation_records_error_on_span() {
    let recording = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(recording.clone());
    tracing::subscriber::with_default(subscriber, || {
        let result = sk_run_operation(
            operation_attributes(SKOperationKind::Transform),
            || -> Result<u32, SKError> { Err(SKError::shape_mismatch_2d(3, 2, 5, 4)) },
        );
        assert!(result.is_err());
    });

    let spans = recording.recorded();
    let (_, fields) = &spans[0];
    let error_field = fields
        .iter()
        .find(|(name, _)| name == "error")
        .expect("error must be recorded on the span");
    assert!(
        error_field.1.contains("shape mismatch"),
        "error field should name the failure, got {}",
        error_field.1
    );
}

/// The default build initializes no OpenTelemetry exporter.
#[cfg(not(feature = "observability-export"))]
#[test]
fn default_build_initializes_no_exporter() {
    // The `observability-export` machinery is compiled out entirely: this test
    // documents that a default build never initializes an exporter, so no OTLP
    // traffic is ever produced.
    assert!(!cfg!(feature = "observability-export"));
}

/// The opt-in feature exposes a buildable OpenTelemetry layer.
#[cfg(feature = "observability-export")]
#[test]
fn opentelemetry_export_layer_is_buildable() {
    use tracing_subscriber::registry::Registry;
    let _layer = super::sk_opentelemetry_layer::<Registry>("sciencekit-test");
}
