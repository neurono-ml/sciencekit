//! Companion tests for the OpenTelemetry trace bridge (spec `telemetry-export`).

use std::collections::HashMap;

use ndarray::array;
use opentelemetry::Value;
use opentelemetry::trace::Status;
use opentelemetry_sdk::trace::InMemorySpanExporter;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKOperationAttributes, SKOperationKind,
    SKUnsupervisedFit, sk_run_operation,
};
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;
use tracing_subscriber::prelude::*;

use super::{sk_build_trace_layer, sk_build_tracer_provider};
use crate::configuration::SKTelemetryConfiguration;

fn span_attributes_map(attributes: &[opentelemetry::KeyValue]) -> HashMap<String, String> {
    attributes
        .iter()
        .map(|entry| {
            let rendered = match &entry.value {
                Value::String(text) => text.to_string(),
                Value::I64(integer) => integer.to_string(),
                Value::F64(float) => float.to_string(),
                Value::Bool(flag) => flag.to_string(),
                other => format!("{other:?}"),
            };
            (entry.key.as_str().to_string(), rendered)
        })
        .collect()
}

#[test]
fn fit_operation_exports_span_with_rows_columns_backend_and_execution_mode() {
    let exporter = InMemorySpanExporter::default();
    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-trace-test")
        .build()
        .expect("configuration should build");
    let provider = sk_build_tracer_provider(&configuration, exporter.clone());
    let subscriber = tracing_subscriber::registry().with(sk_build_trace_layer(&provider));

    let matrix = array![[1.0_f64, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
    tracing::subscriber::with_default(subscriber, || {
        let scaler = SKStandardScaler::build()
            .backend(SKBackendKind::MatrixMultiply)
            .execution_mode(SKExecutionMode::InProcessSynchronous)
            .build()
            .expect("scaler should build");
        let _fitted = scaler.fit(&matrix).expect("fit should succeed");
    });

    provider.force_flush().expect("trace flush should succeed");
    let spans = exporter
        .get_finished_spans()
        .expect("in-memory spans should be readable");
    assert_eq!(spans.len(), 1);
    let span = &spans[0];
    assert_eq!(span.name.as_ref(), "sciencekit.operation");
    let attributes = span_attributes_map(&span.attributes);
    assert_eq!(attributes.get("operation").map(String::as_str), Some("fit"));
    assert_eq!(attributes.get("rows").map(String::as_str), Some("4"));
    assert_eq!(attributes.get("columns").map(String::as_str), Some("2"));
    assert_eq!(
        attributes.get("backend").map(String::as_str),
        Some("matrixmultiply")
    );
    assert_eq!(
        attributes.get("mode").map(String::as_str),
        Some("InProcessSynchronous")
    );
    assert!(attributes.contains_key("duration_ms"));
    provider
        .shutdown()
        .expect("provider shutdown should succeed");
}

#[test]
fn failed_operation_exports_span_with_error_status_and_message() {
    let exporter = InMemorySpanExporter::default();
    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-trace-error-test")
        .build()
        .expect("configuration should build");
    let provider = sk_build_tracer_provider(&configuration, exporter.clone());
    let subscriber = tracing_subscriber::registry().with(sk_build_trace_layer(&provider));

    tracing::subscriber::with_default(subscriber, || {
        let result = sk_run_operation(
            SKOperationAttributes {
                operation: SKOperationKind::Transform,
                rows_count: 8,
                columns_count: 3,
                execution_mode: SKExecutionMode::InProcessSynchronous,
                backend: SKBackendKind::Faer,
            },
            || -> Result<(), SKError> { Err(SKError::shape_mismatch_2d(8, 4, 8, 3)) },
        );
        assert!(result.is_err());
    });

    provider.force_flush().expect("trace flush should succeed");
    let spans = exporter
        .get_finished_spans()
        .expect("in-memory spans should be readable");
    assert_eq!(spans.len(), 1);
    match &spans[0].status {
        Status::Error { description } => {
            assert!(
                description.contains("shape mismatch"),
                "expected error description to mention shape mismatch, got `{description}`"
            );
        }
        other => panic!("expected Status::Error on failed operation span, got {other:?}"),
    }
    provider
        .shutdown()
        .expect("provider shutdown should succeed");
}
