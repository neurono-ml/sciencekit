//! Companion tests for the OpenTelemetry log bridge (spec `telemetry-export`).

use std::collections::HashMap;

use opentelemetry::logs::{AnyValue, Severity};
use opentelemetry_sdk::logs::InMemoryLogExporter;
use sciencekit_common::SKBuilder;
use tracing_subscriber::prelude::*;

use super::{sk_build_log_layer, sk_build_logger_provider};
use crate::configuration::SKTelemetryConfiguration;

fn render_any_value(value: &AnyValue) -> String {
    match value {
        AnyValue::String(text) => text.as_str().to_string(),
        AnyValue::Int(integer) => integer.to_string(),
        AnyValue::Double(float) => float.to_string(),
        AnyValue::Boolean(flag) => flag.to_string(),
        other => format!("{other:?}"),
    }
}

#[test]
fn structured_tracing_events_export_with_matching_severity_target_body_and_fields() {
    let exporter = InMemoryLogExporter::default();
    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-log-test")
        .build()
        .expect("configuration should build");
    let provider = sk_build_logger_provider(&configuration, exporter.clone());
    let subscriber = tracing_subscriber::registry().with(sk_build_log_layer(&provider));

    tracing::subscriber::with_default(subscriber, || {
        tracing::info!(
            target: "sciencekit.training",
            rows = 64_i64,
            backend = "faer",
            converged = true,
            "completed training epoch"
        );
        tracing::warn!(
            target: "sciencekit.device",
            fallback_reason = "no_enumerated_device",
            "falling back to host backend"
        );
    });

    provider.force_flush().expect("log flush should succeed");
    let logs = exporter
        .get_emitted_logs()
        .expect("in-memory logs should be readable");
    assert_eq!(logs.len(), 2);

    let info_record = &logs[0].record;
    assert_eq!(info_record.severity_number(), Some(Severity::Info));
    assert_eq!(
        info_record.target().map(|t| t.as_ref()),
        Some("sciencekit.training")
    );
    let info_body = info_record.body().map(render_any_value);
    assert_eq!(info_body.as_deref(), Some("completed training epoch"));
    let info_attributes: HashMap<String, String> = info_record
        .attributes_iter()
        .map(|(key, value)| (key.as_str().to_string(), render_any_value(value)))
        .collect();
    assert_eq!(info_attributes.get("rows").map(String::as_str), Some("64"));
    assert_eq!(
        info_attributes.get("backend").map(String::as_str),
        Some("faer")
    );
    assert_eq!(
        info_attributes.get("converged").map(String::as_str),
        Some("true")
    );

    let warn_record = &logs[1].record;
    assert_eq!(warn_record.severity_number(), Some(Severity::Warn));
    let warn_attributes: HashMap<String, String> = warn_record
        .attributes_iter()
        .map(|(key, value)| (key.as_str().to_string(), render_any_value(value)))
        .collect();
    assert_eq!(
        warn_attributes.get("fallback_reason").map(String::as_str),
        Some("no_enumerated_device")
    );

    provider.shutdown().expect("logger shutdown should succeed");
}

#[test]
fn structured_event_inside_operation_span_preserves_both_span_and_log_records() {
    use opentelemetry_sdk::trace::InMemorySpanExporter;
    use sciencekit_common::{
        SKBackendKind, SKError, SKExecutionMode, SKOperationAttributes, SKOperationKind,
        sk_run_operation,
    };

    use super::{sk_build_trace_layer, sk_build_tracer_provider};

    let span_exporter = InMemorySpanExporter::default();
    let log_exporter = InMemoryLogExporter::default();
    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-correlated-log-test")
        .build()
        .expect("configuration should build");
    let tracer_provider = sk_build_tracer_provider(&configuration, span_exporter.clone());
    let logger_provider = sk_build_logger_provider(&configuration, log_exporter.clone());
    let subscriber = tracing_subscriber::registry()
        .with(sk_build_trace_layer(&tracer_provider))
        .with(sk_build_log_layer(&logger_provider));

    tracing::subscriber::with_default(subscriber, || {
        let result = sk_run_operation(
            SKOperationAttributes {
                operation: SKOperationKind::Predict,
                rows_count: 32,
                columns_count: 8,
                execution_mode: SKExecutionMode::InProcessSynchronous,
                backend: SKBackendKind::Faer,
            },
            || -> Result<usize, SKError> {
                tracing::error!(batch_index = 3_i64, "numerical guard triggered");
                Ok(32)
            },
        );
        assert_eq!(result.expect("operation should complete"), 32);
    });

    tracer_provider
        .force_flush()
        .expect("trace flush should succeed");
    logger_provider
        .force_flush()
        .expect("log flush should succeed");

    let spans = span_exporter.get_finished_spans().expect("spans readable");
    let logs = log_exporter.get_emitted_logs().expect("logs readable");
    assert_eq!(spans.len(), 1);
    assert_eq!(logs.len(), 1);
    assert_eq!(logs[0].record.severity_number(), Some(Severity::Error));
    assert!(
        logs[0].record.trace_context().is_some(),
        "log emitted inside span should carry trace context"
    );

    tracer_provider.shutdown().expect("tracer shutdown");
    logger_provider.shutdown().expect("logger shutdown");
}
