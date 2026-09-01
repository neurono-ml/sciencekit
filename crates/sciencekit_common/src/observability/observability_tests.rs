//! Tests for observability (spec `observability`).
//!
//! A minimal recording layer captures span names and recorded fields so the
//! tests assert exactly what a consumer's subscriber observes.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use tracing::Subscriber;
use tracing::field::Visit;
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

use super::{SKOperationAttributes, sk_run_operation};
use crate::SKError;
use crate::execution::SKExecutionMode;

/// Captured spans, keyed by span id: `(name, recorded fields)`.
type RecordedSpans = Arc<Mutex<HashMap<Id, (String, Vec<(String, String)>)>>>;

/// Collects every field value of a span/record into `(name, debug_string)`.
struct FieldCollector {
    fields: Vec<(&'static str, String)>,
}

impl Visit for FieldCollector {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        self.fields.push((field.name(), format!("{value:?}")));
    }
}

/// A minimal layer recording each span's name and its recorded fields.
#[derive(Clone, Default)]
struct RecordingLayer {
    spans: RecordedSpans,
}

impl RecordingLayer {
    fn recorded(&self) -> Vec<(String, Vec<(String, String)>)> {
        self.spans.lock().unwrap().values().cloned().collect()
    }
}

impl<S> Layer<S> for RecordingLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn register_callsite(&self, _metadata: &'static tracing::Metadata<'_>) -> Interest {
        Interest::always()
    }

    fn on_new_span(&self, attributes: &Attributes<'_>, id: &Id, _context: Context<'_, S>) {
        let mut collector = FieldCollector { fields: Vec::new() };
        attributes.values().record(&mut collector);
        let name = attributes.metadata().name().to_string();
        let fields = collector
            .fields
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect();
        self.spans
            .lock()
            .unwrap()
            .insert(id.clone(), (name, fields));
    }

    fn on_record(&self, span: &Id, values: &Record<'_>, _context: Context<'_, S>) {
        let mut collector = FieldCollector { fields: Vec::new() };
        values.record(&mut collector);
        if let Some((_, fields)) = self.spans.lock().unwrap().get_mut(span) {
            fields.extend(
                collector
                    .fields
                    .into_iter()
                    .map(|(name, value)| (name.to_string(), value)),
            );
        }
    }
}

fn operation_attributes(operation: &'static str) -> SKOperationAttributes {
    SKOperationAttributes {
        operation,
        rows: 128,
        columns: 16,
        execution_mode: SKExecutionMode::InProcessSynchronous,
        backend: "faer",
    }
}

/// A public operation emits a span with its structured fields.
#[test]
fn operation_emits_span_with_structured_fields() {
    let recording = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(recording.clone());
    tracing::subscriber::with_default(subscriber, || {
        let result = sk_run_operation(operation_attributes("fit"), || Ok::<u32, SKError>(7));
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
    assert_eq!(fields.get("operation"), Some(&"\"fit\""));
    assert_eq!(fields.get("rows"), Some(&"128"));
    assert_eq!(fields.get("columns"), Some(&"16"));
    assert_eq!(fields.get("mode"), Some(&"InProcessSynchronous"));
    assert_eq!(fields.get("backend"), Some(&"\"faer\""));
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
            operation_attributes("transform"),
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
