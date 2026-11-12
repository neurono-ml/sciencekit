//! Tests for the interop conversion span (spec `dataset-interop-core`).

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use sciencekit_common::SKError;
use tracing::Subscriber;
use tracing::field::Visit;
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

use super::{SKInteropConversionAttributes, SKInteropSourceKind, sk_run_interop_conversion};
use crate::missing::SKMissingValuePolicy;

type RecordedSpans = Arc<Mutex<HashMap<Id, (String, Vec<(String, String)>)>>>;

struct FieldCollector {
    fields: Vec<(&'static str, String)>,
}

impl Visit for FieldCollector {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        self.fields.push((field.name(), format!("{value:?}")));
    }
}

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

fn attributes(policy: SKMissingValuePolicy) -> SKInteropConversionAttributes {
    SKInteropConversionAttributes {
        source_kind: SKInteropSourceKind::Arrow,
        rows_count: 64,
        columns_count: 7,
        missing_value_policy: policy,
    }
}

fn fields_of(spans: &[(String, Vec<(String, String)>)]) -> HashMap<&str, &str> {
    spans[0]
        .1
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect()
}

/// A successful conversion records the engine, shape, policy and duration.
#[test]
fn conversion_span_records_structured_attributes() {
    let recording = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(recording.clone());
    tracing::subscriber::with_default(subscriber, || {
        let result = sk_run_interop_conversion(attributes(SKMissingValuePolicy::Reject), || {
            Ok::<u32, SKError>(11)
        });
        assert_eq!(result.unwrap(), 11);
    });

    let spans = recording.recorded();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].0, "sciencekit.interop.conversion");
    let fields = fields_of(&spans);
    assert_eq!(fields.get("source_kind"), Some(&"arrow"));
    assert_eq!(fields.get("rows"), Some(&"64"));
    assert_eq!(fields.get("columns"), Some(&"7"));
    assert_eq!(fields.get("policy"), Some(&"reject"));
    assert!(fields.contains_key("duration_ms"));
    assert_eq!(fields.get("error"), None);
}

/// A failed conversion records the error on the span.
#[test]
fn failed_conversion_records_the_error() {
    let recording = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(recording.clone());
    tracing::subscriber::with_default(subscriber, || {
        let result = sk_run_interop_conversion(
            attributes(SKMissingValuePolicy::NumericNaN),
            || -> Result<u32, SKError> {
                Err(SKError::NullValueInColumn {
                    column: "price".to_string(),
                    policy: "numeric_nan",
                })
            },
        );
        assert!(result.is_err());
    });

    let spans = recording.recorded();
    let fields = fields_of(&spans);
    assert_eq!(fields.get("policy"), Some(&"numeric_nan"));
    let error = fields.get("error").expect("the error must be recorded");
    assert!(error.contains("price"));
}

/// Source kinds have stable trace names.
#[test]
fn source_kinds_have_stable_names() {
    assert_eq!(SKInteropSourceKind::InMemory.to_string(), "in_memory");
    assert_eq!(SKInteropSourceKind::Arrow.to_string(), "arrow");
    assert_eq!(SKInteropSourceKind::Polars.to_string(), "polars");
    assert_eq!(SKInteropSourceKind::DataFusion.to_string(), "datafusion");
    assert_eq!(SKInteropSourceKind::MemoryMap.to_string(), "memory_map");
}
