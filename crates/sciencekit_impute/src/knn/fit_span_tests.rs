//! Companion tests for the backend and execution-mode transfer of the
//! nearest-neighbor imputer (spec `nearest-neighbor-imputation`).

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilder, SKExecutionMode, SKUnsupervisedFit};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

use crate::knn::core_implementation::SKKNNImputer;

/// Captured spans as `(name, fields)` pairs.
type RecordedSpans = Arc<Mutex<Vec<(String, Vec<(String, String)>)>>>;

struct FieldCollector {
    fields: Vec<(&'static str, String)>,
}

impl Visit for FieldCollector {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.fields.push((field.name(), format!("{value:?}")));
    }
}

#[derive(Clone, Default)]
struct RecordingLayer {
    spans: RecordedSpans,
}

impl RecordingLayer {
    fn recorded(&self) -> Vec<(String, Vec<(String, String)>)> {
        self.spans.lock().unwrap().clone()
    }
}

impl<S> Layer<S> for RecordingLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn register_callsite(&self, _metadata: &'static tracing::Metadata<'_>) -> Interest {
        Interest::always()
    }

    fn on_new_span(&self, attributes: &Attributes<'_>, _id: &Id, _context: Context<'_, S>) {
        let mut collector = FieldCollector { fields: Vec::new() };
        attributes.values().record(&mut collector);
        let fields = collector
            .fields
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect();
        self.spans
            .lock()
            .unwrap()
            .push((attributes.metadata().name().to_string(), fields));
    }

    fn on_record(&self, _span: &Id, _values: &Record<'_>, _context: Context<'_, S>) {}
}

fn fields_of(fields: &[(String, String)]) -> HashMap<&str, &str> {
    fields
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect()
}

#[test]
fn knn_fit_transfers_backend_and_execution_intent_to_the_model() {
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(2)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap()
        .fit(&array![[1.0_f64, 2.0], [3.0, 4.0]])
        .unwrap();
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        model.execution_intent(),
        SKExecutionMode::InProcessSynchronous
    );
}

#[test]
fn knn_fit_records_the_configured_backend_in_the_fit_span() {
    let recording = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(recording.clone());
    tracing::subscriber::with_default(subscriber, || {
        SKKNNImputer::<f64>::build()
            .nearest_neighbors_count(2)
            .backend(SKBackendKind::NdArrayLinalg)
            .execution_mode(SKExecutionMode::InProcessSynchronous)
            .build()
            .unwrap()
            .fit(&array![[1.0_f64, 2.0], [3.0, 4.0]])
            .unwrap();
    });

    let spans = recording.recorded();
    let (name, fields) = spans
        .iter()
        .find(|(name, _)| name == "sciencekit.operation")
        .expect("fit must emit the shared operation span");
    assert_eq!(name, "sciencekit.operation");
    let fields = fields_of(fields);
    assert_eq!(fields.get("operation"), Some(&"fit"));
    assert_eq!(fields.get("backend"), Some(&"ndarray-linalg"));
    assert_eq!(fields.get("mode"), Some(&"InProcessSynchronous"));
}
