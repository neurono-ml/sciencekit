//! The imported model records its configured backend in every prediction span
//! (spec `backend-selection`, spec `observability`).
//!
//! A process-wide subscriber is installed once: `cargo test` runs the suite on
//! parallel threads, and a thread-local subscriber would race with predictor
//! spans emitted by the other round-trip tests.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKRegressorPredictor};
use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

use super::SKOnnxPredictor;
use super::test_support::affine_regressor_model;

#[derive(Default)]
struct DebugFieldVisitor(HashMap<String, String>);

impl Visit for DebugFieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().to_owned(), format!("{value:?}"));
    }
}

#[derive(Clone, Default)]
struct SpanFieldRecorder {
    spans: Arc<Mutex<Vec<HashMap<String, String>>>>,
}

impl<S: Subscriber> Layer<S> for SpanFieldRecorder {
    fn on_new_span(&self, attributes: &Attributes<'_>, _id: &Id, _context: Context<'_, S>) {
        let mut visitor = DebugFieldVisitor::default();
        attributes.values().record(&mut visitor);
        self.spans.lock().expect("span lock").push(visitor.0);
    }
}

fn recorder() -> &'static SpanFieldRecorder {
    static RECORDER: OnceLock<SpanFieldRecorder> = OnceLock::new();
    RECORDER.get_or_init(|| {
        let recorder = SpanFieldRecorder::default();
        let subscriber = tracing_subscriber::registry().with(recorder.clone());
        tracing::subscriber::set_global_default(subscriber)
            .expect("the test process has no global subscriber");
        tracing::callsite::rebuild_interest_cache();
        recorder
    })
}

#[test]
fn the_configured_backend_reaches_the_prediction_span() {
    let recorder = recorder();
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(affine_regressor_model())
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("the fixture imports");
    let _ = predictor
        .predict(array![[1.0, 2.0]].view())
        .expect("the fixture predicts");

    let spans = recorder.spans.lock().expect("span lock");
    let fields = spans
        .iter()
        .find(|fields| {
            fields
                .get("backend")
                .is_some_and(|value| value.contains("matrixmultiply"))
        })
        .unwrap_or_else(|| panic!("the override span must be recorded: {spans:?}"));
    assert!(
        fields
            .get("operation")
            .is_some_and(|value| value.contains("predict")),
        "the override span is a prediction: {fields:?}"
    );
    assert!(
        fields
            .get("mode")
            .is_some_and(|value| value.contains("InProcessSynchronous")),
        "the configured execution mode must be recorded: {fields:?}"
    );
    assert_eq!(fields.get("rows").map(String::as_str), Some("1"));
    assert_eq!(fields.get("columns").map(String::as_str), Some("2"));
}

#[test]
fn the_predictor_reports_the_default_backend_and_execution_mode() {
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(affine_regressor_model())
        .build()
        .expect("the fixture imports");
    assert_eq!(predictor.backend(), SKBackendKind::Faer);
    assert_eq!(predictor.execution_mode(), SKExecutionMode::Automatic);
}
