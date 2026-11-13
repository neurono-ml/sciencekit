//! Companion TDD tests for observability spans and backend transfer
//! (spec `linear-support-vector-classifier`, spec `backend-selection`).

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKExecutionMode, SKSupervisedFit,
};
use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

use super::SKLinearSVC;

#[derive(Default)]
struct DebugFieldVisitor(HashMap<String, String>);

impl Visit for DebugFieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
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

#[test]
fn configured_backend_is_recorded_in_fit_and_prediction_spans() {
    let features = array![
        [-2.0_f64, -1.0],
        [-1.5, -0.5],
        [-1.0, -1.5],
        [1.0, 1.5],
        [1.5, 0.5],
        [2.0, 1.0]
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1];

    tracing::callsite::rebuild_interest_cache();
    let recorder = SpanFieldRecorder::default();
    let subscriber = tracing_subscriber::registry().with(recorder.clone());

    tracing::subscriber::with_default(subscriber, || {
        let model = SKLinearSVC::<f64>::build()
            .backend(SKBackendKind::MatrixMultiply)
            .execution_mode(SKExecutionMode::InProcessSynchronous)
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap();
        let _ = model.decision_function(features.view()).unwrap();
        let _ = model.predict_labels(features.view()).unwrap();

        assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
        assert_eq!(
            model.execution_mode(),
            SKExecutionMode::InProcessSynchronous
        );
    });

    let spans = recorder.spans.lock().expect("span lock");
    assert!(spans.len() >= 3, "expected fit plus two prediction spans");
    assert!(spans.iter().all(|fields| {
        fields
            .get("backend")
            .is_some_and(|backend| backend.contains("matrixmultiply"))
    }));
}
