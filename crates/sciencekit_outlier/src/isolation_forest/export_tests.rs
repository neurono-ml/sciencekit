//! Companion tests: the isolation forest round-trips its tree ensemble.

use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::array;
use sciencekit_common::{SKBackendKind, sk_assert_export_round_trip};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};

use super::SKContamination;
use super::SKIsolationForest;

#[test]
fn isolation_forest_round_trips_trees_offset_and_scores() {
    let training = array![
        [0.0_f64, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [10.0, 10.0],
        [10.1, 10.0],
        [0.0, 0.2],
    ];
    let model = SKIsolationForest::<f64>::build()
        .number_of_trees(8)
        .random_seed(5)
        .contamination(SKContamination::Fraction(0.2))
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.number_of_trees(), model.number_of_trees());
    assert_eq!(imported.offset(), model.offset());
    assert_eq!(
        model.score_samples(training.view()).unwrap(),
        imported.score_samples(training.view()).unwrap()
    );
    assert_eq!(
        model.predict(training.view()).unwrap(),
        imported.predict(training.view()).unwrap()
    );
}

/// Capture the `backend` field of every span created while predicting.
#[derive(Default)]
struct BackendSpanRecorder {
    next_id: AtomicU64,
    backends: Mutex<Vec<String>>,
}

struct BackendVisitor(Option<String>);

impl Visit for BackendVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "backend" {
            self.0 = Some(value.to_string());
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "backend" {
            self.0 = Some(format!("{value:?}"));
        }
    }
}

impl tracing::Subscriber for BackendSpanRecorder {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, span: &Attributes<'_>) -> Id {
        let mut visitor = BackendVisitor(None);
        span.values().record(&mut visitor);
        if let Some(backend) = visitor.0 {
            self.backends.lock().unwrap().push(backend);
        }
        Id::from_u64(self.next_id.fetch_add(1, Ordering::Relaxed) + 1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn event(&self, _: &tracing::Event<'_>) {}
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
}

#[test]
fn imported_isolation_forest_reports_the_exported_backend_in_its_span() {
    let training = array![
        [0.0_f64, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [10.0, 10.0],
        [10.1, 10.0],
        [0.0, 0.2],
    ];
    let model = SKIsolationForest::<f64>::build()
        .number_of_trees(4)
        .random_seed(5)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();

    let recorder = std::sync::Arc::new(BackendSpanRecorder::default());
    tracing::subscriber::with_default(recorder.clone(), || {
        imported.score_samples(training.view()).unwrap();
    });
    assert!(
        recorder
            .backends
            .lock()
            .unwrap()
            .iter()
            .any(|backend| backend.contains("matrixmultiply")),
        "the imported model must report the exported backend in its span"
    );
}
