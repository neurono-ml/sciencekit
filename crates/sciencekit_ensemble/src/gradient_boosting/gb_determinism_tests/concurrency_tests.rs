//! Lock-free concurrent prediction and regressor span backend propagation
//! tests (task 6.4).

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKExecutionMode, SKRegressorPredictor,
    SKSupervisedFit,
};
use sprs::CsMat;
use tracing::Subscriber;
use tracing::field::Visit;
use tracing::span::{Attributes, Id};
use tracing::subscriber::Interest;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

use crate::gradient_boosting::{SKGradientBoostingClassifier, SKGradientBoostingRegressor};

fn sparse_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut indptr = Vec::with_capacity(dense.nrows() + 1);
    let (mut indices, mut data) = (Vec::new(), Vec::new());
    indptr.push(0);
    dense.rows().into_iter().for_each(|row| {
        row.iter()
            .enumerate()
            .filter(|(_, value)| **value != 0.0)
            .for_each(|(column, &value)| {
                indices.push(column);
                data.push(value);
            });
        indptr.push(indices.len());
    });
    CsMat::new((dense.nrows(), dense.ncols()), indptr, indices, data)
}

#[derive(Clone, Default)]
struct SpanFieldRecorder(Arc<Mutex<Vec<HashMap<String, String>>>>);
struct DebugVisitor(HashMap<String, String>);

impl Visit for DebugVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        self.0
            .insert(field.name().to_string(), format!("{value:?}"));
    }
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for SpanFieldRecorder {
    fn register_callsite(&self, _metadata: &'static tracing::Metadata<'_>) -> Interest {
        Interest::always()
    }

    fn on_new_span(&self, attributes: &Attributes<'_>, _id: &Id, _context: Context<'_, S>) {
        let mut visitor = DebugVisitor(HashMap::new());
        attributes.values().record(&mut visitor);
        self.0.lock().expect("lock").push(visitor.0);
    }
}

/// Scenario: Concurrent prediction agrees across 8 threads over dense and CSR
/// inputs, and regressor prediction spans record the configured backend.
#[test]
fn concurrent_prediction_agrees_with_sequential_and_records_regressor_backend() {
    let rows = 320usize;
    let dense = Arc::new(Array2::from_shape_fn((rows, 4), |(row, col)| {
        if (row + col) % 2 == 0 {
            ((row * 13 + col * 7) % 31) as f64 / 10.0
        } else {
            0.0
        }
    }));
    let sparse = Arc::new(sparse_from_dense(&dense));
    let labels = Array1::from_shape_fn(rows, |row| (row % 3) as i64);
    let responses = Array1::from_shape_fn(rows, |row| 1.5 * dense[[row, 0]] - dense[[row, 2]]);

    let classifier = Arc::new(
        SKGradientBoostingClassifier::<f64>::build()
            .number_of_estimators(16)
            .execution_mode(SKExecutionMode::InProcessSynchronous)
            .random_seed(77)
            .build()
            .expect("builds")
            .fit(dense.view(), labels.view())
            .expect("fits"),
    );
    let regressor = Arc::new(
        SKGradientBoostingRegressor::<f64>::build()
            .number_of_estimators(16)
            .backend(SKBackendKind::MatrixMultiply)
            .random_seed(77)
            .build()
            .expect("builds")
            .fit(dense.view(), responses.view())
            .expect("fits"),
    );

    let recorder = SpanFieldRecorder::default();
    let subscriber = tracing_subscriber::registry().with(recorder.clone());
    let expected_values = tracing::subscriber::with_default(subscriber, || {
        regressor.predict(dense.view()).expect("predict")
    });
    assert_eq!(
        recorder.0.lock().expect("lock")[0]
            .get("backend")
            .map(String::as_str),
        Some("matrixmultiply")
    );
    let expected_probs = classifier.predict_probabilities(dense.view()).expect("p");
    let expected_labels = classifier.predict_labels(dense.view()).expect("l");

    let handles: Vec<_> = (0..8)
        .map(|thread_id| {
            let (classifier, regressor) = (Arc::clone(&classifier), Arc::clone(&regressor));
            let (dense, sparse) = (Arc::clone(&dense), Arc::clone(&sparse));
            std::thread::spawn(move || {
                if thread_id % 2 == 0 {
                    (
                        classifier.predict_probabilities(dense.view()).expect("d"),
                        classifier.predict_labels(dense.view()).expect("dl"),
                        regressor.predict(dense.view()).expect("dr"),
                    )
                } else {
                    (
                        classifier.predict_probabilities(sparse.view()).expect("s"),
                        classifier.predict_labels(sparse.view()).expect("sl"),
                        regressor.predict(sparse.view()).expect("sr"),
                    )
                }
            })
        })
        .collect();
    handles.into_iter().for_each(|handle| {
        let (probs, predicted_labels, values) = handle.join().expect("thread joins");
        assert_eq!(probs, expected_probs);
        assert_eq!(predicted_labels, expected_labels);
        assert_eq!(values, expected_values);
    });
}
