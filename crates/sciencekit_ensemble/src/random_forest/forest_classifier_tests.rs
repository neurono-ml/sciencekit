//! Companion tests for classifier probability averaging, canonical
//! tie-breaking and prediction observability (task 2.3).

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilder, SKClassifierPredictor, SKSupervisedFit};
use sciencekit_tree::SKClassWeight;
use tracing::Subscriber;
use tracing::field::Visit;
use tracing::span::{Attributes, Id};
use tracing::subscriber::Interest;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

use super::SKRandomForestClassifier;

/// Find a deterministic seed where two zero-depth bootstrap trees over five
/// rows `[0, 0, 1, 1, 1]` hold class-0 probabilities `0.2` and `0.6`, then
/// verify the forest averages them to `0.4` and predicts class `1`.
#[test]
fn hand_computed_two_tree_forest_averages_probabilities() {
    let features = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
    let labels = array![0_i64, 0, 1, 1, 1];

    let model = (0_u64..500)
        .find_map(|seed| {
            let candidate = SKRandomForestClassifier::<f64>::build()
                .number_of_trees(2)
                .maximum_depth(0)
                .random_seed(seed)
                .build()
                .expect("builds")
                .fit(&features, labels.view())
                .expect("fit succeeds");
            let p0_first = candidate.tree(0)?.node(0)?.value();
            let p0_second = candidate.tree(1)?.node(0)?.value();
            if p0_first.len() == 2 && p0_second.len() == 2 {
                let probs = candidate.predict_probabilities(array![[0.0]].view()).ok()?;
                let first_ok =
                    (p0_first[0] - 0.2).abs() < 1e-12 || (p0_first[1] - 0.2).abs() < 1e-12;
                let second_ok =
                    (p0_second[0] - 0.6).abs() < 1e-12 || (p0_second[1] - 0.6).abs() < 1e-12;
                if (probs[[0, 0]] - 0.4).abs() < 1e-12 && first_ok && second_ok {
                    return Some(candidate);
                }
            }
            None
        })
        .expect("a seed within 0..500 produces 0.2 and 0.6 class-0 trees");

    let probabilities = model
        .predict_probabilities(array![[0.5], [3.5]].view())
        .expect("predict_probabilities succeeds");
    assert_eq!(probabilities.shape(), &[2, 2]);
    assert!((probabilities[[0, 0]] - 0.4).abs() < 1e-12);
    assert!((probabilities[[0, 1]] - 0.6).abs() < 1e-12);
    let predicted = model
        .predict_labels(array![[0.5], [3.5]].view())
        .expect("predict_labels succeeds");
    assert_eq!(predicted, array![1_i64, 1]);
}

/// Exact probability ties break deterministically toward the smallest canonical
/// class index (`0`).
#[test]
fn equal_average_probabilities_break_ties_by_smallest_canonical_index() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let labels = ["beta", "alpha", "beta", "alpha"];
    let model = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(6)
        .maximum_depth(0)
        .bootstrap(false)
        .build()
        .expect("builds")
        .fit(&features, &labels[..])
        .expect("fit succeeds");

    assert_eq!(model.classes().label_of(0), Some("beta"));
    assert_eq!(model.classes().label_of(1), Some("alpha"));

    let probabilities = model
        .predict_probabilities(&features)
        .expect("probabilities");
    for row in probabilities.rows() {
        assert!((row[0] - 0.5).abs() < 1e-12);
        assert!((row[1] - 0.5).abs() < 1e-12);
    }
    let predicted = model.predict_labels(&features).expect("labels");
    assert_eq!(predicted, array![0_i64, 0, 0, 0]);
}

/// Multi-class nominal classification preserves canonical class order, handles
/// class weights and separates three clusters cleanly.
#[test]
fn multiclass_nominal_forest_separates_clusters_and_preserves_class_table() {
    let features = array![
        [0.0, 0.1],
        [0.1, 0.0],
        [0.2, 0.1],
        [5.0, 5.1],
        [5.1, 5.0],
        [5.2, 5.1],
        [10.0, 0.1],
        [10.1, 0.0],
        [10.2, 0.1],
    ];
    let labels = [
        "red", "red", "red", "green", "green", "green", "blue", "blue", "blue",
    ];
    let model = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(25)
        .class_weight(SKClassWeight::Explicit(vec![1.0, 1.5, 2.0]))
        .random_seed(11)
        .build()
        .expect("builds")
        .fit(&features, &labels[..])
        .expect("fit succeeds");

    assert_eq!(model.number_of_classes(), 3);
    assert_eq!(model.classes().label_of(0), Some("red"));
    assert_eq!(model.classes().label_of(1), Some("green"));
    assert_eq!(model.classes().label_of(2), Some("blue"));

    let probabilities = model
        .predict_probabilities(&features)
        .expect("probabilities");
    for row in probabilities.rows() {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }
    let predicted = model.predict_labels(&features).expect("labels");
    assert_eq!(predicted, array![0_i64, 0, 0, 1, 1, 1, 2, 2, 2]);
}

#[derive(Clone, Default)]
struct SpanFieldRecorder {
    spans: Arc<Mutex<Vec<HashMap<String, String>>>>,
}

struct DebugVisitor(HashMap<String, String>);

impl Visit for DebugVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        self.0
            .insert(field.name().to_string(), format!("{value:?}"));
    }
}

impl<S> Layer<S> for SpanFieldRecorder
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn register_callsite(&self, _metadata: &'static tracing::Metadata<'_>) -> Interest {
        Interest::always()
    }

    fn on_new_span(&self, attributes: &Attributes<'_>, _id: &Id, _context: Context<'_, S>) {
        let mut visitor = DebugVisitor(HashMap::new());
        attributes.values().record(&mut visitor);
        self.spans.lock().expect("lock").push(visitor.0);
    }
}

/// A non-default backend configured on the builder reaches the fitted model's
/// prediction span.
#[test]
fn configured_backend_is_recorded_in_prediction_span() {
    let features = array![[0.0, 1.0], [0.2, 0.8], [3.0, 0.1], [3.2, 0.0]];
    let labels = array![0_i64, 0, 1, 1];
    let model = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(4)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");

    let recorder = SpanFieldRecorder::default();
    let subscriber = tracing_subscriber::registry().with(recorder.clone());
    tracing::subscriber::with_default(subscriber, || {
        let _ = model
            .predict_labels(&features)
            .expect("prediction succeeds");
    });

    let spans = recorder.spans.lock().expect("lock");
    assert_eq!(spans.len(), 1);
    assert_eq!(
        spans[0].get("operation").map(String::as_str),
        Some("predict")
    );
    assert_eq!(
        spans[0].get("backend").map(String::as_str),
        Some("matrixmultiply")
    );
}
