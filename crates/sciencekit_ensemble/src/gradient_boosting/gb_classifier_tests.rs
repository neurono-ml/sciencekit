//! Companion tests for `SKGradientBoostingClassifier` binary fit, raw scores,
//! sigmoid probabilities, tie-breaking and span observability (task 4.2).

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilder, SKClassifierPredictor, SKSupervisedFit};
use tracing::Subscriber;
use tracing::field::Visit;
use tracing::span::{Attributes, Id};
use tracing::subscriber::Interest;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

use super::SKGradientBoostingClassifier;

/// Scenario: Binary probabilities are sigmoid of the score, and a hand-computed
/// one-stage binary fit produces exact Newton leaf steps and shrunk log-odds.
#[test]
fn hand_computed_binary_fit_and_zero_score_sigmoid() {
    let features = array![[0.0], [1.0], [10.0], [11.0]];
    let labels = array![0_i64, 0, 1, 1];

    // Depth-0 stump on balanced classes keeps raw score at 0.0 -> p(pos) = 0.5,
    // tie-breaking toward canonical class 0.
    let zero_score_model = SKGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(1)
        .maximum_depth(0)
        .learning_rate(0.5)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fits");
    assert_eq!(zero_score_model.initial_prediction(), Some(0.0));
    let zero_probabilities = zero_score_model
        .predict_probabilities(&features)
        .expect("probabilities");
    zero_probabilities.rows().into_iter().for_each(|row| {
        assert!((row[0] - 0.5).abs() < 1e-12);
        assert!((row[1] - 0.5).abs() < 1e-12);
    });
    assert_eq!(
        zero_score_model.predict_labels(&features).expect("labels"),
        array![0_i64, 0, 0, 0]
    );

    // Depth-1 one-stage fit: Newton leaves [-2.0, +2.0], learning_rate = 0.25 ->
    // raw scores [-0.5, +0.5], positive probabilities sigmoid(-0.5) & sigmoid(0.5).
    let model = SKGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(1)
        .maximum_depth(1)
        .learning_rate(0.25)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fits");

    assert_eq!(model.number_of_stages(), 1);
    assert_eq!(model.trees_per_stage(), 1);
    assert!((model.learning_rate() - 0.25).abs() < 1e-12);
    let tree = model.stage_tree(0, 0).expect("stage 0 tree 0");
    assert!((tree.leaf_value_for(|_| 0.5) - (-2.0)).abs() < 1e-12);
    assert!((tree.leaf_value_for(|_| 10.5) - 2.0).abs() < 1e-12);

    let expected_positive_high = 1.0 / (1.0 + (-0.5_f64).exp());
    let expected_positive_low = 1.0 / (1.0 + 0.5_f64.exp());
    let probabilities = model
        .predict_probabilities(array![[0.5], [10.5]].view())
        .expect("probabilities");
    assert!((probabilities[[0, 1]] - expected_positive_low).abs() < 1e-12);
    assert!((probabilities[[0, 0]] - (1.0 - expected_positive_low)).abs() < 1e-12);
    assert!((probabilities[[1, 1]] - expected_positive_high).abs() < 1e-12);
    assert_eq!(
        model
            .predict_labels(array![[0.5], [10.5]].view())
            .expect("labels"),
        array![0_i64, 1]
    );
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

/// Separable nominal binary fixture reaches 100% accuracy and records the
/// configured backend in prediction spans.
#[test]
fn separable_binary_fixture_and_backend_span_propagation() {
    let features = array![
        [-3.0, -2.0],
        [-2.5, -1.8],
        [-2.0, -2.2],
        [2.0, 2.5],
        [2.4, 2.1],
        [3.0, 2.8],
    ];
    let labels = [
        "negative", "negative", "negative", "positive", "positive", "positive",
    ];
    let model = SKGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(15)
        .learning_rate(0.3)
        .maximum_depth(2)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .expect("builds")
        .fit(&features, &labels[..])
        .expect("fits");

    assert_eq!(model.classes().label_of(0), Some("negative"));
    assert_eq!(model.classes().label_of(1), Some("positive"));
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);

    let recorder = SpanFieldRecorder::default();
    let subscriber = tracing_subscriber::registry().with(recorder.clone());
    let predicted = tracing::subscriber::with_default(subscriber, || {
        model.predict_labels(&features).expect("predicts")
    });
    assert_eq!(predicted, array![0_i64, 0, 0, 1, 1, 1]);

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
