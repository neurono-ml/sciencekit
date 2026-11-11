//! Companion tests for [`SKBaggingClassifier`] probability averaging,
//! canonical tie-breaking, non-tree composition and span backend recording.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKDataView, SKError, SKSupervisedFit,
    SKTargetView,
};
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_neighbors::SKKNeighborsClassifier;
use sciencekit_tree::SKDecisionTreeClassifier;
use tracing::field::{Field, Visit};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;

use crate::SKBaggingClassifier;

#[derive(Clone, Debug)]
struct ScriptedBaseClassifier {
    call_counter: Arc<AtomicUsize>,
    class_zero_probabilities: Vec<f64>,
}

#[derive(Clone, Debug)]
struct ScriptedClassifierModel {
    class_zero_probability: f64,
}

impl SKSupervisedFit<f64> for ScriptedBaseClassifier {
    type Model = ScriptedClassifierModel;
    type Error = SKError;

    fn fit<'a, X, T>(&self, _features: X, _targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
        T: TryInto<SKTargetView<'a, f64>, Error = SKError>,
    {
        let slot = self.call_counter.fetch_add(1, Ordering::SeqCst);
        Ok(ScriptedClassifierModel {
            class_zero_probability: self.class_zero_probabilities[slot],
        })
    }
}

impl SKClassifierPredictor<f64> for ScriptedClassifierModel {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let probabilities = self.predict_probabilities(features)?;
        Ok(Array1::from_shape_fn(probabilities.nrows(), |row| {
            usize::from(probabilities[[row, 1]] > probabilities[[row, 0]]) as i64
        }))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let rows = match features.try_into()? {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
            _ => 0,
        };
        Ok(Array2::from_shape_fn((rows, 2), |(_, column)| {
            if column == 0 {
                self.class_zero_probability
            } else {
                1.0 - self.class_zero_probability
            }
        }))
    }
}

#[test]
fn hand_computed_two_estimator_classifier_averages_probabilities() {
    let features =
        Array2::from_shape_vec((4, 2), vec![0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0]).unwrap();
    let labels = Array1::from_vec(vec![0_i64, 0, 1, 1]);
    let scripted = ScriptedBaseClassifier {
        call_counter: Arc::new(AtomicUsize::new(0)),
        class_zero_probabilities: vec![0.2, 0.6],
    };

    let model = SKBaggingClassifier::build(scripted)
        .number_of_estimators(2)
        .bootstrap(false)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let probabilities = model.predict_probabilities(features.view()).unwrap();
    let predicted_labels = model.predict_labels(features.view()).unwrap();
    probabilities.rows().into_iter().for_each(|row| {
        assert!((row[0] - 0.4).abs() < 1e-12);
        assert!((row[1] - 0.6).abs() < 1e-12);
    });
    assert_eq!(predicted_labels, Array1::from_vec(vec![1, 1, 1, 1]));
}

#[test]
fn equal_average_probabilities_break_ties_toward_smallest_canonical_index() {
    let features =
        Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    let nominal_labels = ["beta", "beta", "alpha", "alpha"];
    let scripted = ScriptedBaseClassifier {
        call_counter: Arc::new(AtomicUsize::new(0)),
        class_zero_probabilities: vec![0.2, 0.8],
    };

    let model = SKBaggingClassifier::build(scripted)
        .number_of_estimators(2)
        .bootstrap(false)
        .build()
        .unwrap()
        .fit(features.view(), &nominal_labels[..])
        .unwrap();

    assert_eq!(model.classes().label_of(0), Some("beta"));
    let probabilities = model.predict_probabilities(features.view()).unwrap();
    probabilities.rows().into_iter().for_each(|row| {
        assert!((row[0] - 0.5).abs() < 1e-12);
        assert!((row[1] - 0.5).abs() < 1e-12);
    });
    assert_eq!(
        model.predict_labels(features.view()).unwrap(),
        Array1::from_vec(vec![0, 0, 0, 0])
    );
}

#[test]
fn tree_logistic_and_neighbor_classifiers_compose_and_record_backend_span() {
    let features = Array2::from_shape_fn((30, 2), |(row, column)| {
        let cluster = if row < 15 { -3.0 } else { 3.0 };
        cluster + ((row % 5) as f64) * 0.1 + (column as f64) * 0.2
    });
    let labels = Array1::from_shape_fn(30, |row| i64::from(row >= 15));

    let tree_base = SKDecisionTreeClassifier::<f64>::build().build().unwrap();
    let tree_bagging = SKBaggingClassifier::build(tree_base)
        .number_of_estimators(7)
        .random_seed(42)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let recorded_backends = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::registry().with(BackendRecorder {
        recorded: Arc::clone(&recorded_backends),
    });
    let tree_predictions = tracing::subscriber::with_default(subscriber, || {
        tree_bagging.predict_labels(features.view()).unwrap()
    });
    assert_eq!(tree_predictions, labels);
    assert!(
        recorded_backends
            .lock()
            .unwrap()
            .iter()
            .any(|backend| backend.contains("matrixmultiply"))
    );

    let logistic_base = SKLogisticRegression::<f64>::build().build().unwrap();
    let logistic_predictions = SKBaggingClassifier::build(logistic_base)
        .number_of_estimators(5)
        .random_seed(11)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap()
        .predict_labels(features.view())
        .unwrap();
    assert_eq!(logistic_predictions, labels);

    let knn_base = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .unwrap();
    let knn_predictions = SKBaggingClassifier::build(knn_base)
        .number_of_estimators(5)
        .random_seed(11)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap()
        .predict_labels(features.view())
        .unwrap();
    assert_eq!(knn_predictions, labels);
}

struct BackendRecorder {
    recorded: Arc<Mutex<Vec<String>>>,
}

struct BackendVisitor {
    backend: Option<String>,
}

impl Visit for BackendVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "backend" {
            self.backend = Some(format!("{value:?}"));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "backend" {
            self.backend = Some(value.to_string());
        }
    }
}

impl<S: tracing::Subscriber> Layer<S> for BackendRecorder {
    fn on_new_span(
        &self,
        attributes: &tracing::span::Attributes<'_>,
        _id: &tracing::span::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut visitor = BackendVisitor { backend: None };
        attributes.record(&mut visitor);
        if let Some(backend) = visitor.backend {
            self.recorded.lock().unwrap().push(backend);
        }
    }
}
