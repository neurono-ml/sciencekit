//! Classifier probability averaging, tie-breaking and composition tests.

use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBackendKind, SKBuilder, SKClassifierPredictor, SKSupervisedFit};
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_neighbors::SKKNeighborsClassifier;
use sciencekit_tree::SKDecisionTreeClassifier;
use tracing_subscriber::layer::SubscriberExt;

use crate::SKBaggingClassifier;

use super::fixtures::{BackendRecorder, ScriptedBaseClassifier};

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
