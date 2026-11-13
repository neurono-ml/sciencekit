//! Companion tests: logistic regression round-trips heads and class tables.

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKSupervisedFit, sk_assert_export_round_trip,
};

use crate::logistic_regression::SKLogisticRegression;

#[test]
fn logistic_regression_round_trips_binary_predictions() {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0]];
    let labels = ["low", "low", "high", "high"];
    let model = SKLogisticRegression::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), labels.as_slice())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.number_of_classes(), 2);
    assert_eq!(
        imported.classes().labels().collect::<Vec<_>>(),
        vec!["low", "high"]
    );
    let probe = array![[0.5_f64], [2.5]];
    assert_eq!(
        model.predict_labels(probe.view()).unwrap(),
        imported.predict_labels(probe.view()).unwrap()
    );
}

#[test]
fn logistic_regression_round_trips_probabilities() {
    let features = array![[0.0_f64, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]];
    let labels = ["a", "b", "a", "b"];
    let model = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.as_slice())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(
        model.predict_probabilities(features.view()).unwrap(),
        imported.predict_probabilities(features.view()).unwrap()
    );
}
