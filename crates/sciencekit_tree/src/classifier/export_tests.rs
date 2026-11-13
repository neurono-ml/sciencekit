//! Companion tests: the decision trees round-trip their node arrays.

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit,
    sk_assert_export_round_trip,
};

use crate::classifier::SKDecisionTreeClassifier;
use crate::regressor::SKDecisionTreeRegressor;

#[test]
fn decision_tree_classifier_round_trips_nodes_and_predictions() {
    let features = array![[0.0_f64, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let labels = ["a", "a", "b", "b"];
    let model = SKDecisionTreeClassifier::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), labels.as_slice())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.node_count(), model.node_count());
    assert_eq!(imported.depth(), model.depth());
    assert_eq!(
        model.predict_labels(features.view()).unwrap(),
        imported.predict_labels(features.view()).unwrap()
    );
}

#[test]
fn decision_tree_regressor_round_trips_nodes_and_predictions() {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0]];
    let targets = array![1.0_f64, 3.0, 5.0, 7.0];
    let model = SKDecisionTreeRegressor::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.node_count(), model.node_count());
    assert_eq!(
        model.predict(features.view()).unwrap(),
        imported.predict(features.view()).unwrap()
    );
}
