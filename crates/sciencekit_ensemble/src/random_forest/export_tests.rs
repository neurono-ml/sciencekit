//! Companion tests: the forests round-trip their embedded trees.

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit,
    sk_assert_export_round_trip,
};

use super::{SKRandomForestClassifier, SKRandomForestRegressor};

#[test]
fn random_forest_classifier_round_trips_members_and_oob() {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let labels = ["a", "a", "a", "b", "b", "b"];
    let model = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(4)
        .random_seed(7)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), labels.as_slice())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.number_of_trees(), model.number_of_trees());
    assert_eq!(
        imported.classes().labels().collect::<Vec<_>>(),
        model.classes().labels().collect::<Vec<_>>()
    );
    assert_eq!(
        model.predict_labels(features.view()).unwrap(),
        imported.predict_labels(features.view()).unwrap()
    );
}

#[test]
fn random_forest_regressor_round_trips_trees() {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let targets = array![1.0_f64, 3.0, 5.0, 7.0, 9.0, 11.0];
    let model = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(4)
        .random_seed(7)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.number_of_trees(), model.number_of_trees());
    assert_eq!(
        model.predict(features.view()).unwrap(),
        imported.predict(features.view()).unwrap()
    );
}
