//! Companion tests: the boosters round-trip their stage trees.

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit,
    sk_assert_export_round_trip,
};

use super::{SKGradientBoostingClassifier, SKGradientBoostingRegressor};

#[test]
fn gradient_boosting_regressor_round_trips_stages() {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let targets = array![1.0_f64, 3.0, 5.0, 7.0, 9.0, 11.0];
    let model = SKGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(5)
        .without_early_stopping()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.number_of_stages(), model.number_of_stages());
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        model.predict(features.view()).unwrap(),
        imported.predict(features.view()).unwrap()
    );
}

#[test]
fn gradient_boosting_classifier_round_trips_classes_and_stages() {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let labels = ["a", "a", "a", "b", "b", "b"];
    let model = SKGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(5)
        .without_early_stopping()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), labels.as_slice())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.number_of_classes(), 2);
    assert_eq!(
        imported.classes().labels().collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert_eq!(
        model.predict_labels(features.view()).unwrap(),
        imported.predict_labels(features.view()).unwrap()
    );
}
