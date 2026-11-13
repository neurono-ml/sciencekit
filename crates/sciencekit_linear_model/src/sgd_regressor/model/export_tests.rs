//! Companion tests: the stochastic-gradient models round-trip their state.

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit,
    sk_assert_export_round_trip,
};

use crate::sgd::SKSGDLearningRate;
use crate::sgd::SKSGDPenalty;
use crate::sgd::SKSGDRegressorLoss;
use crate::sgd_classifier::SKSGDClassifier;
use crate::sgd_regressor::SKSGDRegressor;

#[test]
fn sgd_regressor_round_trips_schedule_and_iterate() {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0]];
    let targets = array![1.0_f64, 3.0, 5.0, 7.0];
    let model = SKSGDRegressor::<f64>::build()
        .loss(SKSGDRegressorLoss::SquaredError)
        .penalty(SKSGDPenalty::L2)
        .learning_rate(SKSGDLearningRate::Constant)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.learning_rate(), SKSGDLearningRate::Constant);
    assert_eq!(imported.epochs_taken(), model.epochs_taken());
    let probe = array![[0.5_f64], [2.5]];
    assert_eq!(
        model.predict(probe.view()).unwrap(),
        imported.predict(probe.view()).unwrap()
    );
}

#[test]
fn sgd_classifier_round_trips_classes_and_heads() {
    let features = array![[0.0_f64, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]];
    let labels = ["a", "b", "a", "b"];
    let model = SKSGDClassifier::<f64>::build()
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
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        model.predict_labels(features.view()).unwrap(),
        imported.predict_labels(features.view()).unwrap()
    );
}
