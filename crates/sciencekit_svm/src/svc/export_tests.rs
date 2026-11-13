//! Companion tests: the support-vector models round-trip their state.

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit,
    sk_assert_export_round_trip,
};

use crate::linear_svc::SKLinearSVC;
use crate::svc::SKSVC;
use crate::svr::SKSVR;

#[test]
fn svc_round_trips_pairwise_support_vectors_and_classes() {
    let features = array![[0.0_f64, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let labels = ["left", "left", "right", "right"];
    let model = SKSVC::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), labels.as_slice())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.number_of_features(), 2);
    assert_eq!(
        imported.classes().labels().collect::<Vec<_>>(),
        vec!["left", "right"]
    );
    assert_eq!(
        model.predict_labels(features.view()).unwrap(),
        imported.predict_labels(features.view()).unwrap()
    );
}

#[test]
fn svr_round_trips_support_vectors_and_kernel() {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0], [4.0]];
    let targets = array![1.0_f64, 3.0, 5.0, 7.0, 9.0];
    let model = SKSVR::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.kernel_parameters(), model.kernel_parameters());
    assert_eq!(
        model.predict(features.view()).unwrap(),
        imported.predict(features.view()).unwrap()
    );
}

#[test]
fn linear_svc_round_trips_heads_and_class_table() {
    let features = array![[0.0_f64, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let labels = ["a", "a", "b", "b"];
    let model = SKLinearSVC::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), labels.as_slice())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.heads().len(), model.heads().len());
    assert_eq!(
        model.predict_labels(features.view()).unwrap(),
        imported.predict_labels(features.view()).unwrap()
    );
}
