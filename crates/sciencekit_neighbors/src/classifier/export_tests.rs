//! Companion tests: neighbor models round-trip their training design.

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit,
    sk_assert_export_round_trip,
};
use sprs::TriMat;

use crate::classifier::SKKNeighborsClassifier;
use crate::regressor::SKKNeighborsRegressor;

#[test]
fn knn_classifier_round_trips_dense_training() {
    let features = array![[0.0_f64, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let labels = ["a", "b", "a", "b"];
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(2)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), labels.as_slice())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.training_representation(), "dense");
    assert_eq!(
        model.predict_labels(features.view()).unwrap(),
        imported.predict_labels(features.view()).unwrap()
    );
}

#[test]
fn knn_regressor_keeps_a_sparse_design_sparse() {
    let mut triplets = TriMat::new((4, 3));
    triplets.add_triplet(0, 0, 1.0_f64);
    triplets.add_triplet(1, 0, 2.0);
    triplets.add_triplet(2, 1, 3.0);
    triplets.add_triplet(3, 2, 4.0);
    let features = triplets.to_csr();
    let targets = array![1.0_f64, 2.0, 3.0, 4.0];
    let model = SKKNeighborsRegressor::<f64>::build()
        .nearest_neighbors_count(1)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.training_representation(), "csr");
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        model.predict(features.view()).unwrap(),
        imported.predict(features.view()).unwrap()
    );
}
