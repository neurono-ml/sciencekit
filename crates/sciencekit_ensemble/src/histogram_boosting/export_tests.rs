//! Companion tests: histogram boosters round-trip binning, masks and routing.

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKExportable, SKImportable, SKModelState,
    SKRegressorPredictor, SKSupervisedFit, SKTensor, sk_assert_export_round_trip,
};
use sprs::TriMat;

use super::{SKHistGradientBoostingClassifier, SKHistGradientBoostingRegressor};

#[test]
fn histogram_regressor_round_trips_binning_and_missing_routing() {
    let features = array![
        [0.0_f64, 1.0],
        [1.0, f64::NAN],
        [2.0, 3.0],
        [3.0, f64::NAN],
        [4.0, 5.0],
        [5.0, 6.0],
    ];
    let targets = array![1.0_f64, 3.0, 5.0, 7.0, 9.0, 11.0];
    let model = SKHistGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(6)
        .maximum_bins(8)
        .without_early_stopping()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.number_of_stages(), model.number_of_stages());
    for feature in 0..2 {
        assert_eq!(imported.bin_edges(feature), model.bin_edges(feature));
    }
    let probe = array![[1.5_f64, f64::NAN], [3.5, 4.0]];
    assert_eq!(
        model.predict(probe.view()).unwrap(),
        imported.predict(probe.view()).unwrap()
    );
}

#[test]
fn histogram_regressor_round_trips_a_sparse_design() {
    let mut triplets = TriMat::new((6, 2));
    triplets.add_triplet(0, 0, 1.0_f64);
    triplets.add_triplet(1, 0, 2.0);
    triplets.add_triplet(2, 1, 3.0);
    triplets.add_triplet(3, 1, 4.0);
    triplets.add_triplet(4, 0, 5.0);
    triplets.add_triplet(5, 1, 6.0);
    let features = triplets.to_csr();
    let targets = array![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let model = SKHistGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(6)
        .without_early_stopping()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(
        model.predict(features.view()).unwrap(),
        imported.predict(features.view()).unwrap()
    );
}

#[test]
fn histogram_classifier_round_trips_categorical_masks() {
    let features = array![
        [1.0_f64, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [3.0, 0.0],
        [2.0, 1.0],
        [3.0, 0.0],
    ];
    let labels = ["a", "b", "a", "b", "a", "b"];
    let model = SKHistGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(6)
        .categorical_features(vec![1])
        .without_early_stopping()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), labels.as_slice())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(
        imported.categories(1),
        model.categories(1),
        "the categorical table survives the round trip"
    );
    assert_eq!(
        imported.classes().labels().collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert_eq!(
        model.predict_labels(features.view()).unwrap(),
        imported.predict_labels(features.view()).unwrap()
    );
}

#[test]
fn histogram_import_rejects_a_truncated_stage_importance_tensor() {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let targets = array![1.0_f64, 3.0, 5.0, 7.0, 9.0, 11.0];
    let model = SKHistGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(3)
        .without_early_stopping()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let state = model.export_state().unwrap();
    let mut tensors: Vec<_> = state
        .tensors()
        .values()
        .filter(|tensor| tensor.name() != "hist_boosting.stage_importances")
        .cloned()
        .collect();
    tensors.push(
        SKTensor::floating("hist_boosting.stage_importances", vec![1], vec![1.0]).unwrap(),
    );
    let broken = SKModelState::new(state.metadata().clone(), tensors).unwrap();
    assert!(matches!(
        super::SKHistGradientBoostingRegressorModel::<f64>::import_state(&broken),
        Err(sciencekit_common::SKError::ShapeMismatch { .. })
    ));
}
