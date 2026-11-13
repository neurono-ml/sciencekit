//! Export → import round trips beyond the linear/scaler/tree baseline.
//!
//! Every supported family is fitted natively, exported to ONNX, imported back
//! and predicted through the generic predictor. The comparison is against the
//! original Rust model with the tolerances the export tests already document.

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKError, SKFeatureTransformer, SKSupervisedFit,
    SKUnsupervisedFit,
};

use super::SKOnnxPredictor;
use crate::SKToOnnx;

fn binary_features() -> Array2<f64> {
    array![
        [-2.0, -1.0],
        [-1.5, -0.5],
        [-1.0, -1.5],
        [1.0, 1.0],
        [1.5, 0.5],
        [2.0, 1.5]
    ]
}

fn multiclass_features() -> Array2<f64> {
    array![
        [-2.0, -1.0],
        [-1.5, -0.5],
        [2.0, 2.0],
        [2.5, 1.5],
        [-2.0, 2.5],
        [-1.5, 3.0]
    ]
}

fn wide_features() -> Array2<f64> {
    array![
        [1.0, 8.0, 2.0],
        [1.0, 0.0, 5.0],
        [9.0, 1.0, 1.0],
        [9.0, 6.0, 2.0],
        [1.0, 7.0, 2.0],
        [9.0, 0.0, 5.0]
    ]
}

fn import<F: sciencekit_common::SKFloat>(model: &crate::graph::SKOnnxModel) -> SKOnnxPredictor<F> {
    SKOnnxPredictor::<F>::builder()
        .model_bytes(model.bytes().to_vec())
        .build()
        .expect("the exported model imports")
}

fn assert_close<F: sciencekit_common::SKFloat>(expected: &[F], actual: &[F], tolerance: F) {
    assert_eq!(expected.len(), actual.len(), "the output size differs");
    let tolerance = tolerance.to_f64().expect("the tolerance converts");
    for (expected, actual) in expected.iter().zip(actual) {
        let expected = expected.to_f64().expect("the fixture converts");
        let actual = actual.to_f64().expect("the fixture converts");
        let difference = (expected - actual).abs();
        assert!(
            difference <= tolerance,
            "{expected:?} != {actual:?} (difference {difference:?})"
        );
    }
}

#[test]
fn binary_rbf_svc_round_trips_labels_and_decision_values() {
    use sciencekit_svm::{SKKernel, SKKernelGamma, SKSVC};

    let features = binary_features();
    let labels = ["left", "left", "left", "right", "right", "right"];
    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .gamma(SKKernelGamma::Value(0.5_f64))
        .probability(false)
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), labels.as_slice())
        .expect("the classifier fits");
    let expected_labels = model.predict_labels(features.view()).expect("labels");
    let expected_decisions = model.decision_function(features.view()).expect("decisions");
    let imported = import::<f64>(&model.to_onnx().expect("the classifier exports"));
    assert_eq!(
        imported
            .predict_labels(features.view())
            .expect("the imported labels"),
        expected_labels
    );
    let restored = imported
        .predict_probabilities(features.view())
        .expect("the imported decision matrix");
    assert_eq!(restored.shape(), expected_decisions.shape());
    assert_close(
        &expected_decisions.iter().copied().collect::<Vec<_>>(),
        &restored.iter().copied().collect::<Vec<_>>(),
        1e-9,
    );
}

#[test]
fn multiclass_rbf_svc_round_trips_labels_and_decision_values() {
    use sciencekit_svm::{SKKernel, SKKernelGamma, SKSVC};

    let features = multiclass_features();
    let labels = ["a", "a", "b", "b", "c", "c"];
    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .gamma(SKKernelGamma::Value(0.5_f64))
        .probability(false)
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), labels.as_slice())
        .expect("the classifier fits");
    let expected_labels = model.predict_labels(features.view()).expect("labels");
    let expected_decisions = model.decision_function(features.view()).expect("decisions");
    let imported = import::<f64>(&model.to_onnx().expect("the classifier exports"));
    assert_eq!(
        imported
            .predict_labels(features.view())
            .expect("the imported labels"),
        expected_labels
    );
    let restored = imported
        .predict_probabilities(features.view())
        .expect("the imported decision matrix");
    assert_eq!(restored.shape(), expected_decisions.shape());
    assert_close(
        &expected_decisions.iter().copied().collect::<Vec<_>>(),
        &restored.iter().copied().collect::<Vec<_>>(),
        1e-9,
    );
}

#[test]
fn pca_round_trips_the_transform() {
    use sciencekit_decomposition::pca::SKPCA;

    let features = wide_features();
    let model = SKPCA::<f64>::build()
        .number_of_components(2)
        .build()
        .expect("the decomposition configuration is valid")
        .fit(&features)
        .expect("the decomposition fits");
    let expected = model.transform(&features).expect("the transform");
    let imported = import::<f64>(&model.to_onnx().expect("the decomposition exports"));
    let outputs = imported
        .predict_arrays(features.view())
        .expect("the imported transform");
    let restored = outputs.values().next().expect("one output");
    assert_eq!(restored.shape(), expected.shape());
    assert_close(
        &expected.iter().copied().collect::<Vec<_>>(),
        &restored.iter().copied().collect::<Vec<_>>(),
        1e-9,
    );
}

#[test]
fn truncated_svd_round_trips_the_transform() {
    use sciencekit_decomposition::truncated_svd::SKTruncatedSVD;

    let features = wide_features();
    let model = SKTruncatedSVD::<f64>::build()
        .number_of_components(2)
        .build()
        .expect("the decomposition configuration is valid")
        .fit(&features)
        .expect("the decomposition fits");
    let expected = model.transform(&features).expect("the transform");
    let imported = import::<f64>(&model.to_onnx().expect("the decomposition exports"));
    let outputs = imported
        .predict_arrays(features.view())
        .expect("the imported transform");
    let restored = outputs.values().next().expect("one output");
    assert_eq!(restored.shape(), expected.shape());
    assert_close(
        &expected.iter().copied().collect::<Vec<_>>(),
        &restored.iter().copied().collect::<Vec<_>>(),
        1e-9,
    );
}

#[test]
fn standard_scaler_without_mean_round_trips_the_transform() {
    use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

    let features = wide_features();
    let model = SKStandardScaler::build()
        .with_mean(false)
        .with_std(true)
        .build()
        .expect("the scaler configuration is valid")
        .fit(&features)
        .expect("the scaler fits");
    let expected = model.transform(&features).expect("the transform");
    let imported = import::<f64>(&model.to_onnx().expect("the scaler exports"));
    let outputs = imported
        .predict_arrays(features.view())
        .expect("the imported transform");
    let restored = outputs.values().next().expect("one output");
    assert_eq!(restored.shape(), expected.shape());
    assert_close(
        &expected.iter().copied().collect::<Vec<_>>(),
        &restored.iter().copied().collect::<Vec<_>>(),
        1e-9,
    );
}

#[test]
fn dense_knn_classifier_round_trips_labels_and_probabilities() {
    use sciencekit_neighbors::SKKNeighborsClassifier;

    let features = wide_features();
    let labels = ["left", "left", "right", "right", "left", "right"];
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), labels.as_slice())
        .expect("the classifier fits");
    let expected_labels = model.predict_labels(features.view()).expect("labels");
    let expected_probabilities = model
        .predict_probabilities(features.view())
        .expect("probabilities");
    let imported = import::<f64>(&model.to_onnx().expect("the classifier exports"));
    assert_eq!(
        imported
            .predict_labels(features.view())
            .expect("the imported labels"),
        expected_labels
    );
    let restored = imported
        .predict_probabilities(features.view())
        .expect("the imported probabilities");
    assert_eq!(restored.shape(), expected_probabilities.shape());
    assert_close(
        &expected_probabilities.iter().copied().collect::<Vec<_>>(),
        &restored.iter().copied().collect::<Vec<_>>(),
        1e-9,
    );
}

#[test]
fn sparse_trained_knn_cannot_round_trip_because_export_rejects_csr() {
    use sciencekit_neighbors::SKKNeighborsClassifier;

    let dense = wide_features();
    let mut triplet = sprs::TriMat::new((dense.nrows(), dense.ncols()));
    for ((row, column), value) in dense.indexed_iter() {
        triplet.add_triplet(row, column, *value);
    }
    let sparse = triplet.to_csr();
    let labels = array![0_i64, 0, 1, 1, 0, 1];
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .expect("the classifier configuration is valid")
        .fit(sparse.view(), labels.view())
        .expect("the classifier fits");
    let error = model
        .to_onnx()
        .expect_err("the exporter documents CSR states as unsupported");
    match error {
        SKError::UnsupportedOperation { operation, reason } => {
            assert_eq!(operation, "onnx_export");
            assert!(reason.contains("SKKNeighborsClassifier"), "{reason}");
            assert!(reason.contains("CSR") || reason.contains("csr"), "{reason}");
        }
        other => panic!("expected an unsupported-operation error, got {other}"),
    }
}

#[test]
fn histogram_gradient_boosting_cannot_round_trip_because_export_rejects_boosting() {
    use sciencekit_ensemble::SKHistGradientBoostingRegressor;

    let features = array![
        [0.0, 1.0],
        [1.0, 0.0],
        [2.0, 1.0],
        [3.0, 0.0],
        [4.0, 1.0],
        [f64::NAN, 0.0]
    ];
    let targets: Array1<f64> = array![0.0, 1.0, 2.0, 3.0, 4.0, 2.5];
    let model = SKHistGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(2)
        .maximum_bins(4)
        .minimum_samples_leaf(1)
        .categorical_features(vec![1])
        .build()
        .expect("the boosting configuration is valid")
        .fit(features.view(), targets.view())
        .expect("the boosting model fits");
    let error = model
        .to_onnx()
        .expect_err("the exporter documents boosting as unsupported");
    match error {
        SKError::UnsupportedOperation { operation, reason } => {
            assert_eq!(operation, "onnx_export");
            assert!(
                reason.contains("SKHistGradientBoostingRegressor"),
                "{reason}"
            );
        }
        other => panic!("expected an unsupported-operation error, got {other}"),
    }
}
