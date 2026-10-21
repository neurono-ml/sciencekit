//! Structured-error tests for `SKRidge` (spec `ridge-regression`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKRegressorPredictor, SKSupervisedFit};

use super::builder::SKRidgeSolver;
use super::core_implementation::SKRidge;
use super::test_datasets::{default_builder, default_ridge, one_feature};

/// Empty input is rejected before any element is processed.
#[test]
fn rejects_empty_design() {
    let features: Array2<f64> = Array2::zeros((0, 2));
    let targets: Array1<f64> = Array1::zeros(0);
    let error = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// A target count disagreeing with the row count is rejected.
#[test]
fn rejects_target_length_mismatch() {
    let (features, _) = one_feature();
    let targets = array![1.0, 2.0];
    let error = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// A negative penalty is rejected at build time.
#[test]
fn rejects_negative_alpha() {
    let error = SKRidge::<f64>::build().alpha(-1.0).build().unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter { name: "alpha", .. }
    ));
}

/// An empty penalty sequence is rejected at build time.
#[test]
fn rejects_empty_alpha_sequence() {
    let error = SKRidge::<f64>::build()
        .alpha(Vec::<f64>::new())
        .build()
        .unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter { name: "alpha", .. }
    ));
}

/// A non-positive tolerance is rejected at build time.
#[test]
fn rejects_zero_tolerance() {
    let error = SKRidge::<f64>::build().tolerance(0.0).build().unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "tolerance",
            ..
        }
    ));
}

/// A zero iteration budget is rejected at build time.
#[test]
fn rejects_zero_iteration_budget() {
    let error = SKRidge::<f64>::build()
        .maximum_number_of_iterations(0)
        .build()
        .unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "maximum_number_of_iterations",
            ..
        }
    ));
}

/// Prediction on a column count differing from the fitted design is rejected.
#[test]
fn rejects_prediction_shape_mismatch() {
    let (features, targets) = one_feature();
    let model = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    let wrong = array![[1.0, 2.0]];
    let error = model.predict(wrong.view()).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// The SVD solver rejects sparse input with a structured error.
#[test]
fn svd_rejects_sparse_input() {
    let (sparse, targets) = sparse_design();
    let estimator = default_builder()
        .solver(SKRidgeSolver::Svd)
        .build()
        .unwrap();
    let error = estimator.fit(sparse.view(), targets.view()).unwrap_err();
    assert!(matches!(
        error,
        SKError::UnsupportedRepresentation {
            representation: "csr",
            ..
        }
    ));
}

/// A small sparse identity design with matching targets.
fn sparse_design() -> (sprs::CsMat<f64>, Array1<f64>) {
    use sprs::TriMat;
    let mut triplets = TriMat::new((2, 2));
    triplets.add_triplet(0, 0, 1.0_f64);
    triplets.add_triplet(1, 1, 1.0_f64);
    (triplets.to_csr(), array![1.0, 2.0])
}
