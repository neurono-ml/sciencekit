//! Sparse design tests for `SKLasso` (spec `lasso`).

use crate::coordinate_descent::test_datasets::{
    assert_close, assert_close_scalar, orthogonal_design, to_csr,
};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::SKSupervisedFit;

use super::core_implementation::SKLasso;

/// A tight lasso over `f64`.
fn lasso(alpha: f64) -> SKLasso<f64> {
    SKLasso::build()
        .alpha(alpha)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap()
}

/// A sparse fit matches the equivalent dense fit.
#[test]
fn sparse_matches_dense() {
    let (features, targets) = orthogonal_design();
    let sparse = to_csr(&features);
    let dense = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let sparse_model = lasso(0.5).fit(sparse.view(), targets.view()).unwrap();
    assert_close(
        &dense.coefficient_for(0).to_vec(),
        &sparse_model.coefficient_for(0).to_vec(),
        1e-8,
    );
    assert_close_scalar(
        dense.intercept_for(0),
        sparse_model.intercept_for(0),
        1e-8,
    );
}

/// A sparse feature excluded by the penalty is still exactly zero.
#[test]
fn sparse_irrelevant_feature_is_exactly_zero() {
    let (features, targets) = orthogonal_design();
    let sparse = to_csr(&features);
    let model = lasso(0.5).fit(sparse.view(), targets.view()).unwrap();
    assert_eq!(model.coefficient_for(0)[2], 0.0);
}

/// Sparse and dense prediction agree for the same model.
#[test]
fn sparse_prediction_matches_dense() {
    let (features, targets) = orthogonal_design();
    let sparse = to_csr(&features);
    let model = lasso(0.5).fit(sparse.view(), targets.view()).unwrap();
    let dense_prediction = model.predict_with_alpha_index(0, features.view()).unwrap();
    let sparse_prediction = model
        .predict_with_alpha_index(0, sparse.view())
        .unwrap();
    assert_close(&sparse_prediction.to_vec(), &dense_prediction.to_vec(), 1e-8);
}
