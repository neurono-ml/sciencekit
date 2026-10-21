//! Incremental `partial_fit` tests (spec `linear-regression`).

use ndarray::{Array1, Array2, array, s};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKSupervisedFit};

use super::core_implementation::SKLinearRegression;
use super::test_datasets::{
    assert_close_scalar, assert_close_with_tolerance, default_regressor, large_features,
    large_targets, small_features, small_targets,
};

/// Splitting the data into batches and accumulating with `partial_fit` matches
/// a single fit on the concatenated data.
#[test]
fn incremental_matches_the_concatenated_fit() {
    let features = small_features();
    let targets = small_targets(&features);
    let dense = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();

    let first_features = features.slice(s![0..2, ..]).to_owned();
    let first_targets = targets.slice(s![0..2]).to_owned();
    let mut model = default_regressor()
        .fit(first_features.view(), first_targets.view())
        .unwrap();

    let rest_features = features.slice(s![2.., ..]).to_owned();
    let rest_targets = targets.slice(s![2..]).to_owned();
    model
        .partial_fit(rest_features.view(), rest_targets.view())
        .unwrap();

    assert_close_with_tolerance(
        model.coefficients(),
        dense.coefficients().as_slice().unwrap(),
        1e-9,
    );
    assert_close_scalar(model.intercept(), dense.intercept(), 1e-9);
}

/// Accumulating row by row converges to the dense fit.
#[test]
fn row_by_row_accumulation_matches_the_dense_fit() {
    let features = small_features();
    let targets = small_targets(&features);
    let dense = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();

    let first_features = features.slice(s![0..1, ..]).to_owned();
    let first_targets = targets.slice(s![0..1]).to_owned();
    let mut model = default_regressor()
        .fit(first_features.view(), first_targets.view())
        .unwrap();
    for row in 1..features.nrows() {
        let row_features = features.slice(s![row..row + 1, ..]).to_owned();
        let row_targets = targets.slice(s![row..row + 1]).to_owned();
        model
            .partial_fit(row_features.view(), row_targets.view())
            .unwrap();
    }
    assert_close_with_tolerance(
        model.coefficients(),
        dense.coefficients().as_slice().unwrap(),
        1e-9,
    );
    assert_eq!(model.row_count(), features.nrows() as u64);
}

/// An empty batch leaves the model unchanged.
#[test]
fn empty_batch_is_a_noop() {
    let features = small_features();
    let targets = small_targets(&features);
    let mut model = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();
    let before = model.coefficients().clone();
    let empty_features: Array2<f64> = Array2::zeros((0, 2));
    let empty_targets: Array1<f64> = Array1::zeros(0);
    model
        .partial_fit(empty_features.view(), empty_targets.view())
        .unwrap();
    assert_eq!(model.coefficients(), &before);
}

/// A column count differing from the fitted design is rejected.
#[test]
fn partial_fit_rejects_a_column_mismatch() {
    let features = small_features();
    let targets = small_targets(&features);
    let mut model = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();
    let wrong_features: Array2<f64> = Array2::zeros((1, 3));
    let wrong_targets = array![1.0];
    let error = model
        .partial_fit(wrong_features.view(), wrong_targets.view())
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// `partial_fit` returns the model for chaining.
#[test]
fn partial_fit_chains() {
    let features = small_features();
    let targets = small_targets(&features);
    let first_features = features.slice(s![0..2, ..]).to_owned();
    let first_targets = targets.slice(s![0..2]).to_owned();
    let second_features = features.slice(s![2..3, ..]).to_owned();
    let second_targets = targets.slice(s![2..3]).to_owned();
    let third_features = features.slice(s![3.., ..]).to_owned();
    let third_targets = targets.slice(s![3..]).to_owned();

    let mut model = default_regressor()
        .fit(first_features.view(), first_targets.view())
        .unwrap();
    model
        .partial_fit(second_features.view(), second_targets.view())
        .unwrap()
        .partial_fit(third_features.view(), third_targets.view())
        .unwrap();
    assert_eq!(model.row_count(), features.nrows() as u64);
}

/// The incremental update honors a disabled intercept.
#[test]
fn partial_fit_honors_a_disabled_intercept() {
    let features = small_features();
    let targets = small_targets(&features);
    let estimator = SKLinearRegression::build()
        .include_intercept(false)
        .build()
        .unwrap();
    let first_features = features.slice(s![0..2, ..]).to_owned();
    let first_targets = targets.slice(s![0..2]).to_owned();
    let mut model = estimator
        .fit(first_features.view(), first_targets.view())
        .unwrap();
    let rest_features = features.slice(s![2.., ..]).to_owned();
    let rest_targets = targets.slice(s![2..]).to_owned();
    model
        .partial_fit(rest_features.view(), rest_targets.view())
        .unwrap();
    assert_eq!(model.intercept(), 0.0);
}

/// Incremental accumulation matches the dense fit at scale.
#[test]
fn incremental_matches_the_dense_fit_on_large_data() {
    let features = large_features();
    let targets = large_targets(&features);
    let dense = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();

    let split = 8_000;
    let first_features = features.slice(s![0..split, ..]).to_owned();
    let first_targets = targets.slice(s![0..split]).to_owned();
    let mut model = default_regressor()
        .fit(first_features.view(), first_targets.view())
        .unwrap();
    let rest_features = features.slice(s![split.., ..]).to_owned();
    let rest_targets = targets.slice(s![split..]).to_owned();
    model
        .partial_fit(rest_features.view(), rest_targets.view())
        .unwrap();

    assert_close_with_tolerance(
        model.coefficients(),
        dense.coefficients().as_slice().unwrap(),
        1e-6,
    );
    assert_close_scalar(model.intercept(), dense.intercept(), 1e-6);
}
