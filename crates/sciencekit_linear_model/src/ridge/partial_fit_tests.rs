//! Incremental `partial_fit` tests (spec `ridge-regression`).

use ndarray::{Array1, Array2, Axis, s};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKSupervisedFit};
use sciencekit_math::SKBackendSelection;

use super::core_implementation::SKRidge;
use super::sparse_test_datasets::to_csr;
use super::streaming_test_datasets::{
    assert_close_generic, assert_close_scalar_generic, large_features, large_targets,
    small_features, small_targets,
};
use super::test_datasets::default_ridge;

/// Splitting the data and accumulating with `partial_fit` matches the fit on the
/// concatenated data.
#[test]
fn incremental_after_a_dense_fit_matches_the_dense_fit() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let dense = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    let mut model = default_ridge()
        .fit(features.slice(s![0..2, ..]), targets.slice(s![0..2]))
        .unwrap();
    model
        .partial_fit(features.slice(s![2.., ..]), targets.slice(s![2..]))
        .unwrap();
    assert_close_generic(
        &model.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-9,
    );
    assert_close_scalar_generic(model.intercept_for(0), dense.intercept_for(0), 1e-9);
    assert_eq!(model.row_count(), features.nrows() as u64);
}

/// Accumulating row by row converges to the dense fit.
#[test]
fn single_row_accumulation_matches_the_dense_fit() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let dense = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    let mut model = default_ridge()
        .fit(features.slice(s![0..1, ..]), targets.slice(s![0..1]))
        .unwrap();
    for (feature_row, target_row) in features.outer_iter().skip(1).zip(targets.iter().skip(1)) {
        let batch = feature_row.insert_axis(Axis(0));
        let batch_target = Array1::from_elem(1, *target_row);
        model.partial_fit(batch, batch_target.view()).unwrap();
    }
    assert_close_generic(
        &model.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-9,
    );
}

/// An empty batch leaves the model unchanged.
#[test]
fn empty_batch_is_a_noop() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let mut model = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    let before = model.coefficients().to_owned();
    let empty_features: Array2<f64> = Array2::zeros((0, 3));
    let empty_targets: Array1<f64> = Array1::zeros(0);
    model
        .partial_fit(empty_features.view(), empty_targets.view())
        .unwrap();
    assert_eq!(model.coefficients(), before.view());
}

/// A column count differing from the fitted design is rejected.
#[test]
fn partial_fit_rejects_a_column_mismatch() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let mut model = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    let wrong_features: Array2<f64> = Array2::zeros((1, 4));
    let wrong_targets: Array1<f64> = Array1::zeros(1);
    let error = model
        .partial_fit(wrong_features.view(), wrong_targets.view())
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// `partial_fit` returns the model for chaining across several batches.
#[test]
fn partial_fit_chains_several_batches() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let dense = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    let mut model = default_ridge()
        .fit(features.slice(s![0..1, ..]), targets.slice(s![0..1]))
        .unwrap();
    model
        .partial_fit(features.slice(s![1..3, ..]), targets.slice(s![1..3]))
        .unwrap()
        .partial_fit(features.slice(s![3.., ..]), targets.slice(s![3..]))
        .unwrap();
    assert_eq!(model.row_count(), features.nrows() as u64);
    assert_close_generic(
        &model.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-9,
    );
}

/// Sparse batches accumulate into the same centered Gram as dense ones.
#[test]
fn sparse_batches_accumulate_like_dense_batches() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let dense = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    let first = to_csr(&features.slice(s![0..3, ..]).to_owned());
    let rest = to_csr(&features.slice(s![3.., ..]).to_owned());
    let mut model = default_ridge()
        .fit(first.view(), targets.slice(s![0..3]))
        .unwrap();
    model
        .partial_fit(rest.view(), targets.slice(s![3..]))
        .unwrap();
    assert_close_generic(
        &model.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-9,
    );
}

/// The incremental update works on `f32` and `f64`.
#[test]
fn incremental_fits_both_supported_floats() {
    incremental_matches_dense_for::<f32>(1e-3);
    incremental_matches_dense_for::<f64>(1e-9);
}

/// Incremental accumulation matches the dense fit at scale.
#[test]
fn incremental_matches_the_dense_fit_on_large_data() {
    let features = large_features::<f64>();
    let targets = large_targets(&features);
    let dense = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    let split = 5_000;
    let mut model = default_ridge()
        .fit(
            features.slice(s![0..split, ..]),
            targets.slice(s![0..split]),
        )
        .unwrap();
    model
        .partial_fit(features.slice(s![split.., ..]), targets.slice(s![split..]))
        .unwrap();
    assert_close_generic(
        &model.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-6,
    );
    assert_close_scalar_generic(model.intercept_for(0), dense.intercept_for(0), 1e-6);
}

/// Fit a small design densely and incrementally and assert they agree, for any
/// float.
fn incremental_matches_dense_for<F: SKBackendSelection + std::fmt::Debug>(tolerance: f64) {
    let features = small_features::<F>();
    let targets = small_targets(&features);
    let estimator = SKRidge::<F>::build().build().unwrap();
    let dense = estimator.fit(features.view(), targets.view()).unwrap();
    let mut model = estimator
        .fit(features.slice(s![0..2, ..]), targets.slice(s![0..2]))
        .unwrap();
    model
        .partial_fit(features.slice(s![2.., ..]), targets.slice(s![2..]))
        .unwrap();
    assert_close_generic(
        &model.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        tolerance,
    );
    assert_close_scalar_generic(model.intercept_for(0), dense.intercept_for(0), tolerance);
}
