//! Incremental `partial_fit` tests for `SKLasso` (spec `lasso`).

use crate::coordinate_descent::test_datasets::{assert_close, orthogonal_design, to_csr};
use ndarray::{Array1, Array2, s};
use sciencekit_common::{SKError, SKSupervisedFit};

use super::streaming_tests::lasso;

/// `partial_fit` accumulates batches to the same fit as one dense fit.
#[test]
fn partial_fit_split_batches_equal_the_whole() {
    let (features, targets) = orthogonal_design();
    let dense = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let mut model = lasso(0.5)
        .fit(features.slice(s![0..1, ..]), targets.slice(s![0..1]))
        .unwrap();
    model
        .partial_fit(features.slice(s![1.., ..]), targets.slice(s![1..]))
        .unwrap();
    assert_close(
        &model.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-8,
    );
    assert_eq!(model.row_count(), 4);
}

/// An empty incremental batch is a no-op.
#[test]
fn partial_fit_empty_batch_is_a_noop() {
    let (features, targets) = orthogonal_design();
    let mut model = lasso(0.5).fit(features.view(), targets.view()).unwrap();
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
    let (features, targets) = orthogonal_design();
    let mut model = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let wrong_features: Array2<f64> = Array2::zeros((1, 4));
    let wrong_targets: Array1<f64> = Array1::zeros(1);
    let error = model
        .partial_fit(wrong_features.view(), wrong_targets.view())
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// A sparse incremental batch matches the dense accumulation.
#[test]
fn partial_fit_sparse_batch_matches_dense() {
    let (features, targets) = orthogonal_design();
    let dense = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let first = to_csr(&features.slice(s![0..2, ..]).to_owned());
    let rest = to_csr(&features.slice(s![2.., ..]).to_owned());
    let mut model = lasso(0.5)
        .fit(first.view(), targets.slice(s![0..2]))
        .unwrap();
    model
        .partial_fit(rest.view(), targets.slice(s![2..]))
        .unwrap();
    assert_close(
        &model.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-8,
    );
}
