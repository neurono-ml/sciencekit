//! Solver agreement and resolution tests for `SKRidge` (spec
//! `ridge-regression`).

use sciencekit_common::SKSupervisedFit;
use sciencekit_common::builders::SKBuilder;

use super::builder::SKRidgeSolver;
use super::core_implementation::SKRidge;
use super::test_datasets::{
    assert_close, assert_close_scalar, cholesky_ridge, closed_form_no_intercept, default_builder,
    rank_deficient, two_feature,
};

/// The Cholesky path reproduces the explicit `(XᵀX + αI)⁻¹Xᵀy` solution.
#[test]
fn cholesky_matches_the_closed_form() {
    let (features, targets) = two_feature();
    let alpha = 1.5;
    let estimator = SKRidge::build()
        .alpha(alpha)
        .include_intercept(false)
        .solver(SKRidgeSolver::Cholesky)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    let expected = closed_form_no_intercept(&features, &targets, alpha);
    assert_close(&model.coefficient_for(0).to_vec(), &expected, 1e-9);
}

/// The SVD path agrees with the Cholesky path on a well-conditioned design.
#[test]
fn svd_agrees_with_cholesky() {
    let (features, targets) = two_feature();
    let cholesky = cholesky_ridge(0.5)
        .fit(features.view(), targets.view())
        .unwrap();
    let svd = default_builder()
        .alpha(0.5)
        .solver(SKRidgeSolver::Svd)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_close(
        &cholesky.coefficient_for(0).to_vec(),
        &svd.coefficient_for(0).to_vec(),
        1e-9,
    );
    assert_close_scalar(cholesky.intercept_for(0), svd.intercept_for(0), 1e-9);
    assert_eq!(cholesky.solver(), SKRidgeSolver::Cholesky);
    assert_eq!(svd.solver(), SKRidgeSolver::Svd);
}

/// A rank-deficient design with a zero penalty still yields a reduced rank.
#[test]
fn rank_deficient_svd_reports_reduced_rank() {
    let (features, targets) = rank_deficient();
    let model = default_builder()
        .alpha(0.0)
        .solver(SKRidgeSolver::Svd)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(model.feature_count(), 2);
    assert!(model.rank() < model.feature_count());
    assert_eq!(model.solver(), SKRidgeSolver::Svd);
    assert!(!model.singular_values().is_empty());
}

/// `Automatic` falls back from Cholesky to SVD when the penalty is zero.
#[test]
fn automatic_falls_back_to_svd_for_zero_penalty() {
    let (features, targets) = rank_deficient();
    let model = default_builder()
        .alpha(0.0)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(model.solver(), SKRidgeSolver::Svd);
    assert!(model.rank() < model.feature_count());
}

/// The Cholesky path reports full rank and no singular values.
#[test]
fn cholesky_surfaces_no_singular_values() {
    let (features, targets) = two_feature();
    let model = cholesky_ridge(1.0)
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(model.solver(), SKRidgeSolver::Cholesky);
    assert!(model.singular_values().is_empty());
    assert_eq!(model.rank(), model.feature_count());
}
