//! Shared deterministic datasets and helpers for the ridge tests.
//!
//! Test-only companion (declared under `#[cfg(test)]`): hand-checkable designs,
//! a closed-form ridge reference and float-comparison helpers. All data is built
//! by closed-form formulas — no RNG, identical on every machine.

use ndarray::{Array1, Array2, array};
use sciencekit_common::builders::SKBuilder;

use super::builder::{SKRidgeBuilder, SKRidgeSolver};
use super::core_implementation::SKRidge;

/// The default-configured ridge estimator (`alpha = [1.0]`, intercept on).
pub(super) fn default_ridge() -> SKRidge<f64> {
    SKRidge::build().build().expect("default ridge builds")
}

/// A fresh builder with the documented defaults for `f64`.
pub(super) fn default_builder() -> SKRidgeBuilder<f64> {
    SKRidge::build()
}

/// One-feature design with the exact line `y = 2x` through the origin.
pub(super) fn one_feature() -> (Array2<f64>, Array1<f64>) {
    (array![[1.0], [2.0], [3.0]], array![2.0, 4.0, 6.0])
}

/// Two-feature design used for the closed-form comparison.
pub(super) fn two_feature() -> (Array2<f64>, Array1<f64>) {
    (
        array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        array![1.0, 2.0, 3.0, 5.0],
    )
}

/// A design whose first feature is exactly twice its second (rank-deficient).
pub(super) fn rank_deficient() -> (Array2<f64>, Array1<f64>) {
    (
        array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]],
        array![1.0, 2.0, 3.0, 4.0],
    )
}

/// A one-feature line with a large intercept and a zero-mean feature, so the
/// unpenalized intercept is invariant to `alpha`.
pub(super) fn zero_mean_feature() -> (Array2<f64>, Array1<f64>) {
    let features = array![[-2.0], [-1.0], [0.0], [1.0], [2.0]];
    let targets = array![-1.0, 2.0, 5.0, 8.0, 11.0];
    (features, targets)
}

/// The closed-form ridge solution `(XᵀX + αI)⁻¹ Xᵀy` for a two-feature design
/// without an intercept, using the explicit 2×2 inverse.
pub(super) fn closed_form_no_intercept(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    alpha: f64,
) -> [f64; 2] {
    let gram = features.t().dot(features);
    let cross = features.t().dot(targets);
    let a = gram[(0, 0)] + alpha;
    let b = gram[(0, 1)];
    let d = gram[(1, 1)] + alpha;
    let determinant = a * d - b * b;
    let inverse_00 = d / determinant;
    let inverse_01 = -b / determinant;
    let inverse_11 = a / determinant;
    [
        inverse_00 * cross[0] + inverse_01 * cross[1],
        inverse_01 * cross[0] + inverse_11 * cross[1],
    ]
}

/// Assert two `f64` slices agree within a relative tolerance.
pub(super) fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (value, target) in actual.iter().zip(expected.iter()) {
        let bound = tolerance * (1.0 + value.abs().max(target.abs()));
        assert!(
            (value - target).abs() <= bound,
            "values {value} and {target} differ beyond {bound}"
        );
    }
}

/// Assert two scalars agree within a relative tolerance.
pub(super) fn assert_close_scalar(actual: f64, expected: f64, tolerance: f64) {
    let bound = tolerance * (1.0 + actual.abs().max(expected.abs()));
    assert!(
        (actual - expected).abs() <= bound,
        "values {actual} and {expected} differ beyond {bound}"
    );
}

/// The Cholesky-backed estimator used by the solver-agreement tests.
pub(super) fn cholesky_ridge(alpha: f64) -> SKRidge<f64> {
    default_builder()
        .alpha(alpha)
        .solver(SKRidgeSolver::Cholesky)
        .build()
        .expect("cholesky ridge builds")
}
