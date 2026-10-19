//! Shared deterministic datasets for the robust scaler tests.
//!
//! Test-only companion (declared under `#[cfg(test)]`). Datasets are built by
//! closed-form formulas — no RNG, no fixtures on disk, identical on every
//! machine.

use ndarray::{Array1, Array2};
use sciencekit_common::builders::SKBuilder;

use super::core_implementation::SKRobustScaler;

/// Large deterministic matrix (100 000 rows × 8 columns) from a modular closed
/// form, with the last column held constant to exercise the zero-spread path at
/// scale.
pub(super) fn large_features() -> Array2<f64> {
    Array2::from_shape_fn((100_000, 8), |(row, column)| {
        if column == 7 {
            5.0
        } else {
            ((row * 7 + column * 13) % 97) as f64 * 0.25 + column as f64
        }
    })
}

/// Assert two float vectors agree within relative tolerance.
pub(super) fn assert_close(actual: &Array1<f64>, expected: &[f64]) {
    assert_close_with_tolerance(actual, expected, 1e-12);
}

/// Assert two float vectors agree within a caller-chosen relative tolerance
/// (the online sketch path reuses this bound).
pub(super) fn assert_close_with_tolerance(
    actual: &Array1<f64>,
    expected: &[f64],
    relative_tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (value, target) in actual.iter().zip(expected.iter()) {
        let tolerance = relative_tolerance * (1.0 + value.abs().max(target.abs()));
        assert!(
            (value - target).abs() <= tolerance,
            "values {value} and {target} differ beyond tolerance {tolerance}"
        );
    }
}

/// The default-configured scaler every test starts from.
pub(super) fn default_scaler() -> SKRobustScaler<f64> {
    SKRobustScaler::<f64>::build()
        .build()
        .expect("default scaler builds")
}
