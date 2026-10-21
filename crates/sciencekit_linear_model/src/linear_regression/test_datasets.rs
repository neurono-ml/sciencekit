//! Shared deterministic datasets for the linear-regression tests, dense and
//! streaming.
//!
//! Test-only companion (declared under `#[cfg(test)]`): a small hand-checkable
//! design with an exact line and a large generated one, plus the row-chunking
//! and supervised-source helpers every streaming test needs. All data is built
//! by closed-form formulas — no RNG, no fixtures on disk, identical on every
//! machine.

use ndarray::{Array1, Array2, Axis, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKFloat, SKSupervisedBatch, SKSupervisedLazySource};

use super::core_implementation::SKLinearRegression;

/// Small hand-checkable design (4 rows × 2 columns).
pub(super) fn small_features() -> Array2<f64> {
    array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
}

/// The exact targets for [`small_features`]: `3·x₀ − 2·x₁ + 0.5`.
pub(super) fn small_targets(features: &Array2<f64>) -> Array1<f64> {
    features.map_axis(Axis(1), |row| 3.0 * row[0] - 2.0 * row[1] + 0.5)
}

/// The coefficients `small_targets` encodes.
pub(super) const SMALL_COEFFICIENTS: [f64; 2] = [3.0, -2.0];

/// The intercept `small_targets` encodes.
pub(super) const SMALL_INTERCEPT: f64 = 0.5;

/// Large deterministic design (20 000 rows × 6 columns) from a modular closed
/// form.
pub(super) fn large_features() -> Array2<f64> {
    Array2::from_shape_fn((20_000, 6), |(row, column)| {
        ((row * 7 + column * 13) % 89) as f64 * 0.25 + column as f64
    })
}

/// The exact targets for [`large_features`].
pub(super) fn large_targets(features: &Array2<f64>) -> Array1<f64> {
    const COEFFICIENTS: [f64; 6] = [1.5, -0.75, 2.0, 0.5, -1.25, 3.0];
    features.map_axis(Axis(1), |row| {
        row.iter()
            .zip(COEFFICIENTS.iter())
            .map(|(value, coefficient)| value * coefficient)
            .sum::<f64>()
            + 0.25
    })
}

/// Split features and targets into owned chunks of the given lengths, which
/// must sum to the row count.
pub(super) fn chunk_pairs<F: SKFloat>(
    features: &Array2<F>,
    targets: &Array1<F>,
    lengths: &[usize],
) -> Vec<(Array2<F>, Array1<F>)> {
    assert_eq!(
        lengths.iter().sum::<usize>(),
        features.nrows(),
        "chunk lengths must cover every row exactly once"
    );
    assert_eq!(targets.len(), features.nrows());
    let mut start = 0;
    lengths
        .iter()
        .map(|&len| {
            let feature_chunk = features
                .slice(ndarray::s![start..start + len, ..])
                .to_owned();
            let target_chunk = targets.slice(ndarray::s![start..start + len]).to_owned();
            start += len;
            (feature_chunk, target_chunk)
        })
        .collect()
}

/// An in-memory supervised source: each vector entry is one owned batch.
pub(super) struct SupervisedVecSource<F> {
    batches: Vec<(Array2<F>, Array1<F>)>,
    next: usize,
}

impl<F> SupervisedVecSource<F> {
    pub(super) fn new(batches: Vec<(Array2<F>, Array1<F>)>) -> Self {
        Self { batches, next: 0 }
    }
}

impl<F: SKFloat> SKSupervisedLazySource<F> for SupervisedVecSource<F> {
    type Error = SKError;

    fn next_supervised_batch(&mut self) -> Result<Option<SKSupervisedBatch<F>>, Self::Error> {
        if self.next >= self.batches.len() {
            return Ok(None);
        }
        let position = self.next;
        self.next += 1;
        let (features, targets) = self.batches[position].clone();
        let is_final = self.next == self.batches.len();
        Ok(Some(SKSupervisedBatch::new(
            features, targets, position, is_final,
        )))
    }
}

/// Stream a design in row chunks of the given lengths.
pub(super) fn chunked_source<F: SKFloat>(
    features: &Array2<F>,
    targets: &Array1<F>,
    lengths: &[usize],
) -> SupervisedVecSource<F> {
    SupervisedVecSource::new(chunk_pairs(features, targets, lengths))
}

/// Assert two float vectors agree within a caller-chosen relative tolerance
/// (cross-regime comparisons use a looser bound than same-regime ones).
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

/// Assert two scalars agree within a relative tolerance.
pub(super) fn assert_close_scalar(actual: f64, expected: f64, relative_tolerance: f64) {
    let tolerance = relative_tolerance * (1.0 + actual.abs().max(expected.abs()));
    assert!(
        (actual - expected).abs() <= tolerance,
        "values {actual} and {expected} differ beyond tolerance {tolerance}"
    );
}

/// The default-configured regressor every test starts from.
pub(super) fn default_regressor() -> SKLinearRegression {
    SKLinearRegression::build()
        .build()
        .expect("default regressor builds")
}
