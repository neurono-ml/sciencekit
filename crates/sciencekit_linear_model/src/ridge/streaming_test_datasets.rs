//! Shared deterministic datasets and source helpers for the ridge streaming and
//! incremental tests.
//!
//! Test-only companion (declared under `#[cfg(test)]`): a small hand-checkable
//! design, a large generated one, a supervised in-memory source and generic
//! float-comparison helpers. All data is built by closed-form formulas — no RNG,
//! identical on every machine.

use ndarray::{Array1, Array2, Axis};
use sciencekit_common::{SKError, SKFloat, SKSupervisedBatch, SKSupervisedLazySource};

/// Small hand-checkable design (6 rows × 3 columns) from a modular closed form.
pub(super) fn small_features<F: SKFloat>() -> Array2<F> {
    Array2::from_shape_fn((6, 3), |(row, column)| {
        F::from(((row * 2 + column * 3) % 7) as f64 - 3.0).expect("values fit in the scalar")
    })
}

/// The exact targets for [`small_features`]: `2·x₀ − 1.5·x₁ + 0.5·x₂ + 1`.
pub(super) fn small_targets<F: SKFloat>(features: &Array2<F>) -> Array1<F> {
    let coefficients = [2.0, -1.5, 0.5];
    features.map_axis(Axis(1), |row| {
        let linear = row
            .iter()
            .zip(coefficients.iter())
            .map(|(value, coefficient)| *value * F::from(*coefficient).expect("fits"))
            .fold(F::zero(), |accumulator, term| accumulator + term);
        linear + F::one()
    })
}

/// Large deterministic design (12 000 rows × 5 columns) from a modular closed
/// form.
pub(super) fn large_features<F: SKFloat>() -> Array2<F> {
    Array2::from_shape_fn((12_000, 5), |(row, column)| {
        F::from(((row * 7 + column * 13) % 97) as f64 * 0.25 + column as f64).expect("fits")
    })
}

/// The exact targets for [`large_features`].
pub(super) fn large_targets<F: SKFloat>(features: &Array2<F>) -> Array1<F> {
    let coefficients = [1.5, -0.75, 2.0, 0.5, -1.25];
    features.map_axis(Axis(1), |row| {
        let linear = row
            .iter()
            .zip(coefficients.iter())
            .map(|(value, coefficient)| *value * F::from(*coefficient).expect("fits"))
            .fold(F::zero(), |accumulator, term| accumulator + term);
        linear + F::from(0.25).expect("fits")
    })
}

/// Split features and targets into owned chunks of the given lengths, which must
/// sum to the row count.
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
    /// A source that yields the given batches in order.
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

/// Assert two float slices agree within a caller-chosen relative tolerance.
pub(super) fn assert_close_generic<F: SKFloat + std::fmt::Debug>(
    actual: &[F],
    expected: &[F],
    tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());
    let tolerance = F::from(tolerance).expect("tolerance fits in the scalar");
    for (value, target) in actual.iter().zip(expected.iter()) {
        let bound = tolerance * (F::one() + value.abs().max(target.abs()));
        assert!(
            (*value - *target).abs() <= bound,
            "values {value:?} and {target:?} differ beyond {bound:?}"
        );
    }
}

/// Assert two scalars agree within a caller-chosen relative tolerance.
pub(super) fn assert_close_scalar_generic<F: SKFloat + std::fmt::Debug>(
    actual: F,
    expected: F,
    tolerance: f64,
) {
    let tolerance = F::from(tolerance).expect("tolerance fits in the scalar");
    let bound = tolerance * (F::one() + actual.abs().max(expected.abs()));
    assert!(
        (actual - expected).abs() <= bound,
        "values {actual:?} and {expected:?} differ beyond {bound:?}"
    );
}
