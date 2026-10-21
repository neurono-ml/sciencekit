//! Shared deterministic datasets and helpers for the coordinate-descent tests.
//!
//! Test-only companion (declared under `#[cfg(test)]`): a hand-checkable
//! orthogonal design with a known Lasso solution, a one-feature design, dense
//! and CSR forms and float-comparison helpers. All data is built by closed-form
//! formulas — no RNG, identical on every machine.

use ndarray::{Array1, Array2, Axis, array};
use sciencekit_common::{SKError, SKFloat, SKSupervisedBatch, SKSupervisedLazySource};
use sprs::{CsMat, TriMat};

/// The convergence tolerance the substrate tests solve to.
pub(crate) const TIGHT_TOLERANCE: f64 = 1e-10;

/// A three-column orthogonal design (a Hadamard-like matrix) whose columns have
/// mean zero and squared norm four, with targets `X·[2, −1, 0]`.
///
/// Because the columns are orthogonal and mean-zero, the Lasso solution is
/// exactly `wⱼ = S(βⱼ, α)` and the intercept is zero, so expected coefficients
/// are hand-checkable for any penalty.
pub(crate) fn orthogonal_design() -> (Array2<f64>, Array1<f64>) {
    let features = array![
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];
    let targets = array![1.0, 3.0, -3.0, -1.0];
    (features, targets)
}

/// A single feature with the exact line `y = 2x` through the origin.
pub(crate) fn one_feature() -> (Array2<f64>, Array1<f64>) {
    (array![[1.0], [2.0], [3.0]], array![2.0, 4.0, 6.0])
}

/// A one-feature design whose mean is zero and whose targets have a large
/// intercept, so the unpenalized intercept is invariant to the penalty.
pub(crate) fn zero_mean_feature() -> (Array2<f64>, Array1<f64>) {
    (array![[-2.0], [-1.0], [0.0], [1.0], [2.0]], array![-1.0, 2.0, 5.0, 8.0, 11.0])
}

/// Convert a dense design to an equivalent CSR matrix, dropping exact zeros.
pub(crate) fn to_csr(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplets = TriMat::new((dense.nrows(), dense.ncols()));
    dense.indexed_iter().for_each(|((row, column), &value)| {
        if value != 0.0 {
            triplets.add_triplet(row, column, value);
        }
    });
    triplets.to_csr()
}

/// The soft-threshold operator `S(z, γ) = sign(z)·max(|z| − γ, 0)`.
pub(crate) fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}

/// Assert two `f64` slices agree within a relative tolerance.
pub(crate) fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
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
pub(crate) fn assert_close_scalar(actual: f64, expected: f64, tolerance: f64) {
    let bound = tolerance * (1.0 + actual.abs().max(expected.abs()));
    assert!(
        (actual - expected).abs() <= bound,
        "values {actual} and {expected} differ beyond {bound}"
    );
}

/// An in-memory supervised source: each entry is one owned batch.
pub(crate) struct SupervisedVecSource<F> {
    batches: Vec<(Array2<F>, Array1<F>)>,
    next: usize,
}

impl<F> SupervisedVecSource<F> {
    /// A source that yields the given batches in order.
    pub(crate) fn new(batches: Vec<(Array2<F>, Array1<F>)>) -> Self {
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

/// Split a design and its targets into owned chunks of the given lengths, which
/// must sum to the row count.
pub(crate) fn chunk_pairs<F: SKFloat>(
    features: &Array2<F>,
    targets: &Array1<F>,
    lengths: &[usize],
) -> Vec<(Array2<F>, Array1<F>)> {
    assert_eq!(lengths.iter().sum::<usize>(), features.nrows());
    let mut start = 0;
    lengths
        .iter()
        .map(|&length| {
            let feature_chunk = features.slice(ndarray::s![start..start + length, ..]).to_owned();
            let target_chunk = targets.slice(ndarray::s![start..start + length]).to_owned();
            start += length;
            (feature_chunk, target_chunk)
        })
        .collect()
}

/// Stream a design in row chunks of the given lengths.
pub(crate) fn chunked_source<F: SKFloat>(
    features: &Array2<F>,
    targets: &Array1<F>,
    lengths: &[usize],
) -> SupervisedVecSource<F> {
    SupervisedVecSource::new(chunk_pairs(features, targets, lengths))
}

/// A large deterministic design (12 000 rows × 5 columns) from a modular closed
/// form.
pub(crate) fn large_features<F: SKFloat>() -> Array2<F> {
    Array2::from_shape_fn((12_000, 5), |(row, column)| {
        F::from(((row * 7 + column * 13) % 97) as f64 * 0.25 + column as f64).expect("fits")
    })
}

/// The exact targets for [`large_features`] with an L1-sparse coefficient set.
pub(crate) fn large_targets<F: SKFloat>(features: &Array2<F>) -> Array1<F> {
    let coefficients = [1.5, 0.0, -0.75, 0.0, 2.0];
    features.map_axis(Axis(1), |row| {
        let linear = row
            .iter()
            .zip(coefficients.iter())
            .map(|(value, coefficient)| *value * F::from(*coefficient).expect("fits"))
            .fold(F::zero(), |accumulator, term| accumulator + term);
        linear + F::from(0.25).expect("fits")
    })
}
