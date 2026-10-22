//! Shared deterministic datasets and helpers for the logistic tests.
//!
//! Test-only companion (declared under `#[cfg(test)]`): hand-checkable
//! separable, three-class, imbalanced and non-separable designs, dense and CSR
//! forms, an in-memory streaming source and float-comparison helpers. All data
//! is built by closed-form formulas — no RNG, identical on every machine.

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKError, SKFloat, SKSupervisedBatch, SKSupervisedLazySource};
use sprs::{CsMat, TriMat};

/// A two-feature linearly separable binary design.
pub(crate) fn separable_binary_design() -> (Array2<f64>, Vec<usize>) {
    let features = array![
        [-2.0, -1.0],
        [-1.0, -2.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0]
    ];
    (features, vec![0, 0, 0, 1, 1, 1])
}

/// A two-feature separable binary design whose third column is an exact zero,
/// so an L1 fit must drive that coefficient to exactly zero.
pub(crate) fn separable_binary_design_with_zero_column() -> (Array2<f64>, Vec<usize>) {
    let features = array![
        [-2.0, -1.0, 0.0],
        [-1.0, -2.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
    ];
    (features, vec![0, 0, 0, 1, 1, 1])
}

/// A three-class design clustered along the axes.
pub(crate) fn three_class_design() -> (Array2<f64>, Vec<usize>) {
    let features = array![
        [-3.0, 0.0],
        [-2.0, 0.0],
        [0.0, 3.0],
        [0.0, 2.0],
        [3.0, 0.0],
        [2.0, 0.0]
    ];
    (features, vec![0, 0, 1, 1, 2, 2])
}

/// A single-feature design with a heavily imbalanced class split.
pub(crate) fn imbalanced_binary_design() -> (Array2<f64>, Vec<usize>) {
    (
        array![[-2.0], [-1.0], [0.5], [1.0], [2.0], [3.0]],
        vec![0, 0, 0, 0, 1, 1],
    )
}

/// A single-feature design whose best separator has a large intercept.
pub(crate) fn large_intercept_binary_design() -> (Array2<f64>, Vec<usize>) {
    (
        array![[-2.0], [-1.0], [0.0], [1.0], [2.0]],
        vec![0, 0, 1, 1, 1],
    )
}

/// An XOR-like, deliberately non-separable binary design, so the unregularised
/// maximum-likelihood estimate is finite.
pub(crate) fn non_separable_binary_design() -> (Array2<f64>, Vec<usize>) {
    let features = array![
        [-2.0, -2.0],
        [-2.0, 2.0],
        [2.0, -2.0],
        [2.0, 2.0],
        [-1.0, -1.0],
        [-1.0, 1.0]
    ];
    (features, vec![0, 1, 1, 0, 0, 1])
}

/// The labels of a designed dataset as an `i64` array (estimator tests).
pub(crate) fn labels_i64(labels: &[usize]) -> Array1<i64> {
    Array1::from_vec(labels.iter().map(|&label| label as i64).collect())
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

/// Assert two slices agree within a relative tolerance.
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
pub(crate) struct SupervisedFloatSource<F> {
    batches: Vec<(Array2<F>, Array1<F>)>,
    next: usize,
}

impl<F> SupervisedFloatSource<F> {
    /// A source that yields the given batches in order.
    pub(crate) fn new(batches: Vec<(Array2<F>, Array1<F>)>) -> Self {
        Self { batches, next: 0 }
    }
}

impl<F: SKFloat> SKSupervisedLazySource<F> for SupervisedFloatSource<F> {
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
