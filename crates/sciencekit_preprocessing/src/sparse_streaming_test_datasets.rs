//! Shared sparse streaming test data (test-only companion).
//!
//! The sparse conformance suites build their batches from deterministic dense
//! matrices so the same data can be streamed both ways: dense blocks and CSR
//! blocks. A guard matrix builds a CSR design whose dense twin is larger than
//! the test budget allows; that only proves the stream completes at that
//! shape, not that no dense copy exists.

use ndarray::Array2;
use sciencekit_common::{SKDataBatch, SKError, SKFloat, SKLazySource};
use sprs::{CsMat, TriMat};

/// Convert a dense matrix to its equivalent CSR, dropping exact zeros.
pub(crate) fn dense_to_csr<F: SKFloat>(features: &Array2<F>) -> CsMat<F> {
    let mut triplets = TriMat::new((features.nrows(), features.ncols()));
    features.indexed_iter().for_each(|((row, column), &value)| {
        if value != F::zero() {
            triplets.add_triplet(row, column, value);
        }
    });
    triplets.to_csr()
}

/// Split a dense matrix into CSR row chunks of the given lengths.
pub(crate) fn chunk_csr<F: SKFloat>(features: &Array2<F>, lengths: &[usize]) -> Vec<CsMat<F>> {
    assert_eq!(
        lengths.iter().sum::<usize>(),
        features.nrows(),
        "chunk lengths must cover every row exactly once"
    );
    let mut start = 0;
    lengths
        .iter()
        .map(|&len| {
            let chunk = features
                .slice(ndarray::s![start..start + len, ..])
                .to_owned();
            start += len;
            dense_to_csr(&chunk)
        })
        .collect()
}

/// A densely stored design with a majority of exact zeros.
pub(crate) fn zeros_rich_features() -> Array2<f64> {
    ndarray::array![
        [1.0, 0.0, 0.0, 4.0],
        [0.0, 2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0],
        [0.0, 4.0, 0.0, 8.0],
        [6.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 7.0, 0.0],
        [0.0, 8.0, 0.0, 0.0],
    ]
}

/// A larger zeros-rich design (a majority of exact zeros) from a modular
/// closed form, used where the online quantile sketch needs a stable sample.
pub(crate) fn zeros_rich_large_features(rows: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, 4), |(row, column)| {
        if (row + column) % 3 == 0 {
            ((row * 7 + column * 13) % 97) as f64 * 0.25 + column as f64
        } else {
            0.0
        }
    })
}

/// An in-memory CSR batch source: each entry is one owned sparse batch.
pub(crate) struct SparseVecSource<F> {
    batches: Vec<CsMat<F>>,
    next: usize,
}

impl<F> SparseVecSource<F> {
    /// A source that yields the given batches in order.
    pub(crate) fn new(batches: Vec<CsMat<F>>) -> Self {
        Self { batches, next: 0 }
    }
}

impl<F: SKFloat> SKLazySource<F> for SparseVecSource<F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        if self.next >= self.batches.len() {
            return Ok(None);
        }
        let position = self.next;
        self.next += 1;
        let is_final = self.next == self.batches.len();
        Ok(Some(SKDataBatch::new_sparse(
            self.batches[position].clone(),
            position,
            is_final,
        )))
    }
}

/// An in-memory source of already-built batches, dense or sparse, in order.
pub(crate) struct OwnedBatchSource<F> {
    batches: Vec<SKDataBatch<F>>,
    next: usize,
}

impl<F> OwnedBatchSource<F> {
    /// A source that yields the given batches in order.
    pub(crate) fn new(batches: Vec<SKDataBatch<F>>) -> Self {
        Self { batches, next: 0 }
    }
}

impl<F: SKFloat> SKLazySource<F> for OwnedBatchSource<F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        if self.next >= self.batches.len() {
            return Ok(None);
        }
        let batch = self.batches[self.next].clone();
        self.next += 1;
        Ok(Some(batch))
    }
}

/// A one-nonzero-per-row CSR design whose dense twin is impractical for the
/// test budget.
pub(crate) fn guard_matrix(rows: usize, columns: usize) -> CsMat<f64> {
    let mut triplets = TriMat::new((rows, columns));
    for row in 0..rows {
        triplets.add_triplet(row, (row * 7) % columns, 1.0 + (row % 5) as f64);
    }
    triplets.to_csr()
}

/// Assert two float slices agree within a caller-chosen relative tolerance.
pub(crate) fn assert_close_sparse<F: SKFloat>(actual: &[F], expected: &[F], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (value, target) in actual.iter().zip(expected.iter()) {
        let value = value.to_f64().expect("scalars convert to f64");
        let target = target.to_f64().expect("scalars convert to f64");
        let bound = tolerance * (1.0 + value.abs().max(target.abs()));
        assert!(
            (value - target).abs() <= bound,
            "values {value} and {target} differ beyond {bound}"
        );
    }
}
