//! Shared sparse supervised streaming fixtures for the linear-model conformance
//! suites.
//!
//! Batches are built from deterministic dense designs so the same data can be
//! streamed both ways. The guard matrix has one nonzero per row and a shape
//! whose dense twin is impractical for the test budget; that only proves the
//! stream completes at that shape, not that no dense copy exists.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKError, SKFloat, SKSupervisedBatch, SKSupervisedLazySource};
use sprs::{CsMat, TriMat};

/// Convert a dense design to its equivalent CSR, dropping exact zeros.
pub(crate) fn dense_to_csr<F: SKFloat>(features: &Array2<F>) -> CsMat<F> {
    let mut triplets = TriMat::new((features.nrows(), features.ncols()));
    features.indexed_iter().for_each(|((row, column), &value)| {
        if value != F::zero() {
            triplets.add_triplet(row, column, value);
        }
    });
    triplets.to_csr()
}

/// Split a design and its targets into CSR row chunks of the given lengths.
pub(crate) fn chunk_pairs_csr<F: SKFloat>(
    features: &Array2<F>,
    targets: &Array1<F>,
    lengths: &[usize],
) -> Vec<(CsMat<F>, Array1<F>)> {
    assert_eq!(
        lengths.iter().sum::<usize>(),
        features.nrows(),
        "chunk lengths must cover every row exactly once"
    );
    let mut start = 0;
    lengths
        .iter()
        .map(|&len| {
            let feature_chunk = features
                .slice(ndarray::s![start..start + len, ..])
                .to_owned();
            let target_chunk = targets.slice(ndarray::s![start..start + len]).to_owned();
            start += len;
            (dense_to_csr(&feature_chunk), target_chunk)
        })
        .collect()
}

/// An in-memory sparse supervised source: each entry is one owned batch.
pub(crate) struct SparseSupervisedSource<F> {
    batches: Vec<(CsMat<F>, Array1<F>)>,
    next: usize,
}

impl<F> SparseSupervisedSource<F> {
    /// A source that yields the given batches in order.
    pub(crate) fn new(batches: Vec<(CsMat<F>, Array1<F>)>) -> Self {
        Self { batches, next: 0 }
    }
}

impl<F: SKFloat> SKSupervisedLazySource<F> for SparseSupervisedSource<F> {
    type Error = SKError;

    fn next_supervised_batch(&mut self) -> Result<Option<SKSupervisedBatch<F>>, Self::Error> {
        if self.next >= self.batches.len() {
            return Ok(None);
        }
        let position = self.next;
        self.next += 1;
        let (features, targets) = self.batches[position].clone();
        let is_final = self.next == self.batches.len();
        Ok(Some(
            SKSupervisedBatch::new_sparse(features, targets, position, is_final)
                .expect("targets pair with the block rows"),
        ))
    }
}

/// An in-memory source of already-built supervised batches, dense or sparse.
pub(crate) struct OwnedSupervisedBatchSource<F> {
    batches: Vec<SKSupervisedBatch<F>>,
    next: usize,
}

impl<F> OwnedSupervisedBatchSource<F> {
    /// A source that yields the given batches in order.
    pub(crate) fn new(batches: Vec<SKSupervisedBatch<F>>) -> Self {
        Self { batches, next: 0 }
    }
}

impl<F: SKFloat> SKSupervisedLazySource<F> for OwnedSupervisedBatchSource<F> {
    type Error = SKError;

    fn next_supervised_batch(&mut self) -> Result<Option<SKSupervisedBatch<F>>, Self::Error> {
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

/// Deterministic targets for [`guard_matrix`].
pub(crate) fn guard_targets(rows: usize) -> Array1<f64> {
    Array1::from_shape_fn(rows, |row| 1.0 + (row % 3) as f64)
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
