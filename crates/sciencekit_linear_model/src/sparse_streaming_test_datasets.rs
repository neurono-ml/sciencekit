//! Shared sparse supervised streaming fixtures for the linear-model conformance
//! suites.
//!
//! Batches are built from deterministic dense designs so the same data can be
//! streamed both ways. The guard matrix has one nonzero per row and a shape
//! whose dense twin exceeds any test budget, so a hidden `n × p` copy fails by
//! exhaustion rather than by an assertion.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKError, SKSupervisedBatch, SKSupervisedLazySource};
use sprs::{CsMat, TriMat};

/// Convert a dense design to its equivalent CSR, dropping exact zeros.
pub(crate) fn dense_to_csr(features: &Array2<f64>) -> CsMat<f64> {
    let mut triplets = TriMat::new((features.nrows(), features.ncols()));
    features.indexed_iter().for_each(|((row, column), &value)| {
        if value != 0.0 {
            triplets.add_triplet(row, column, value);
        }
    });
    triplets.to_csr()
}

/// Split a design and its targets into CSR row chunks of the given lengths.
pub(crate) fn chunk_pairs_csr(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    lengths: &[usize],
) -> Vec<(CsMat<f64>, Array1<f64>)> {
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
pub(crate) struct SparseSupervisedSource {
    batches: Vec<(CsMat<f64>, Array1<f64>)>,
    next: usize,
}

impl SparseSupervisedSource {
    /// A source that yields the given batches in order.
    pub(crate) fn new(batches: Vec<(CsMat<f64>, Array1<f64>)>) -> Self {
        Self { batches, next: 0 }
    }
}

impl SKSupervisedLazySource<f64> for SparseSupervisedSource {
    type Error = SKError;

    fn next_supervised_batch(&mut self) -> Result<Option<SKSupervisedBatch<f64>>, Self::Error> {
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

/// A one-nonzero-per-row CSR design whose dense twin exceeds any test budget.
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
