//! Shared sparse streaming fixtures for the imputer conformance suite.

use ndarray::Array2;
use sciencekit_common::{SKDataBatch, SKError, SKLazySource};
use sprs::{CsMat, TriMat};

/// Convert a dense matrix to its equivalent CSR, dropping exact zeros.
pub(crate) fn dense_to_csr(features: &Array2<f64>) -> CsMat<f64> {
    let mut triplets = TriMat::new((features.nrows(), features.ncols()));
    features.indexed_iter().for_each(|((row, column), &value)| {
        if value != 0.0 {
            triplets.add_triplet(row, column, value);
        }
    });
    triplets.to_csr()
}

/// Split a dense matrix into CSR row chunks of the given lengths.
pub(crate) fn chunk_csr(features: &Array2<f64>, lengths: &[usize]) -> Vec<CsMat<f64>> {
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

/// An in-memory CSR batch source: each entry is one owned sparse batch.
pub(crate) struct SparseVecSource {
    batches: Vec<CsMat<f64>>,
    next: usize,
}

impl SparseVecSource {
    /// A source that yields the given batches in order.
    pub(crate) fn new(batches: Vec<CsMat<f64>>) -> Self {
        Self { batches, next: 0 }
    }
}

impl SKLazySource<f64> for SparseVecSource {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<f64>>, Self::Error> {
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

/// A one-nonzero-per-row CSR design whose dense twin exceeds any test budget.
pub(crate) fn guard_matrix(rows: usize, columns: usize) -> CsMat<f64> {
    let mut triplets = TriMat::new((rows, columns));
    for row in 0..rows {
        triplets.add_triplet(row, (row * 7) % columns, 1.0 + (row % 5) as f64);
    }
    triplets.to_csr()
}

/// Assert two float slices agree within a caller-chosen relative tolerance.
pub(crate) fn assert_close_sparse(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (value, target) in actual.iter().zip(expected.iter()) {
        let bound = tolerance * (1.0 + value.abs().max(target.abs()));
        assert!(
            (value - target).abs() <= bound,
            "values {value} and {target} differ beyond {bound}"
        );
    }
}
