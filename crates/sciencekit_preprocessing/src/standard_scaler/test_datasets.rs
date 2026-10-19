//! Shared deterministic datasets for the scaler tests, dense and streaming.
//!
//! Test-only companion (declared under `#[cfg(test)]`): a small hand-checkable
//! matrix and a large generated one, plus the row-chunking and batch-source
//! helpers every streaming test needs. All data is built by closed-form
//! formulas — no RNG, no fixtures on disk, identical on every machine.

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKDataBatch, SKError, SKFloat, SKLazySource};

use super::core_implementation::SKStandardScaler;
use sciencekit_common::builders::SKBuilder;

/// Small hand-checkable matrix (6 rows × 2 columns).
pub(super) fn small_features() -> Array2<f64> {
    array![
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ]
}

/// Large deterministic matrix (100 000 rows × 8 columns): varying entries from
/// a modular closed form, with the last column held constant to exercise the
/// zero-variance path at scale.
pub(super) fn large_features() -> Array2<f64> {
    Array2::from_shape_fn((100_000, 8), |(row, column)| {
        if column == 7 {
            5.0
        } else {
            ((row * 7 + column * 13) % 97) as f64 * 0.25 + column as f64
        }
    })
}

/// Split matrix rows into owned chunks of the given lengths, which must sum to
/// the row count.
pub(super) fn chunk_rows<F: SKFloat>(features: &Array2<F>, lengths: &[usize]) -> Vec<Array2<F>> {
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
            chunk
        })
        .collect()
}

/// An in-memory batch source: each vector entry is one owned batch.
pub(super) struct VecSource<F> {
    batches: Vec<Array2<F>>,
    next: usize,
}

impl<F> VecSource<F> {
    pub(super) fn new(batches: Vec<Array2<F>>) -> Self {
        Self { batches, next: 0 }
    }
}

impl<F: SKFloat> SKLazySource<F> for VecSource<F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        if self.next >= self.batches.len() {
            return Ok(None);
        }
        let position = self.next;
        self.next += 1;
        let data = self.batches[position].clone();
        let is_final = self.next == self.batches.len();
        Ok(Some(SKDataBatch::new(data, position, is_final)))
    }
}

/// Stream a matrix in row chunks of the given lengths.
pub(super) fn chunked_source<F: SKFloat>(features: &Array2<F>, lengths: &[usize]) -> VecSource<F> {
    VecSource::new(chunk_rows(features, lengths))
}

/// Assert two float vectors agree within relative tolerance.
pub(super) fn assert_close(actual: &Array1<f64>, expected: &[f64]) {
    assert_close_with_tolerance(actual, expected, 1e-12);
}

/// Assert two float vectors agree within a caller-chosen relative tolerance
/// (streaming merges reorder float operations, so cross-regime comparisons
/// use a looser bound than same-regime ones).
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
pub(super) fn default_scaler() -> SKStandardScaler {
    SKStandardScaler::build()
        .build()
        .expect("default scaler builds")
}
