//! Shared fixtures of the acceptance tests.

use ndarray::{Array2, ArrayView1, s};
use sciencekit_common::{SKDataBatch, SKError, SKFloat, SKLazySource};

use crate::missing_indicator::SKIndicatorOutput;
use crate::simple_imputer::SKImputedOutput;

/// A finite source replaying in-memory batches, mirroring the scaler fixture.
pub(super) struct SKVectorSource<F> {
    batches: Vec<Array2<F>>,
    next: usize,
}

impl<F> SKVectorSource<F> {
    pub(super) fn new(batches: Vec<Array2<F>>) -> Self {
        Self { batches, next: 0 }
    }
}

impl<F: SKFloat> SKLazySource<F> for SKVectorSource<F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        if self.next >= self.batches.len() {
            return Ok(None);
        }
        let position = self.next;
        self.next += 1;
        let is_final = self.next == self.batches.len();
        Ok(Some(SKDataBatch::new(
            self.batches[position].clone(),
            position,
            is_final,
        )))
    }
}

/// Split a matrix into row chunks of the given length.
pub(super) fn chunk_rows<F: SKFloat>(
    features: &Array2<F>,
    rows_per_chunk: usize,
) -> Vec<Array2<F>> {
    let mut batches = Vec::new();
    let mut start = 0;
    while start < features.nrows() {
        let end = (start + rows_per_chunk).min(features.nrows());
        batches.push(features.slice(s![start..end, ..]).to_owned());
        start = end;
    }
    batches
}

/// A deterministic matrix: every cell whose patterned index falls under
/// `missing_percent` is missing, the rest carries a positive value.
pub(super) fn generated_features(
    rows_count: usize,
    columns_count: usize,
    missing_percent: usize,
) -> Array2<f64> {
    Array2::from_shape_fn((rows_count, columns_count), |(row, column)| {
        let pattern = (row * 31 + column * 17 + column * column) % 100;
        if pattern < missing_percent {
            f64::NAN
        } else {
            ((row * 7 + column * 13) % 1000) as f64 / 10.0 + 1.0
        }
    })
}

/// The dense payload of an imputation, panicking on a sparse one.
pub(super) fn dense_imputation(output: SKImputedOutput<f64>) -> Array2<f64> {
    match output {
        SKImputedOutput::Dense(matrix) => matrix,
        SKImputedOutput::Sparse(matrix) => panic!("expected a dense result, found {matrix:?}"),
    }
}

/// The dense payload of a mask, panicking on a sparse one.
pub(super) fn dense_mask(output: SKIndicatorOutput<f64>) -> Array2<f64> {
    match output {
        SKIndicatorOutput::Dense(matrix) => matrix,
        SKIndicatorOutput::Sparse(matrix) => panic!("expected a dense mask, found {matrix:?}"),
    }
}

/// Compare two statistics vectors, allowing the last-bit differences between
/// summation orders while requiring exact agreement on the absent statistics.
pub(super) fn assert_statistics_agree(actual: ArrayView1<'_, f64>, expected: ArrayView1<'_, f64>) {
    assert_eq!(actual.len(), expected.len());
    for (column, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "column {column} must have no statistic");
            continue;
        }
        let difference = (actual - expected).abs();
        assert!(
            difference <= 1e-9 * expected.abs().max(1.0),
            "column {column}: the streamed statistic {actual} drifted from the exact one {expected}"
        );
    }
}
