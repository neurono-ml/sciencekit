//! Subsample views (dense now, sparse CSC in Task 2.3) for isolation tree growth.

#![cfg_attr(not(test), allow(dead_code))]

use ndarray::ArrayView2;
use sciencekit_common::SKFloat;

/// Representation-neutral subsample interface for growing an isolation tree.
pub(crate) trait IsolationSampleSource<F: SKFloat> {
    /// Number of sampled rows in this subsample.
    fn sample_count(&self) -> usize;

    /// Number of columns in the feature space.
    fn feature_count(&self) -> usize;

    /// Observed `(minimum, maximum)` of `feature_index` across `rows` (which is non-empty).
    fn feature_bounds(&self, rows: &[usize], feature_index: usize) -> (F, F);

    /// Splits `rows` into `(left, right)` using `value <= threshold` for left.
    fn partition_rows(
        &self,
        rows: &[usize],
        feature_index: usize,
        threshold: F,
    ) -> (Vec<usize>, Vec<usize>);
}

/// Borrowed dense subsample matrix.
pub(crate) struct DenseSubsample<'a, F> {
    matrix: ArrayView2<'a, F>,
}

impl<'a, F: SKFloat> DenseSubsample<'a, F> {
    /// Wraps a dense subsample view.
    pub(crate) fn new(matrix: ArrayView2<'a, F>) -> Self {
        Self { matrix }
    }
}

impl<F: SKFloat> IsolationSampleSource<F> for DenseSubsample<'_, F> {
    fn sample_count(&self) -> usize {
        self.matrix.nrows()
    }

    fn feature_count(&self) -> usize {
        self.matrix.ncols()
    }

    fn feature_bounds(&self, rows: &[usize], feature_index: usize) -> (F, F) {
        let first_value = self.matrix[[rows[0], feature_index]];
        rows.iter().skip(1).fold(
            (first_value, first_value),
            |(current_minimum, current_maximum), &row_index| {
                let value = self.matrix[[row_index, feature_index]];
                let next_minimum = if value < current_minimum {
                    value
                } else {
                    current_minimum
                };
                let next_maximum = if value > current_maximum {
                    value
                } else {
                    current_maximum
                };
                (next_minimum, next_maximum)
            },
        )
    }

    fn partition_rows(
        &self,
        rows: &[usize],
        feature_index: usize,
        threshold: F,
    ) -> (Vec<usize>, Vec<usize>) {
        rows.iter()
            .copied()
            .partition(|&row_index| self.matrix[[row_index, feature_index]] <= threshold)
    }
}
