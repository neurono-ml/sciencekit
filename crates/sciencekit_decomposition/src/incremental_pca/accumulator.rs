//! Running `(samples_seen, mean, scatter)` state for incremental PCA (spec
//! `incremental-pca`, decision 1).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::SKFloat;

/// Running sample count, feature mean and centered scatter matrix
/// `S = Σ (x − mean)(x − mean)ᵀ` across processed batches.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(super) struct SKCovarianceAccumulator<F: SKFloat> {
    pub(super) samples_seen: usize,
    pub(super) mean: Array1<F>,
    pub(super) scatter: Array2<F>,
}

#[allow(dead_code)]
impl<F: SKFloat> SKCovarianceAccumulator<F> {
    /// Initializes an empty accumulator for `number_of_features` columns.
    pub(super) fn new(number_of_features: usize) -> Self {
        Self {
            samples_seen: 0,
            mean: Array1::zeros(number_of_features),
            scatter: Array2::zeros((number_of_features, number_of_features)),
        }
    }

    /// Total number of rows merged into the accumulator so far.
    pub(super) fn samples_seen(&self) -> usize {
        self.samples_seen
    }

    /// Number of features tracked by the accumulator.
    pub(super) fn number_of_features(&self) -> usize {
        self.mean.len()
    }

    /// Running empirical feature mean.
    pub(super) fn mean(&self) -> ArrayView1<'_, F> {
        self.mean.view()
    }

    /// Running centered scatter matrix of shape `d × d`.
    pub(super) fn scatter(&self) -> ArrayView2<'_, F> {
        self.scatter.view()
    }
}
