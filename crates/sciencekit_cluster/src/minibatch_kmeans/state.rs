//! Mutable batch-accumulation state shared across `fit`, `partial_fit` and
//! `fit_streaming` (spec `mini-batch-k-means`, decisions 3–6).

use ndarray::{Array1, Array2};
use sciencekit_common::SKFloat;

use super::config::SKMiniBatchKMeansConfiguration;

/// Running centers, per-center sample counts and batch/convergence progress.
#[derive(Debug, Clone)]
pub(crate) struct SKMiniBatchState<F: SKFloat> {
    /// Dense center matrix of shape `(number_of_clusters, number_of_features)`.
    pub(crate) centers: Array2<F>,
    /// Cumulative number of rows assigned to each center.
    pub(crate) counts: Array1<usize>,
    /// Total mini-batches processed so far.
    pub(crate) batches_processed: usize,
    /// Whether positive-tolerance center movement met the convergence threshold.
    pub(crate) converged: bool,
    /// Sum of squared distances accumulated over the most recent pass or batch.
    pub(crate) last_batch_inertia: F,
    /// Validated estimator configuration governing updates and reassignment.
    pub(crate) configuration: SKMiniBatchKMeansConfiguration<F>,
}

impl<F: SKFloat> SKMiniBatchState<F> {
    /// Initializes a fresh state from initial centers with zero per-center counts.
    pub(crate) fn new(
        centers: Array2<F>,
        configuration: SKMiniBatchKMeansConfiguration<F>,
    ) -> Self {
        let number_of_clusters = centers.nrows();
        Self {
            centers,
            counts: Array1::zeros(number_of_clusters),
            batches_processed: 0,
            converged: false,
            last_batch_inertia: F::zero(),
            configuration,
        }
    }
}
