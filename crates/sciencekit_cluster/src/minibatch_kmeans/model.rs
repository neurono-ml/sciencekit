//! The fitted MiniBatchKMeans model (spec `mini-batch-k-means`, decision 1).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKBackendKind, SKClusterLabel, SKExecutionMode, SKFloat};

use super::config::SKMiniBatchKMeansConfiguration;
use super::state::SKMiniBatchState;

/// Fitted MiniBatchKMeans state: dense centers, per-center counts, optional
/// in-memory training assignments, inertia, batch/epoch progress and metadata.
#[derive(Debug, Clone)]
pub struct SKMiniBatchKMeansModel<F: SKFloat> {
    pub(super) centers: Array2<F>,
    pub(super) counts: Array1<usize>,
    pub(super) labels: Option<Array1<SKClusterLabel>>,
    pub(super) inertia: F,
    pub(super) batches_processed: usize,
    pub(super) epochs_consumed: usize,
    pub(super) converged: bool,
    pub(super) configuration: SKMiniBatchKMeansConfiguration<F>,
}

impl<F: SKFloat> SKMiniBatchKMeansModel<F> {
    /// Builds the public model from an accumulated [`SKMiniBatchState`].
    pub(super) fn from_state(
        state: SKMiniBatchState<F>,
        labels: Option<Array1<SKClusterLabel>>,
        inertia: F,
        epochs_consumed: usize,
    ) -> Self {
        Self {
            centers: state.centers,
            counts: state.counts,
            labels,
            inertia,
            batches_processed: state.batches_processed,
            epochs_consumed,
            converged: state.converged,
            configuration: state.configuration,
        }
    }

    /// Cluster centers, one row per cluster.
    pub fn cluster_centers(&self) -> ArrayView2<'_, F> {
        self.centers.view()
    }

    /// Cumulative number of samples assigned to each cluster center.
    pub fn counts(&self) -> ArrayView1<'_, usize> {
        self.counts.view()
    }

    /// Typed training assignments when fitted in memory (`None` for streaming
    /// fits or after incremental `partial_fit` updates). Never contains `Noise`.
    pub fn labels(&self) -> Option<ArrayView1<'_, SKClusterLabel>> {
        self.labels.as_ref().map(|labels| labels.view())
    }

    /// Sum of squared distances to assigned centers (exact over training data
    /// for `fit`, over the last pass/batch for streaming and incremental fits).
    pub fn inertia(&self) -> F {
        self.inertia
    }

    /// Total mini-batches processed across all passes and `partial_fit` calls.
    pub fn batches_processed(&self) -> usize {
        self.batches_processed
    }

    /// Streaming passes or in-memory epochs consumed.
    pub fn epochs_consumed(&self) -> usize {
        self.epochs_consumed
    }

    /// Whether positive-tolerance center movement met the convergence criterion.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Number of clusters in the model.
    pub fn number_of_clusters(&self) -> usize {
        self.centers.nrows()
    }

    /// Number of features the model was fitted on.
    pub fn number_of_features(&self) -> usize {
        self.centers.ncols()
    }

    /// Backend kind recorded by the producing estimator and model operations.
    pub fn backend(&self) -> SKBackendKind {
        self.configuration.backend
    }

    /// Execution intent carried from the estimator.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.configuration.execution_mode
    }
}
