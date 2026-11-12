//! Mutable batch-accumulation state shared across `fit`, `partial_fit` and
//! `fit_streaming` (spec `mini-batch-k-means`, decisions 3–6).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::assign::sk_assign_batch_rows;
use super::config::SKMiniBatchKMeansConfiguration;
use super::reassign::sk_reassign_low_count_centers;
use super::update::sk_update_centers_and_counts;

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

    /// Processes one validated mini-batch: parallel assignment, per-center
    /// learning-rate update, deterministic low-count reassignment and
    /// convergence check. Returns `true` when positive-tolerance convergence is
    /// reached.
    pub(crate) fn apply_batch(&mut self, features: SKDataView<'_, F>) -> Result<bool, SKError> {
        let (assignments, squared_distances) =
            sk_assign_batch_rows(features, &self.centers.view())?;
        self.last_batch_inertia = squared_distances
            .iter()
            .fold(F::zero(), |accumulator, &value| accumulator + value);

        let maximum_shift = sk_update_centers_and_counts(
            &mut self.centers,
            &mut self.counts,
            features,
            &assignments,
        )?;
        let reassigned = sk_reassign_low_count_centers(
            &mut self.centers,
            &mut self.counts,
            features,
            &assignments,
            self.configuration.reassignment_ratio,
        )?;

        self.batches_processed += 1;
        self.converged = self.configuration.tolerance > F::zero()
            && reassigned == 0
            && maximum_shift <= self.configuration.tolerance;
        Ok(self.converged)
    }
}
