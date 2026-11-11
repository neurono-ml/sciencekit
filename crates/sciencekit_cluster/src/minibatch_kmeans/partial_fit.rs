//! Incremental batch continuation on [`SKMiniBatchKMeansModel`] (spec
//! `mini-batch-k-means`, decision 5).

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, sk_run_operation,
};

use super::model::SKMiniBatchKMeansModel;
use super::state::SKMiniBatchState;
use super::validation::sk_validate_continuation_batch;
use crate::kmeans::distance::sk_view_dimensions;

impl<F: SKFloat> SKMiniBatchKMeansModel<F> {
    /// Updates centers, per-center counts and convergence progress from one
    /// additional mini-batch, returning the model for chaining.
    pub fn partial_fit<'a, X>(&mut self, features: X) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::PartialFit,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode,
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || -> Result<(), SKError> {
            sk_validate_continuation_batch(
                features,
                self.number_of_features(),
                self.number_of_clusters(),
                true,
                "partial_fit",
            )?;
            let mut state = SKMiniBatchState {
                centers: self.centers.clone(),
                counts: self.counts.clone(),
                batches_processed: self.batches_processed,
                converged: self.converged,
                last_batch_inertia: self.inertia,
                configuration: self.configuration,
            };
            state.apply_batch(features)?;
            self.centers = state.centers;
            self.counts = state.counts;
            self.batches_processed = state.batches_processed;
            self.converged = state.converged;
            self.inertia = state.last_batch_inertia;
            self.labels = None;
            Ok(())
        })?;
        Ok(self)
    }
}
