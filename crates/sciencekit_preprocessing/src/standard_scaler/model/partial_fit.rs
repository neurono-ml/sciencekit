//! Incremental learning: one more batch merged into the running moments.

use sciencekit_common::{
    SKDataView, SKError, SKExecutionContext, SKFloat, SKOperationAttributes, SKOperationKind,
    observability::sk_run_operation,
};

use super::super::streaming::batches::{batch_moments, batch_parallelism};
use super::super::streaming::state::merge_moments;
use super::{SKStandardScalerModel, derive_statistics};

impl<F: SKFloat> SKStandardScalerModel<F> {
    /// Continue learning from one more batch, updating the running moments
    /// with the same Chan merge the streaming driver uses. Returns the model
    /// itself for chaining. An empty batch is a no-op.
    pub fn partial_fit<'a, X>(&mut self, features: X) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let dense = features.try_into()?.as_dense()?;
        if dense.ncols() != self.moments_mean.len() {
            return Err(SKError::ShapeMismatch {
                expected: vec![self.moments_mean.len()],
                found: vec![dense.ncols()],
            });
        }
        if dense.nrows() == 0 {
            return Ok(self);
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::PartialFit,
            rows: dense.nrows(),
            columns: dense.ncols(),
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || -> Result<(), SKError> {
            let machine = SKExecutionContext::real();
            let parallelism =
                batch_parallelism(&dense, machine.available_memory_bytes, machine.cpu_cores);
            let (batch_mean, batch_deviations) = batch_moments(&dense, parallelism);
            merge_moments(
                &mut self.sample_count,
                &mut self.moments_mean,
                &mut self.sum_squared_deviations,
                &batch_mean,
                &batch_deviations,
                dense.nrows() as u64,
            );
            let (mean, scale) = derive_statistics(
                &self.moments_mean,
                &self.sum_squared_deviations,
                self.sample_count,
                self.with_mean,
                self.with_std,
            );
            self.mean = mean;
            self.scale = scale;
            Ok(())
        })?;
        Ok(self)
    }
}
