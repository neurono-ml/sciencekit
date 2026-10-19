//! Incremental learning: one more batch fed into the per-column sketches.

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    observability::sk_run_operation,
};

use super::core_implementation::SKRobustScalerModel;
use crate::robust_scaler::core_implementation::finalize_statistics;
use crate::robust_scaler::sketch::{extend_sketches, sketch_statistics};

impl<F: SKFloat> SKRobustScalerModel<F> {
    /// Continue learning from one more batch, feeding it into the per-column
    /// online sketches and refreshing the center and scale from their quantiles.
    /// Returns the model itself for chaining. An empty batch is a no-op.
    pub fn partial_fit<'a, X>(&mut self, features: X) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let dense = features.try_into()?.as_dense()?;
        if dense.ncols() != self.center.len() {
            return Err(SKError::ShapeMismatch {
                expected: vec![self.center.len()],
                found: vec![dense.ncols()],
            });
        }
        if dense.nrows() == 0 {
            return Ok(self);
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::PartialFit,
            rows_count: dense.nrows(),
            columns_count: dense.ncols(),
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || -> Result<(), SKError> {
            extend_sketches(&mut self.sketches, &dense);
            let (raw_center, raw_scale) = sketch_statistics(&self.sketches, &self.configuration);
            let (center, scale) = finalize_statistics(&self.configuration, &raw_center, &raw_scale);
            self.center = center;
            self.scale = scale;
            self.sample_count += dense.nrows() as u64;
            Ok(())
        })?;
        Ok(self)
    }
}
