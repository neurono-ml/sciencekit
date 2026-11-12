//! Incremental learning: one more batch fed into the per-column sketches.

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    observability::sk_run_operation,
};

use super::core_implementation::SKRobustScalerModel;
use crate::robust_scaler::configuration::finalize_statistics;
use crate::robust_scaler::sketch::{extend_sketches, extend_sparse_sketches, sketch_statistics};

impl<F: SKFloat> SKRobustScalerModel<F> {
    /// Continue learning from one more batch, feeding it into the per-column
    /// online sketches and refreshing the center and scale from their quantiles.
    /// Returns the model itself for chaining. An empty batch is a no-op.
    pub fn partial_fit<'a, X>(&mut self, features: X) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'_, F> = features.try_into()?;
        let (rows_count, columns_count) = match &view {
            SKDataView::Dense(dense) => dense.dim(),
            SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the features to a dense matrix or a CSR matrix",
                });
            }
        };
        if columns_count != self.center.len() {
            return Err(SKError::ShapeMismatch {
                expected: vec![self.center.len()],
                found: vec![columns_count],
            });
        }
        if rows_count == 0 {
            return Ok(self);
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::PartialFit,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || -> Result<(), SKError> {
            match &view {
                SKDataView::Dense(dense) => extend_sketches(&mut self.sketches, dense),
                SKDataView::Sparse(sparse) => extend_sparse_sketches(&mut self.sketches, sparse),
                _ => unreachable!("the representation check rejected every other variant"),
            }
            let (raw_center, raw_scale) = sketch_statistics(&self.sketches, &self.configuration);
            let (center, scale) = finalize_statistics(&self.configuration, &raw_center, &raw_scale);
            self.center = center;
            self.scale = scale;
            self.sample_count += rows_count as u64;
            Ok(())
        })?;
        Ok(self)
    }
}
