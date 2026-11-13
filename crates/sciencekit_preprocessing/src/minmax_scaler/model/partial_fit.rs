//! Incremental learning: one more batch merged into the running extrema.

use sciencekit_common::{
    SKDataView, SKError, SKExecutionContext, SKFloat, SKOperationAttributes, SKOperationKind,
    observability::sk_run_operation,
};

use super::super::streaming::batches::{batch_extrema, batch_parallelism, sparse_batch_extrema};
use super::super::streaming::state::merge_extrema;
use super::core_implementation::{SKMinMaxScalerModel, derive_scale};

impl<F: SKFloat> SKMinMaxScalerModel<F> {
    /// Continue learning from one more batch, updating the running minimum and
    /// maximum with the same merge the streaming driver uses. Returns the model
    /// itself for chaining. An empty batch is a no-op.
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
        if columns_count != self.minimum.len() {
            return Err(SKError::ShapeMismatch {
                expected: vec![self.minimum.len()],
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
            let (batch_minimum, batch_maximum) = match &view {
                SKDataView::Dense(dense) => {
                    let machine = SKExecutionContext::real();
                    let parallelism =
                        batch_parallelism(dense, machine.available_memory_bytes, machine.cpu_cores);
                    batch_extrema(dense, parallelism)
                }
                SKDataView::Sparse(sparse) => sparse_batch_extrema(sparse),
                _ => unreachable!("the representation check rejected every other variant"),
            };
            merge_extrema(
                &mut self.minimum,
                &mut self.maximum,
                &mut self.data_range,
                &batch_minimum,
                &batch_maximum,
            );
            self.scale = derive_scale(&self.data_range, self.feature_minimum, self.feature_maximum);
            self.sample_count += rows_count as u64;
            Ok(())
        })?;
        Ok(self)
    }
}
