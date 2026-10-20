//! Incremental learning: one more batch merged into the running state
//! (spec `simple-imputer`).
//!
//! The state seeded by `fit` accumulates the batch and the statistics and the
//! empty-column policy are refreshed from it, so a model learned batch by
//! batch answers like one fitted on the concatenation.

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, sk_run_operation,
};

use super::SKSimpleImputerModel;
use crate::simple_imputer::core_implementation::empty_columns;

impl<F: SKFloat> SKSimpleImputerModel<F> {
    /// Continue learning from one more batch, updating the accumulated state
    /// and refreshing the statistics. Returns the model itself for chaining.
    /// An empty batch is a no-op.
    pub fn partial_fit<'a, X>(&mut self, features: X) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        match features.try_into()? {
            SKDataView::Dense(dense) => {
                self.validate_batch(dense.nrows(), dense.ncols())?;
                if dense.nrows() == 0 {
                    return Ok(self);
                }
                let attributes = self.batch_attributes(dense.nrows(), dense.ncols());
                sk_run_operation(attributes, || -> Result<(), SKError> {
                    self.state
                        .extend_dense(&dense, self.strategy, self.missing_values);
                    Ok(())
                })?;
            }
            SKDataView::Sparse(sparse) => {
                self.validate_batch(sparse.rows(), sparse.cols())?;
                if sparse.rows() == 0 {
                    return Ok(self);
                }
                let attributes = self.batch_attributes(sparse.rows(), sparse.cols());
                sk_run_operation(attributes, || -> Result<(), SKError> {
                    self.state
                        .extend_sparse(&sparse, self.strategy, self.missing_values);
                    Ok(())
                })?;
            }
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the batch to a dense or CSR matrix",
                });
            }
        }
        self.refresh_statistics();
        Ok(self)
    }

    /// Reject a batch whose column count differs from the fitted one before
    /// any element is read.
    fn validate_batch(&self, rows_count: usize, columns_count: usize) -> Result<(), SKError> {
        if columns_count == self.number_of_input_columns {
            return Ok(());
        }
        Err(SKError::shape_mismatch_2d(
            rows_count,
            self.number_of_input_columns,
            rows_count,
            columns_count,
        ))
    }

    /// The span attributes of one incremental batch, carrying the carried
    /// execution mode and backend kind.
    fn batch_attributes(&self, rows_count: usize, columns_count: usize) -> SKOperationAttributes {
        SKOperationAttributes {
            operation: SKOperationKind::PartialFit,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        }
    }

    /// Refresh the statistics, the empty-column policy and the composed
    /// indicator's scope from the accumulated state.
    fn refresh_statistics(&mut self) {
        self.statistics = self.state.statistics(self.strategy, self.fill_value);
        self.dropped_columns = empty_columns(
            &self.statistics,
            self.state.counts(),
            self.keep_empty_features,
        );
        if let Some(indicator) = self.indicator.as_mut() {
            let scope: Vec<usize> = self
                .state
                .missing_columns()
                .iter()
                .enumerate()
                .filter_map(|(column, missing)| missing.then_some(column))
                .collect();
            *indicator = indicator.with_scope(scope);
        }
    }
}
