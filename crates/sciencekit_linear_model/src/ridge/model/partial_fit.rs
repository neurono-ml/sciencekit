//! Incremental learning: one more batch merged into the running centered Gram
//! and re-solved (spec `ridge-regression`).

use sciencekit_common::{
    SKDataView, SKError, SKExecutionContext, SKOperationAttributes, SKOperationKind, SKTargetView,
    sk_run_operation,
};
use sciencekit_math::SKBackendSelection;

use super::SKRidgeModel;
use crate::ridge::builder::SKRidgeSolver;
use crate::ridge::solving::solve_from_gram;
use crate::ridge::streaming::batches::batch_parallelism;

impl<F: SKBackendSelection> SKRidgeModel<F> {
    /// Continue learning from one more batch, merging its centered cross
    /// products into the running Gram state and re-solving the accumulated
    /// normal equations. Returns the model itself for chaining. An empty batch
    /// is a no-op; a column count differing from the fitted design is rejected
    /// with `ShapeMismatch`. Dense and sparse batches are both accepted.
    ///
    /// A model fitted through the `SparseCg` path never materialized `XᵀX`, so
    /// it cannot be continued incrementally: the update returns a structured
    /// representation error pointing at the `Cholesky` solver rather than
    /// solving a partial system.
    pub fn partial_fit<'a, X, T>(&mut self, features: X, targets: T) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let target = targets.try_into()?.as_continuous()?.into_owned();
        let (rows_count, columns_count) = match data {
            SKDataView::Dense(dense) => dense.dim(),
            SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
            _ => return Err(unsupported_representation()),
        };
        let expected_columns = self.feature_count();
        if columns_count != expected_columns {
            return Err(SKError::ShapeMismatch {
                expected: vec![expected_columns],
                found: vec![columns_count],
            });
        }
        if rows_count == 0 {
            return Ok(self);
        }
        if target.len() != rows_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![rows_count],
                found: vec![target.len()],
            });
        }
        if !self.gram.has_full_gram {
            return Err(SKError::UnsupportedRepresentation {
                representation: "csr",
                suggestion: "the model was fitted without materializing XᵀX (SparseCg); refit with the Cholesky solver to enable incremental updates",
            });
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::PartialFit,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || -> Result<(), SKError> {
            match data {
                SKDataView::Dense(dense) => {
                    let machine = SKExecutionContext::real();
                    let parallelism = batch_parallelism(
                        &dense,
                        machine.available_memory_bytes,
                        machine.cpu_cores,
                    );
                    self.gram
                        .accumulate_dense(&dense, &target.view(), parallelism);
                }
                SKDataView::Sparse(sparse) => {
                    self.gram.accumulate_sparse(&sparse, &target.view());
                }
                _ => return Err(unsupported_representation()),
            }
            let solution = solve_from_gram(
                &self.gram,
                self.include_intercept,
                &self.alphas,
                self.backend,
            )?;
            self.coefficients = solution.coefficients;
            self.intercepts = solution.intercepts;
            self.rank = solution.rank;
            self.singular_values = Vec::new();
            self.solver = SKRidgeSolver::Cholesky;
            Ok(())
        })?;
        Ok(self)
    }
}

/// The error for a feature representation the incremental update cannot consume.
fn unsupported_representation() -> SKError {
    SKError::UnsupportedRepresentation {
        representation: "unknown",
        suggestion: "convert the features to a dense matrix or a CSR matrix",
    }
}
