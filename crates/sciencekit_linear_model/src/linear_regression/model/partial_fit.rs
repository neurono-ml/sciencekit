//! Incremental learning: one more batch merged into the running Gram state
//! (spec `linear-regression`).

use sciencekit_common::{
    SKDataView, SKError, SKExecutionContext, SKFloat, SKOperationAttributes, SKOperationKind,
    SKTargetView, sk_run_operation,
};
use sciencekit_math::SKBackendSelection;

use super::SKLinearRegressionModel;
use super::split_solution;
use crate::linear_regression::core_implementation::build_design;
use crate::linear_regression::solving::solve_least_squares;
use crate::linear_regression::streaming::batches::batch_parallelism;

impl<F: SKFloat> SKLinearRegressionModel<F> {
    /// Continue learning from one more batch, merging its cross products into
    /// the running Gram state and re-solving. Returns the model itself for
    /// chaining. An empty batch is a no-op.
    pub fn partial_fit<'a, X, T>(&mut self, features: X, targets: T) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
        F: SKBackendSelection,
    {
        let view: SKDataView<'_, F> = features.try_into()?;
        let target = targets.try_into()?.as_continuous()?.into_owned();
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
        let expected_columns = self.gram.design_width - usize::from(self.include_intercept);
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
        let backend = self.backend;
        let include_intercept = self.include_intercept;
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::PartialFit,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || -> Result<(), SKError> {
            match &view {
                SKDataView::Dense(dense) => {
                    let machine = SKExecutionContext::real();
                    let parallelism =
                        batch_parallelism(dense, machine.available_memory_bytes, machine.cpu_cores);
                    let design = build_design(dense, include_intercept);
                    self.gram
                        .accumulate(&design.view(), &target.view(), parallelism);
                }
                SKDataView::Sparse(sparse) => {
                    self.gram
                        .accumulate_sparse(sparse, &target.view(), include_intercept);
                }
                _ => unreachable!("the representation check rejected every other variant"),
            }
            let solution = solve_least_squares(
                backend,
                self.gram.gram.view(),
                self.gram.right_hand_side.view(),
            )?;
            let (intercept, coefficients) = split_solution(&solution.solution, include_intercept);
            self.intercept = intercept;
            self.coefficients = coefficients;
            self.rank = solution.rank;
            self.singular_values = solution.singular_values;
            Ok(())
        })?;
        Ok(self)
    }
}
