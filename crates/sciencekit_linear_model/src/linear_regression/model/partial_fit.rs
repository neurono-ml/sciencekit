//! Incremental learning: one more batch merged into the running Gram state
//! (spec `linear-regression`).

use sciencekit_common::{
    SKDataView, SKError, SKExecutionContext, SKFloat, SKOperationAttributes, SKOperationKind,
    SKTargetView, sk_run_operation,
};
use sciencekit_math::{SKFaerBackend, SKMathBackend, SKMatrixMultiplyBackend};

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
        SKFaerBackend: SKMathBackend<F>,
        SKMatrixMultiplyBackend: SKMathBackend<F>,
    {
        let dense = features.try_into()?.as_dense()?;
        let target = targets.try_into()?.as_continuous()?.into_owned();
        let expected_columns = self.gram.design_width - usize::from(self.include_intercept);
        if dense.ncols() != expected_columns {
            return Err(SKError::ShapeMismatch {
                expected: vec![expected_columns],
                found: vec![dense.ncols()],
            });
        }
        if dense.nrows() == 0 {
            return Ok(self);
        }
        if target.len() != dense.nrows() {
            return Err(SKError::ShapeMismatch {
                expected: vec![dense.nrows()],
                found: vec![target.len()],
            });
        }
        let backend = self.backend;
        let include_intercept = self.include_intercept;
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::PartialFit,
            rows_count: dense.nrows(),
            columns_count: dense.ncols(),
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || -> Result<(), SKError> {
            let machine = SKExecutionContext::real();
            let parallelism =
                batch_parallelism(&dense, machine.available_memory_bytes, machine.cpu_cores);
            let design = build_design(&dense, include_intercept);
            self.gram
                .accumulate(&design.view(), &target.view(), parallelism);
            let solution =
                solve_least_squares(backend, self.gram.gram.view(), self.gram.rhs.view())?;
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
