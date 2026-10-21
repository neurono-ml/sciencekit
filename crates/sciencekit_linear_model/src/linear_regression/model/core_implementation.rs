//! The fitted linear-regression model, its accessors and its prediction
//! (spec `linear-regression`).

use ndarray::{Array1, Array2, s};
use sciencekit_common::{
    SKBackendKind, SKDataView, SKError, SKExecutionMode, SKFloat, SKOperationAttributes,
    SKOperationKind, SKRegressorPredictor, sk_run_operation,
};
use sciencekit_math::SKBackendSelection;
use sciencekit_math::SKLeastSquaresSolution;

use crate::linear_regression::gram::SKGramState;
use crate::linear_regression::solving::solve_least_squares;

/// Fitted ordinary-least-squares model: the coefficients, the intercept, the
/// solve's rank and singular values, and the running cross-product state.
///
/// Distinct type from the estimator (`SKLinearRegression`): the estimator is
/// immutable configuration, the model is the learned state and the export
/// boundary. The model is `Send + Sync` by construction, so one model serves
/// concurrent predictions.
#[derive(Debug, Clone)]
pub struct SKLinearRegressionModel<F> {
    pub(super) coefficients: Array1<F>,
    pub(super) intercept: F,
    pub(super) rank: usize,
    pub(super) singular_values: Vec<F>,
    pub(super) gram: SKGramState<F>,
    pub(super) include_intercept: bool,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKLinearRegressionModel<F> {
    /// Build a fitted model from a dense least-squares solution and the design's
    /// Gram state. Crate-internal: models are produced by `fit`/`fit_streaming`,
    /// never constructed directly by consumers.
    pub(crate) fn from_solution(
        solution: SKLeastSquaresSolution<F>,
        gram: SKGramState<F>,
        include_intercept: bool,
        backend: SKBackendKind,
        execution_intent: SKExecutionMode,
    ) -> Self {
        let (intercept, coefficients) = split_solution(&solution.solution, include_intercept);
        Self {
            coefficients,
            intercept,
            rank: solution.rank,
            singular_values: solution.singular_values,
            gram,
            include_intercept,
            backend,
            execution_intent,
        }
    }

    /// Build a fitted model by solving the running Gram state (the streaming and
    /// incremental regimes).
    pub(crate) fn from_gram(
        gram: SKGramState<F>,
        include_intercept: bool,
        backend: SKBackendKind,
        execution_intent: SKExecutionMode,
    ) -> Result<Self, SKError>
    where
        F: SKBackendSelection,
    {
        let solution = solve_least_squares(backend, gram.gram.view(), gram.right_hand_side.view())?;
        Ok(Self::from_solution(
            solution,
            gram,
            include_intercept,
            backend,
            execution_intent,
        ))
    }

    /// The learned coefficients (one per feature).
    pub fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    /// The learned intercept (zero when `include_intercept` is false).
    pub fn intercept(&self) -> F {
        self.intercept
    }

    /// The effective rank of the matrix the last solve factored, after
    /// truncation.
    ///
    /// A densely fitted model reports the rank of the design matrix `[1 ‖ X]`;
    /// a model reached through `fit_streaming` or `partial_fit` reports the rank
    /// of the accumulated Gram matrix `XᵀX` (whose singular values are the
    /// squares of the design's), because those regimes solve `XᵀX β = Xᵀy`.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The singular values of the matrix the last solve factored, non-increasing.
    ///
    /// Dense `fit` surfaces the design matrix's singular values; the streaming
    /// and incremental regimes surface the Gram matrix's (≈ the squares of the
    /// design's), for the same reason as [`SKLinearRegressionModel::rank`].
    pub fn singular_values(&self) -> &[F] {
        &self.singular_values
    }

    /// The number of rows learned from, across `fit` and every `partial_fit`.
    pub fn row_count(&self) -> u64 {
        self.gram.row_count
    }

    /// The backend kind carried from the estimator that produced this model.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }
}

/// Split a least-squares solution column into the intercept and the feature
/// coefficients, honoring `include_intercept`.
pub(crate) fn split_solution<F: SKFloat>(
    solution: &Array2<F>,
    include_intercept: bool,
) -> (F, Array1<F>) {
    let column = solution.column(0);
    if include_intercept {
        (column[0], column.slice(s![1..]).to_owned())
    } else {
        (F::zero(), column.to_owned())
    }
}

impl<F: SKFloat> SKRegressorPredictor<F> for SKLinearRegressionModel<F> {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let dense = features.try_into()?.as_dense()?;
        if dense.ncols() != self.coefficients.len() {
            return Err(SKError::shape_mismatch_2d(
                dense.nrows(),
                self.coefficients.len(),
                dense.nrows(),
                dense.ncols(),
            ));
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count: dense.nrows(),
            columns_count: dense.ncols(),
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            Ok(dense
                .dot(&self.coefficients)
                .mapv(|value| value + self.intercept))
        })
    }
}
